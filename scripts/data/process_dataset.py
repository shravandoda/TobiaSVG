import argparse
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import logging
from pathlib import Path
import shutil
from typing import Any

from datasets import Dataset, Features, IterableDataset, Value, concatenate_datasets
from datasets import load_from_disk

from scripts.data.process_svg import (
    SvgQualityError,
    TextLabelError,
    build_label_prompt,
    clean_validate_and_render_svg,
    get_default_label_model,
    request_text_label,
)
from scripts.data.sample_dataset import (
    DATASETS,
    DatasetSpec,
    sample_dataset,
)
from scripts.data.utils import (
    build_image_path,
    build_split_checkpoint_path,
    build_split_output_path,
    get_source_filename,
    get_source_svg,
    get_source_text,
    load_env_file,
)
from src.project_x.utils.logger import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

LOG_EVERY = 500
DEFAULT_CAPTION_STYLE = "concise"
FINAL_COLUMNS = ["filename", "svg", "text", "caption_style", "dataset"]
CHECKPOINT_COLUMNS = [
    "row_index",
    "filename",
    "svg",
    "text",
    "caption_style",
    "dataset",
    "keep",
    "error",
]
FINAL_FEATURES = Features(
    {
        "filename": Value("string"),
        "svg": Value("string"),
        "text": Value("string"),
        "caption_style": Value("string"),
        "dataset": Value("string"),
    }
)
CHECKPOINT_FEATURES = Features(
    {
        "row_index": Value("int64"),
        "filename": Value("string"),
        "svg": Value("string"),
        "text": Value("string"),
        "caption_style": Value("string"),
        "dataset": Value("string"),
        "keep": Value("bool"),
        "error": Value("string"),
    }
)


def process_row(
    raw_row: dict[str, Any],
    row_index: int,
    spec: DatasetSpec,
    split_name: str,
    label_provider: str,
    label_model: str,
) -> dict[str, Any]:
    filename = get_source_filename(
        raw_row,
        spec,
        split_name=split_name,
        row_index=row_index,
    )
    raw_svg = get_source_svg(raw_row, spec)
    source_text = get_source_text(raw_row, spec)
    image_path = build_image_path(
        spec=spec,
        split_name=split_name,
        filename=filename,
    )

    try:
        svg = clean_validate_and_render_svg(
            raw_svg,
            image_path,
            validate_quality=spec.validate_svg_quality,
        )
        row: dict[str, str] = {"filename": filename, "svg": svg}
        if source_text is None:
            prompt = build_label_prompt(row, caption_style=spec.caption_style)
            text = request_text_label(
                image_path,
                prompt=prompt,
                label_provider=label_provider,
                model=label_model,
            )
        else:
            text = source_text

        return {
            "filename": filename,
            "svg": svg,
            "text": text,
            "caption_style": spec.caption_style,
            "dataset": spec.key,
            "keep": True,
            "error": None,
        }
    except TextLabelError as exc:
        logger.info("filtered: %s (%s)", filename, exc)
        return {
            "filename": filename,
            "svg": raw_svg,
            "text": None,
            "caption_style": spec.caption_style,
            "dataset": spec.key,
            "keep": False,
            "error": str(exc),
        }
    except SvgQualityError as exc:
        logger.info("filtered: %s (%s)", filename, exc)
        return {
            "filename": filename,
            "svg": raw_svg,
            "text": None,
            "caption_style": spec.caption_style,
            "dataset": spec.key,
            "keep": False,
            "error": str(exc),
        }


def final_row(row: dict[str, Any]) -> dict[str, Any]:
    return {column_name: row[column_name] for column_name in FINAL_COLUMNS}


def checkpoint_row(row_index: int, row: dict[str, Any]) -> dict[str, Any]:
    return {
        "row_index": row_index,
        "filename": row["filename"],
        "svg": row["svg"],
        "text": row["text"],
        "caption_style": row.get("caption_style", DEFAULT_CAPTION_STYLE),
        "dataset": row["dataset"],
        "keep": row["keep"],
        "error": row["error"],
    }


def iter_checkpoint_rows_sequential(
    dataset: IterableDataset,
    spec: DatasetSpec,
    split_name: str,
    start_index: int,
    label_provider: str,
    label_model: str,
):
    for row_index, raw_row in enumerate(dataset, start=start_index):
        row = process_row(
            raw_row,
            row_index=row_index,
            spec=spec,
            split_name=split_name,
            label_provider=label_provider,
            label_model=label_model,
        )
        yield checkpoint_row(row_index, row)


def iter_checkpoint_rows_parallel(
    dataset: IterableDataset,
    spec: DatasetSpec,
    split_name: str,
    start_index: int,
    label_workers: int,
    max_in_flight: int,
    label_provider: str,
    label_model: str,
):
    row_iterator = enumerate(dataset, start=start_index)
    pending: dict[int, Future[dict[str, Any]]] = {}
    completed: dict[int, dict[str, Any]] = {}
    next_output_index = start_index

    def submit_next(executor: ThreadPoolExecutor) -> bool:
        try:
            row_index, raw_row = next(row_iterator)
        except StopIteration:
            return False

        pending[row_index] = executor.submit(
            process_row,
            raw_row,
            row_index,
            spec,
            split_name,
            label_provider,
            label_model,
        )
        return True

    with ThreadPoolExecutor(max_workers=label_workers) as executor:
        for _ in range(max_in_flight):
            if not submit_next(executor):
                break

        while pending or completed:
            while next_output_index in completed:
                row = completed.pop(next_output_index)
                yield checkpoint_row(next_output_index, row)
                next_output_index += 1

                while len(pending) < max_in_flight:
                    if not submit_next(executor):
                        break

            if not pending:
                continue

            done, _ = wait(set(pending.values()), return_when=FIRST_COMPLETED)

            for future in done:
                row_index = next(
                    index
                    for index, pending_future in pending.items()
                    if pending_future is future
                )
                del pending[row_index]
                completed[row_index] = future.result()

            while len(pending) < max_in_flight:
                if not submit_next(executor):
                    break


def iter_checkpoint_rows(
    dataset: IterableDataset,
    spec: DatasetSpec,
    split_name: str,
    start_index: int,
    label_workers: int,
    max_in_flight: int,
    label_provider: str,
    label_model: str,
):
    if label_workers <= 1:
        yield from iter_checkpoint_rows_sequential(
            dataset=dataset,
            spec=spec,
            split_name=split_name,
            start_index=start_index,
            label_provider=label_provider,
            label_model=label_model,
        )
        return

    yield from iter_checkpoint_rows_parallel(
        dataset=dataset,
        spec=spec,
        split_name=split_name,
        start_index=start_index,
        label_workers=label_workers,
        max_in_flight=max_in_flight,
        label_provider=label_provider,
        label_model=label_model,
    )


def list_checkpoint_chunks(checkpoint_path: Path) -> list[Path]:
    if not checkpoint_path.exists():
        return []

    return sorted(
        path
        for path in checkpoint_path.iterdir()
        if path.is_dir() and path.name.startswith("chunk_")
    )


def load_checkpoint_chunk(chunk_path: Path) -> Dataset:
    dataset = load_from_disk(str(chunk_path))
    if not isinstance(dataset, Dataset):
        raise TypeError(f"Expected checkpoint chunk to be a Dataset: {chunk_path}")

    dataset_key = chunk_path.parent.parent.name

    if "caption_style" not in dataset.column_names:
        dataset = dataset.add_column(
            "caption_style",
            [DEFAULT_CAPTION_STYLE] * len(dataset),
        )

    if "dataset" not in dataset.column_names:
        dataset = dataset.add_column("dataset", [dataset_key] * len(dataset))

    return dataset.select_columns(CHECKPOINT_COLUMNS)


def get_checkpoint_state(checkpoint_path: Path) -> tuple[int, int, int]:
    chunks = list_checkpoint_chunks(checkpoint_path)
    row_count = 0
    kept_count = 0

    for chunk_path in chunks:
        dataset = load_checkpoint_chunk(chunk_path)
        row_count += len(dataset)
        kept_count += sum(bool(value) for value in dataset["keep"])

    return len(chunks), row_count, kept_count


def save_checkpoint_chunk(rows: list[dict[str, Any]], chunk_path: Path) -> None:
    if chunk_path.exists():
        raise FileExistsError(f"Checkpoint chunk already exists: {chunk_path}")

    chunk_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = chunk_path.with_name(f".{chunk_path.name}.tmp")
    if temp_path.exists():
        shutil.rmtree(temp_path)

    dataset = Dataset.from_list(rows, features=CHECKPOINT_FEATURES)
    dataset.save_to_disk(str(temp_path))
    temp_path.replace(chunk_path)


def build_final_dataset_from_checkpoints(
    checkpoint_path: Path,
    target_size: int,
) -> Dataset:
    chunks = list_checkpoint_chunks(checkpoint_path)
    if not chunks:
        return Dataset.from_dict(
            {column_name: [] for column_name in FINAL_COLUMNS},
            features=FINAL_FEATURES,
        )

    dataset = concatenate_datasets(
        [load_checkpoint_chunk(chunk_path) for chunk_path in chunks]
    )
    dataset = dataset.filter(lambda row: row["keep"])
    dataset = dataset.select_columns(FINAL_COLUMNS)

    if len(dataset) > target_size:
        dataset = dataset.select(range(target_size))

    return dataset.cast(FINAL_FEATURES)


def process_split(
    dataset: IterableDataset,
    spec: DatasetSpec,
    split_name: str,
    label_workers: int,
    max_in_flight: int,
    chunk_size: int,
    resume: bool,
    label_provider: str,
    label_model: str,
) -> Dataset:
    checkpoint_path = build_split_checkpoint_path(spec=spec, split_name=split_name)
    target_size = spec.splits[split_name].target_size
    existing_chunk_count, processed_count, kept_count = get_checkpoint_state(
        checkpoint_path
    )

    if existing_chunk_count and not resume:
        raise ValueError(
            f"Checkpoint chunks already exist at {checkpoint_path}. "
            "Pass --resume to continue from them."
        )

    if processed_count:
        logger.info(
            "resuming %s/%s from %s processed rows (%s kept)",
            spec.key,
            split_name,
            processed_count,
            kept_count,
        )
        dataset = dataset.skip(processed_count)

    chunk_rows: list[dict[str, Any]] = []
    chunk_index = existing_chunk_count

    for row in iter_checkpoint_rows(
        dataset=dataset,
        spec=spec,
        split_name=split_name,
        start_index=processed_count,
        label_workers=label_workers,
        max_in_flight=max_in_flight,
        label_provider=label_provider,
        label_model=label_model,
    ):
        chunk_rows.append(row)
        if row["keep"]:
            kept_count += 1

        if len(chunk_rows) >= chunk_size:
            chunk_path = checkpoint_path / f"chunk_{chunk_index:06d}"
            save_checkpoint_chunk(chunk_rows, chunk_path)
            logger.info("saved checkpoint: %s", chunk_path)
            chunk_rows = []
            chunk_index += 1

        if kept_count >= target_size:
            break

    if chunk_rows:
        chunk_path = checkpoint_path / f"chunk_{chunk_index:06d}"
        save_checkpoint_chunk(chunk_rows, chunk_path)
        logger.info("saved checkpoint: %s", chunk_path)

    return build_final_dataset_from_checkpoints(
        checkpoint_path=checkpoint_path,
        target_size=target_size,
    )


def save_processed_dataset(dataset: Dataset, output_path: Path) -> None:
    """Persist one processed split to disk.

    HF datasets learning checkpoint:
    - create the output directory if needed
    - save the Dataset so it can be loaded later with load_from_disk
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output_path))


def process_dataset(
    spec: DatasetSpec,
    seed: int,
    buffer_size: int,
    sample_size: int | None,
    streaming: bool,
    label_workers: int,
    max_in_flight: int,
    chunk_size: int,
    resume: bool,
    label_provider: str,
    label_model: str,
) -> None:
    sampled = sample_dataset(
        spec=spec,
        seed=seed,
        buffer_size=buffer_size,
        sample_size=sample_size,
        streaming=streaming,
    )
    for split_name, dataset in sampled.items():
        processed = process_split(
            dataset=dataset,
            spec=spec,
            split_name=split_name,
            label_workers=label_workers,
            max_in_flight=max_in_flight,
            chunk_size=chunk_size,
            resume=resume,
            label_provider=label_provider,
            label_model=label_model,
        )
        output_path = build_split_output_path(spec=spec, split_name=split_name)
        save_processed_dataset(processed, output_path)
        logger.info(f"saved: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample SVG datasets, render images, label rows, and save the final "
            "dataset."
        )
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=sorted(DATASETS),
        default=None,
        help=(
            "Dataset key to process. Pass multiple times, or omit for all "
            "configured datasets."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--buffer-size", type=int, default=10_000, help="Buffer size for shuffling"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help=(
            "Override each split's raw scan size while developing. "
            "The processed output is still capped by the split target size."
        ),
    )
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help=(
            "Download/cache raw dataset splits first, then process selected rows "
            "from the local Arrow cache."
        ),
    )
    parser.add_argument(
        "--label-workers",
        type=int,
        default=4,
        help=(
            "Number of rows to render and label concurrently. Use 1 for sequential "
            "processing."
        ),
    )
    parser.add_argument(
        "--max-in-flight",
        type=int,
        default=None,
        help=(
            "Maximum rows submitted to workers at once. Defaults to 2x --label-workers."
        ),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1_000,
        help="Number of raw processed rows to save per checkpoint chunk.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from existing checkpoint chunks for each split.",
    )
    parser.add_argument(
        "--label-provider",
        choices=["groq", "gemini"],
        default="groq",
        help="Vision label provider to use for new checkpoint rows.",
    )
    parser.add_argument(
        "--label-model",
        default=None,
        help="Override the default model for the selected label provider.",
    )
    return parser.parse_args()


def main() -> None:
    load_env_file()

    args = parse_args()
    selected_datasets = args.dataset or sorted(DATASETS)
    label_workers = max(args.label_workers, 1)
    max_in_flight = args.max_in_flight or label_workers * 2
    max_in_flight = max(max_in_flight, label_workers)
    chunk_size = max(args.chunk_size, 1)
    label_model = args.label_model or get_default_label_model(args.label_provider)

    for dataset_key in selected_datasets:
        spec = DATASETS[dataset_key]
        process_dataset(
            spec=spec,
            seed=args.seed,
            buffer_size=args.buffer_size,
            sample_size=args.sample_size,
            streaming=not args.no_streaming,
            label_workers=label_workers,
            max_in_flight=max_in_flight,
            chunk_size=chunk_size,
            resume=args.resume,
            label_provider=args.label_provider,
            label_model=label_model,
        )


if __name__ == "__main__":
    main()
