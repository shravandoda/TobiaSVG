import argparse
import logging
from pathlib import Path
from typing import Any

from datasets import Dataset, IterableDataset

from scripts.data.process_svg import (
    SvgQualityError,
    build_label_prompt,
    clean_validate_and_render_svg,
    request_text_label,
)
from scripts.data.sample_dataset import (
    DATASETS,
    DatasetSpec,
    sample_dataset,
)
from scripts.data.utils import (
    build_image_path,
    build_split_output_path,
    get_source_filename,
    get_source_svg,
    load_env_file,
)
from src.project_x.utils.logger import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

LOG_EVERY = 500
FINAL_COLUMNS = ["filename", "svg", "text"]


def process_row(
    raw_row: dict[str, Any],
    spec: DatasetSpec,
    split_name: str,
) -> dict[str, Any]:
    filename = get_source_filename(raw_row, spec)
    raw_svg = get_source_svg(raw_row, spec)
    image_path = build_image_path(
        spec=spec,
        split_name=split_name,
        filename=filename,
    )

    try:
        svg = clean_validate_and_render_svg(raw_svg, image_path)
        row: dict[str, str] = {"filename": filename, "svg": svg}
        prompt = build_label_prompt(row)
        text = request_text_label(image_path, prompt=prompt)
        return {"filename": filename, "svg": svg, "text": text, "keep": True}
    except SvgQualityError as exc:
        logger.info("filtered: %s (%s)", filename, exc)
        return {"filename": filename, "svg": raw_svg, "text": None, "keep": False}


def iter_dataset(dataset: IterableDataset):
    yield from dataset


def process_split(
    dataset: IterableDataset,
    spec: DatasetSpec,
    split_name: str,
) -> Dataset:
    raw_columns = dataset.column_names or []
    raw_columns_to_remove = [
        column_name for column_name in raw_columns if column_name not in FINAL_COLUMNS
    ]

    dataset = dataset.map(
        process_row,
        fn_kwargs={"spec": spec, "split_name": split_name},
        remove_columns=raw_columns_to_remove,
    )
    dataset = dataset.filter(lambda x: x["keep"])
    dataset = dataset.take(spec.splits[split_name].target_size)
    dataset = dataset.select_columns(FINAL_COLUMNS)
    return Dataset.from_generator(iter_dataset, gen_kwargs={"dataset": dataset})


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
) -> None:
    sampled = sample_dataset(
        spec=spec,
        seed=seed,
        buffer_size=buffer_size,
        sample_size=sample_size,
    )
    for split_name, dataset in sampled.items():
        processed = process_split(
            dataset=dataset,
            spec=spec,
            split_name=split_name,
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
            "When set, target-size stopping is disabled."
        ),
    )
    return parser.parse_args()


def main() -> None:
    load_env_file()

    args = parse_args()
    selected_datasets = args.dataset or sorted(DATASETS)

    for dataset_key in selected_datasets:
        spec = DATASETS[dataset_key]
        process_dataset(
            spec=spec,
            seed=args.seed,
            buffer_size=args.buffer_size,
            sample_size=args.sample_size,
        )


if __name__ == "__main__":
    main()
