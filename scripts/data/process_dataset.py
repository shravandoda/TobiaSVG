import argparse
from pathlib import Path
from typing import Any

from datasets import Dataset, IterableDataset

from scripts.data.process_svg import (
    SvgQualityError,
    build_label_prompt,
    clean_svg,
    render_svg_to_png,
    request_text_label,
    validate_svg_quality,
)
from scripts.data.sample_dataset import DATASETS, DatasetSpec, sample_dataset
from scripts.data.utils import load_env_file, make_example_id


def build_image_path(
    *,
    image_root: Path,
    spec: DatasetSpec,
    split_name: str,
    filename: str,
) -> Path:
    """Path for the rendered PNG that can be joined back by `filename`.

    The final text dataset stores only `filename`, `svg`, and `text`. A later
    image dataset can use the same `filename` value and load images from:
    data/images/<dataset>/<split>/<filename>.png
    """
    return image_root / spec.key / split_name / f"{filename}.png"


def process_row(
    raw_row: dict[str, Any],
    *,
    spec: DatasetSpec,
    split_name: str,
    index: int,
    image_root: Path,
    fail_on_error: bool = True,
) -> dict[str, Any] | None:
    filename = make_example_id(raw_row, spec, split_name=split_name, index=index)
    image_path = build_image_path(
        image_root=image_root,
        spec=spec,
        split_name=split_name,
        filename=filename,
    )

    try:
        svg = clean_svg(str(raw_row[spec.svg_column]))
        validate_svg_quality(svg)

        render_svg_to_png(svg, image_path)

        row = {"filename": filename, "svg": svg}
        prompt = build_label_prompt(row)
        text = request_text_label(image_path, prompt=prompt)
    except SvgQualityError as exc:
        print(f"filtered: {filename} ({exc})")
        return None
    except Exception as exc:
        if fail_on_error:
            raise

        print(f"failed: {filename} ({type(exc).__name__}: {exc})")
        return None

    return {"filename": filename, "svg": svg, "text": text}


def process_split(
    dataset: IterableDataset,
    *,
    spec: DatasetSpec,
    split_name: str,
    image_root: Path,
    fail_on_error: bool,
    target_size: int | None,
) -> Dataset:
    rows = []

    for index, raw_row in enumerate(dataset):
        if target_size is not None and len(rows) >= target_size:
            break

        row = process_row(
            raw_row,
            spec=spec,
            split_name=split_name,
            index=index,
            image_root=image_root,
            fail_on_error=fail_on_error,
        )
        if row is None:
            continue

        rows.append(row)
        print(f"processed: {row['filename']}")

    return build_hf_dataset(rows)


def build_hf_dataset(rows: list[dict[str, Any]]) -> Dataset:
    """Build a Hugging Face Dataset from processed Python rows.

    HF datasets learning checkpoint:
    - convert the list of dictionaries into a Dataset
    - observe how columns and null values are inferred
    """
    return Dataset.from_list(rows)


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
    *,
    output_root: Path,
    image_root: Path,
    seed: int,
    buffer_size: int,
    sample_size: int | None,
    fail_on_error: bool,
) -> None:
    sampled = sample_dataset(
        spec,
        seed=seed,
        buffer_size=buffer_size,
        sample_size=sample_size,
    )
    target_size = None if sample_size is not None else spec.target_size

    for split_name, dataset in sampled.items():
        processed = process_split(
            dataset,
            spec=spec,
            split_name=split_name,
            image_root=image_root,
            fail_on_error=fail_on_error,
            target_size=target_size,
        )
        output_path = output_root / spec.key / split_name
        save_processed_dataset(processed, output_path)
        print(f"saved: {output_path}")


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
    parser.add_argument("--buffer-size", type=int, default=10_000)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help=(
            "Override each split's raw scan size while developing. "
            "When set, target-size stopping is disabled."
        ),
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=Path("data/images"),
        help="Rendered PNG root: <root>/<dataset>/<split>/<filename>.png.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/datasets"),
        help="Where processed Hugging Face datasets should be saved.",
    )
    parser.add_argument(
        "--skip-failed-rows",
        action="store_true",
        help="Skip failed rows instead of stopping at the first row error.",
    )
    return parser.parse_args()


def main() -> None:
    load_env_file()

    args = parse_args()
    selected_datasets = args.dataset or sorted(DATASETS)

    for dataset_key in selected_datasets:
        spec = DATASETS[dataset_key]
        process_dataset(
            spec,
            output_root=args.output_root,
            image_root=args.image_root,
            seed=args.seed,
            buffer_size=args.buffer_size,
            sample_size=args.sample_size,
            fail_on_error=not args.skip_failed_rows,
        )


if __name__ == "__main__":
    main()
