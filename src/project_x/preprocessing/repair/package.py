"""Package TobiaSVG repair checkpoints into a clean DatasetDict."""

import argparse
import shutil
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, Features, Value, concatenate_datasets
from datasets import load_from_disk

from project_x.constants import DATA_PROCESSING_SEED, SPLITS


DEFAULT_CHECKPOINT_ROOTS = [
    Path("data/processed/checkpoints/tobiasvg_repair"),
    Path("data/processed/checkpoints/tobiasvg_repair_animal_illustrations"),
    Path("data/processed/checkpoints/tobiasvg_repair_vfig_shapes"),
]
DEFAULT_OUTPUT_PATH = Path("data/processed/repair_datasets/tobiasvg_repair")

FINAL_COLUMNS = [
    "filename",
    "svg",
    "corrupted_svg",
    "render_mse",
    "changed_pixel_ratio",
    "text",
    "caption_style",
    "dataset",
    "corruption_level",
    "is_truncated",
]
FINAL_FEATURES = Features(
    {
        "filename": Value("string"),
        "svg": Value("string"),
        "corrupted_svg": Value("string"),
        "render_mse": Value("float64"),
        "changed_pixel_ratio": Value("float64"),
        "text": Value("string"),
        "caption_style": Value("string"),
        "dataset": Value("string"),
        "corruption_level": Value("string"),
        "is_truncated": Value("bool"),
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package TobiaSVG repair checkpoint chunks for pushing."
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        action="append",
        default=None,
        help="Checkpoint root to include. Omit to include all generated repair roots.",
    )
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_roots = args.checkpoint_root or DEFAULT_CHECKPOINT_ROOTS
    dataset = package_checkpoints(checkpoint_roots)
    dataset_dict = split_dataset(dataset)
    save_dataset_dict(dataset_dict, args.output_path)
    print(f"saved: {args.output_path}")
    print(f"rows: {sum(len(split) for split in dataset_dict.values()):,}")
    print(f"splits: { {split: len(dataset_dict[split]) for split in dataset_dict} }")


def package_checkpoints(checkpoint_roots: list[Path]) -> Dataset:
    chunks = []
    for checkpoint_root in checkpoint_roots:
        if not checkpoint_root.exists():
            print(f"skipped missing checkpoint root: {checkpoint_root}")
            continue

        for chunk_path in sorted(checkpoint_root.rglob("chunk_*")):
            if not chunk_path.is_dir():
                continue
            chunks.append(clean_chunk(chunk_path, checkpoint_root))

    if not chunks:
        return Dataset.from_list([], features=FINAL_FEATURES)

    return concatenate_datasets(chunks).cast(FINAL_FEATURES)


def split_dataset(dataset: Dataset) -> DatasetDict:
    if len(dataset) == 0:
        return DatasetDict(
            {
                "train": dataset,
                "test": dataset,
                "val": dataset,
            }
        )

    train_and_holdout = dataset.train_test_split(
        test_size=SPLITS["test"] + SPLITS["val"],
        seed=DATA_PROCESSING_SEED,
    )
    test_and_val = train_and_holdout["test"].train_test_split(
        test_size=SPLITS["val"] / (SPLITS["test"] + SPLITS["val"]),
        seed=DATA_PROCESSING_SEED,
    )

    return DatasetDict(
        {
            "train": train_and_holdout["train"],
            "test": test_and_val["train"],
            "val": test_and_val["test"],
        }
    )


def clean_chunk(chunk_path: Path, checkpoint_root: Path) -> Dataset:
    dataset = load_from_disk(str(chunk_path))
    if not isinstance(dataset, Dataset):
        raise TypeError(f"Expected a Dataset at {chunk_path}")

    is_truncated = is_truncation_chunk(chunk_path, checkpoint_root)

    return dataset.map(
        lambda row: final_row(row, is_truncated=is_truncated),
        remove_columns=dataset.column_names,
        features=FINAL_FEATURES,
        desc=f"clean {chunk_path}",
    )


def is_truncation_chunk(chunk_path: Path, checkpoint_root: Path) -> bool:
    relative_parts = chunk_path.relative_to(checkpoint_root).parts
    return "truncation" in relative_parts


def final_row(row: dict[str, Any], *, is_truncated: bool) -> dict[str, Any]:
    return {
        "filename": row["filename"],
        "svg": row["svg"],
        "corrupted_svg": row["corrupted_svg"],
        "render_mse": row["render_mse"],
        "changed_pixel_ratio": row["changed_pixel_ratio"],
        "text": row["text"],
        "caption_style": row["caption_style"],
        "dataset": row["dataset"],
        "corruption_level": row["corruption_level"],
        "is_truncated": is_truncated,
    }


def save_dataset_dict(dataset: DatasetDict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f".{output_path.name}.tmp")
    if temp_path.exists():
        shutil.rmtree(temp_path)
    dataset.save_to_disk(str(temp_path))
    if output_path.exists():
        shutil.rmtree(output_path)
    temp_path.rename(output_path)


if __name__ == "__main__":
    main()
