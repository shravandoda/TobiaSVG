"""Package TobiaSVG repair checkpoints into a clean DatasetDict."""

import argparse
import hashlib
import shutil
from pathlib import Path
from typing import Any

from datasets import (
    Dataset,
    DatasetDict,
    Features,
    Value,
    concatenate_datasets,
    load_dataset,
    load_from_disk,
)

from project_x.constants import DATA_PROCESSING_SEED, HF_TOKEN, SPLITS


DEFAULT_CHECKPOINT_ROOTS = [
    Path("data/processed/checkpoints/tobiasvg_repair"),
    Path("data/processed/checkpoints/tobiasvg_repair_animal_illustrations"),
    Path("data/processed/checkpoints/tobiasvg_repair_vfig_shapes"),
]
DEFAULT_OUTPUT_PATH = Path("data/processed/repair_datasets/tobiasvg_repair")
SOURCE_REPO_ID = "shravandoda/TobiaSVG"
SPLIT_PRIORITY = ("train", "test", "val")

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
    source_dataset = load_dataset(SOURCE_REPO_ID, token=HF_TOKEN)
    if not isinstance(source_dataset, DatasetDict):
        source_dataset = DatasetDict({"train": source_dataset})
    dataset_dict = split_dataset(dataset, source_dataset)
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


def svg_digest(svg: str) -> str:
    return hashlib.sha256(svg.encode("utf-8")).hexdigest()


def build_source_split_lookup(source_dataset: DatasetDict) -> dict[str, str]:
    """Map each clean SVG to one canonical source split."""
    lookup = {}
    for split_name in SPLIT_PRIORITY:
        if split_name not in source_dataset:
            continue
        for svg in source_dataset[split_name]["svg"]:
            lookup.setdefault(svg_digest(svg), split_name)
    return lookup


def split_source_dataset(dataset: Dataset) -> DatasetDict:
    """Assign identical SVGs to one deterministic 80/10/10 split."""
    indices = {split_name: [] for split_name in SPLIT_PRIORITY}
    train_threshold = SPLITS["train"]
    test_threshold = train_threshold + SPLITS["test"]

    for index, svg in enumerate(dataset["svg"]):
        digest = svg_digest(svg)
        seeded_digest = hashlib.sha256(
            f"{DATA_PROCESSING_SEED}:{digest}".encode()
        ).digest()
        fraction = int.from_bytes(seeded_digest[:8], "big") / 2**64

        if fraction < train_threshold:
            split_name = "train"
        elif fraction < test_threshold:
            split_name = "test"
        else:
            split_name = "val"
        indices[split_name].append(index)

    return DatasetDict(
        {
            split_name: dataset.select(split_indices).shuffle(seed=DATA_PROCESSING_SEED)
            for split_name, split_indices in indices.items()
        }
    )


def split_dataset(dataset: Dataset, source_dataset: DatasetDict) -> DatasetDict:
    """Keep every repair variant in the clean SVG's canonical source split."""
    if len(dataset) == 0:
        return DatasetDict(
            {
                "train": dataset,
                "test": dataset,
                "val": dataset,
            }
        )

    source_split_by_svg = build_source_split_lookup(source_dataset)
    indices = {split_name: [] for split_name in SPLIT_PRIORITY}
    missing_digests = set()

    for index, svg in enumerate(dataset["svg"]):
        digest = svg_digest(svg)
        split_name = source_split_by_svg.get(digest)
        if split_name is None:
            missing_digests.add(digest)
            continue
        indices[split_name].append(index)

    if missing_digests:
        raise ValueError(
            f"{len(missing_digests)} repair targets are missing from {SOURCE_REPO_ID}"
        )

    return DatasetDict(
        {
            split_name: dataset.select(split_indices).shuffle(seed=DATA_PROCESSING_SEED)
            for split_name, split_indices in indices.items()
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
