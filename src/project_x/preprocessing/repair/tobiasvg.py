import argparse
import random
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from datasets import (
    Dataset,
    DatasetDict,
    concatenate_datasets,
    load_dataset,
    load_from_disk,
)

from project_x.preprocessing.repair.generate import (
    CORRUPTIONS,
    REPAIR_FEATURES,
    build_repair_rows,
    save_repair_dataset,
)
from project_x.preprocessing.repair import package as repair_package
from project_x.constants import DATA_PROCESSING_SEED, HF_TOKEN

SOURCE_REPO_ID = "shravandoda/TobiaSVG"
DEFAULT_OUTPUT_PATH = Path("data/processed/repair_datasets/tobiasvg_repair")
DEFAULT_CHECKPOINT_ROOT = Path("data/processed/checkpoints/tobiasvg_repair")
DEFAULT_SOURCE_DATASETS = [
    "vfig_shapes",
    "vfig_diagrams",
    "animal_illustrations",
    "starvector_diagrams",
]
DEFAULT_CORRUPTIONS = sorted(set(CORRUPTIONS) - {"truncation"})
REPAIR_LEVELS = ["medium", "hard"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the TobiaSVG medium/hard SVG repair dataset locally."
    )
    parser.add_argument("--repo-id", default=SOURCE_REPO_ID)
    parser.add_argument(
        "--source-dataset",
        action="append",
        choices=DEFAULT_SOURCE_DATASETS,
        default=None,
        help="Source dataset key to include. Omit for all repair-capable datasets.",
    )
    parser.add_argument(
        "--split",
        action="append",
        default=None,
        help="Split to process. Omit for every split in the source DatasetDict.",
    )
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=DATA_PROCESSING_SEED)
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start offset in the filtered source split.",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=None,
        help="End offset, exclusive, in the filtered source split.",
    )
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--package-only",
        action="store_true",
        help="Only package existing repair checkpoints into the final DatasetDict.",
    )
    parser.add_argument(
        "--package-checkpoint-root",
        type=Path,
        action="append",
        default=None,
        help=(
            "Checkpoint root to package. Pass multiple times, or omit to use "
            "the known generated repair roots."
        ),
    )
    parser.add_argument(
        "--truncation-fraction",
        type=float,
        default=0.10,
        help=(
            "Fraction of generated medium/hard repair rows to duplicate with an "
            "additional truncation corruption. Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--min-render-mse",
        type=float,
        default=0.002,
        help="Minimum normalized MSE between clean and corrupted renders.",
    )
    parser.add_argument(
        "--min-changed-pixel-ratio",
        type=float,
        default=0.01,
        help="Minimum fraction of pixels changed between clean and corrupted renders.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.package_only:
        package_existing_checkpoints(args)
        return

    source_dataset_keys = args.source_dataset or DEFAULT_SOURCE_DATASETS
    source = load_dataset(args.repo_id, token=HF_TOKEN)
    if not isinstance(source, DatasetDict):
        source = DatasetDict({"train": source})

    selected_splits = args.split or [str(split_name) for split_name in source]
    output = DatasetDict()

    for split_name in selected_splits:
        if split_name not in source:
            print(f"skipped: source repo does not contain split {split_name}")
            continue

        split = source[split_name]
        split = split.filter(
            lambda row: row.get("dataset") in source_dataset_keys,
            desc=f"filter {split_name}",
        )
        split, row_start_offset = select_range(
            split,
            start_index=args.start_index,
            end_index=args.end_index,
            sample_size=args.sample_size,
        )
        print(
            "selected range: "
            f"{split_name}[{row_start_offset}:"
            f"{row_start_offset + len(split)}] rows={len(split)}"
        )

        checkpoint_split_root = args.checkpoint_root / split_name
        build_split_checkpoints(
            split,
            split_name=split_name,
            checkpoint_split_root=checkpoint_split_root,
            row_start_offset=row_start_offset,
            args=args,
        )
        output[split_name] = combine_split_checkpoints(checkpoint_split_root)

    save_dataset_dict(output, args.output_path)
    print(f"saved: {args.output_path}")


def package_existing_checkpoints(args: argparse.Namespace) -> None:
    checkpoint_roots = (
        args.package_checkpoint_root or repair_package.DEFAULT_CHECKPOINT_ROOTS
    )
    dataset = repair_package.package_checkpoints(checkpoint_roots)
    dataset_dict = repair_package.split_dataset(dataset)
    repair_package.save_dataset_dict(dataset_dict, args.output_path)
    print(f"saved: {args.output_path}")
    print(f"rows: {sum(len(split) for split in dataset_dict.values()):,}")
    print(f"splits: { {split: len(dataset_dict[split]) for split in dataset_dict} }")


def select_range(
    dataset: Dataset,
    *,
    start_index: int,
    end_index: int | None,
    sample_size: int | None,
) -> tuple[Dataset, int]:
    start_index = max(start_index, 0)
    stop_index = len(dataset) if end_index is None else min(end_index, len(dataset))
    if sample_size is not None:
        stop_index = min(stop_index, start_index + sample_size)

    if start_index >= stop_index:
        return dataset.select([]), start_index

    return dataset.select(range(start_index, stop_index)), start_index


def build_split_checkpoints(
    dataset: Dataset,
    *,
    split_name: str,
    checkpoint_split_root: Path,
    row_start_offset: int,
    args: argparse.Namespace,
) -> None:
    jobs: list[dict[str, Any]] = []
    total_rows = len(dataset)
    for chunk_start in range(0, total_rows, args.chunk_size):
        chunk_stop = min(chunk_start + args.chunk_size, total_rows)
        global_chunk_start = row_start_offset + chunk_start
        chunk = dataset.select(range(chunk_start, chunk_stop))

        for level in REPAIR_LEVELS:
            chunk_path = (
                checkpoint_split_root / level / f"chunk_{global_chunk_start:08d}"
            )
            truncation_path = (
                checkpoint_split_root
                / "truncation"
                / level
                / f"chunk_{global_chunk_start:08d}"
            )
            if chunk_path.exists():
                if args.resume:
                    print(f"resume: {chunk_path}")
                    repair_chunk = load_checkpoint_dataset(chunk_path)
                    save_truncation_checkpoint(
                        repair_chunk,
                        checkpoint_path=truncation_path,
                        level=level,
                        seed=args.seed + global_chunk_start,
                        fraction=args.truncation_fraction,
                        resume=args.resume,
                    )
                    continue
                raise ValueError(
                    f"Checkpoint already exists at {chunk_path}. "
                    "Pass --resume to continue."
                )

            jobs.append(
                {
                    "chunk": chunk,
                    "split_name": split_name,
                    "level": level,
                    "chunk_path": chunk_path,
                    "truncation_path": truncation_path,
                    "seed": args.seed + global_chunk_start,
                    "min_render_mse": args.min_render_mse,
                    "min_changed_pixel_ratio": args.min_changed_pixel_ratio,
                    "truncation_fraction": args.truncation_fraction,
                    "resume": args.resume,
                }
            )

    if not jobs:
        return

    workers = max(args.workers, 1)
    if workers == 1:
        for job in jobs:
            print(process_repair_chunk_job(job))
        return

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_repair_chunk_job, job) for job in jobs]
        for future in as_completed(futures):
            print(future.result())


def process_repair_chunk_job(job: dict[str, Any]) -> str:
    rows = build_repair_rows(
        job["chunk"],
        dataset_key="tobiasvg",
        split_name=job["split_name"],
        corruption_types=DEFAULT_CORRUPTIONS,
        corruption_level=job["level"],
        per_row=1,
        seed=job["seed"],
        start_index=0,
        sample_size=None,
        min_render_mse=job["min_render_mse"],
        min_changed_pixel_ratio=job["min_changed_pixel_ratio"],
    )
    repair_chunk = save_repair_dataset(rows, job["chunk_path"])
    save_truncation_checkpoint(
        repair_chunk,
        checkpoint_path=job["truncation_path"],
        level=job["level"],
        seed=job["seed"],
        fraction=job["truncation_fraction"],
        resume=job["resume"],
    )
    return (
        f"saved checkpoint: {job['chunk_path']} "
        f"source_rows={len(job['chunk'])} repair_rows={len(repair_chunk)}"
    )


def save_truncation_checkpoint(
    dataset: Dataset,
    *,
    checkpoint_path: Path,
    level: str,
    seed: int,
    fraction: float,
    resume: bool,
) -> None:
    if fraction <= 0:
        return

    if checkpoint_path.exists():
        if resume:
            print(f"resume: {checkpoint_path}")
            return
        raise ValueError(
            f"Checkpoint already exists at {checkpoint_path}. "
            "Pass --resume to continue."
        )

    rows = build_truncation_rows(
        dataset,
        level=level,
        seed=seed,
        fraction=fraction,
    )
    truncation_chunk = save_repair_dataset(rows, checkpoint_path)
    print(
        f"saved checkpoint: {checkpoint_path} "
        f"source_rows={len(dataset)} repair_rows={len(truncation_chunk)}"
    )


def build_truncation_rows(
    dataset: Dataset,
    *,
    level: str,
    seed: int,
    fraction: float,
) -> list[dict]:
    if not 0 < fraction <= 1:
        return []

    row_count = len(dataset)
    truncation_count = round(row_count * fraction)
    if row_count and truncation_count == 0:
        truncation_count = 1

    rng = random.Random(seed)
    selected_indices = sorted(rng.sample(range(row_count), k=truncation_count))
    rows: list[dict] = []

    for selected_index in selected_indices:
        row = dict(dataset[selected_index])
        truncation_seed = seed * 100_000 + selected_index
        truncated_svg = CORRUPTIONS["truncation"](
            row["corrupted_svg"],
            random.Random(truncation_seed),
            level,
        )
        row["corrupted_svg"] = truncated_svg
        row["corruption_type"] = f"{row['corruption_type']}+truncation"
        row["corruption_count"] = int(row["corruption_count"]) + 1
        row["corruption_seed"] = truncation_seed
        row["render_mse"] = -1.0
        row["changed_pixel_ratio"] = -1.0
        rows.append(row)

    return rows


def combine_split_checkpoints(checkpoint_split_root: Path) -> Dataset:
    chunks: list[Dataset] = []
    for level in REPAIR_LEVELS:
        level_root = checkpoint_split_root / level
        for chunk_path in sorted(level_root.glob("chunk_*")):
            chunks.append(load_checkpoint_dataset(chunk_path))

    for level in REPAIR_LEVELS:
        level_root = checkpoint_split_root / "truncation" / level
        for chunk_path in sorted(level_root.glob("chunk_*")):
            chunks.append(load_checkpoint_dataset(chunk_path))

    if not chunks:
        return Dataset.from_list([], features=REPAIR_FEATURES)

    return concatenate_datasets(chunks)


def load_checkpoint_dataset(checkpoint_path: Path) -> Dataset:
    dataset = load_from_disk(str(checkpoint_path))
    if not isinstance(dataset, Dataset):
        raise TypeError(f"Expected a Dataset at {checkpoint_path}")
    return dataset


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
