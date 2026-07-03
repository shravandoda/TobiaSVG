import argparse
import os
from collections.abc import Iterable
from dataclasses import dataclass

from datasets import IterableDataset, IterableDatasetDict, load_dataset


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    path: str
    config_name: str | None
    svg_column: str
    id_column: str | None
    raw_sample_size: int
    target_size: int


DATASETS = {
    "starvector": DatasetSpec(
        key="starvector",
        path="starvector/svg-diagrams",
        config_name=None,
        svg_column="Svg",
        id_column="Filename",
        raw_sample_size=182_144,
        target_size=27_600,
    ),
    "vfig_shapes": DatasetSpec(
        key="vfig_shapes",
        path="QijiaHe/VFIG-Data",
        config_name="VFIG-Data-Shapes-and-Arrows",
        svg_column="svg",
        id_column="filename",
        raw_sample_size=6_545,
        target_size=6_545,
    ),
    "vfig_complex": DatasetSpec(
        key="vfig_complex",
        path="QijiaHe/VFIG-Data",
        config_name="VFIG-Data-Complex-Diagrams",
        svg_column="svg",
        id_column="filename",
        raw_sample_size=60_034,
        target_size=60_034,
    ),
}


def load_streaming_splits(spec: DatasetSpec) -> IterableDatasetDict:
    token = os.environ.get("HF_TOKEN") or None
    dataset = load_dataset(
        spec.path,
        spec.config_name,
        streaming=True,
        token=token,
    )

    if isinstance(dataset, IterableDatasetDict):
        return dataset

    return IterableDatasetDict({"train": dataset})


def sample_dataset(
    spec: DatasetSpec,
    *,
    seed: int,
    buffer_size: int,
    sample_size: int | None = None,
) -> IterableDatasetDict:
    dataset_dict = load_streaming_splits(spec)
    sampled = {}

    for split_name, dataset in dataset_dict.items():
        sampled[split_name] = sample_split(
            dataset,
            seed=seed,
            buffer_size=buffer_size,
            sample_size=sample_size if sample_size is not None else spec.raw_sample_size,
        )

    return IterableDatasetDict(sampled)


def sample_split(
    dataset: IterableDataset,
    *,
    seed: int,
    buffer_size: int,
    sample_size: int,
) -> Iterable:
    """Sample one streaming split.

    HF datasets learning checkpoint:
    - shuffle this streaming split with seed and buffer_size
    - take sample_size rows
    - return the sampled iterable
    """
    dataset = dataset.shuffle(seed=seed, buffer_size=buffer_size)
    dataset = dataset.take(sample_size)
    return dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load and sample configured SVG datasets."
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=sorted(DATASETS),
        default=None,
        help="Dataset key to sample. Pass multiple times, or omit for all configured datasets.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--buffer-size", type=int, default=10_000)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Override each split's raw scan size while developing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_datasets = args.dataset or sorted(DATASETS)

    for dataset_key in selected_datasets:
        spec = DATASETS[dataset_key]
        sampled = sample_dataset(
            spec,
            seed=args.seed,
            buffer_size=args.buffer_size,
            sample_size=args.sample_size,
        )

        for split_name, dataset in sampled.items():
            print(f"{spec.key}/{split_name}")
            for index, row in enumerate(dataset):
                source_id = row.get(spec.id_column) if spec.id_column else None
                print(f"  {index}: {source_id or '<no source id>'}")


if __name__ == "__main__":
    main()
