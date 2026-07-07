import argparse
import logging
import os
from dataclasses import dataclass

from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    load_dataset,
)

from src.project_x.utils.logger import setup_logging
from src.project_x.constants import *

setup_logging()
logger = logging.getLogger(__name__)


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


def load_local_splits(spec: DatasetSpec) -> DatasetDict:
    token = os.environ.get("HF_TOKEN") or None
    dataset = load_dataset(
        spec.path,
        spec.config_name,
        streaming=False,
        token=token,
    )

    if isinstance(dataset, DatasetDict):
        return dataset


def get_split_spec(spec: DatasetSpec, split_name: str) -> DatasetSpecSplit:
    try:
        return spec.splits[split_name]
    except KeyError as exc:
        raise ValueError(f"{spec.key} does not define split `{split_name}`.") from exc


def sample_dataset(
    spec: DatasetSpec,
    *,
    seed: int,
    buffer_size: int,
    sample_size: int | None = None,
    streaming: bool = True,
) -> dict[str, IterableDataset]:
    return (
        sample_streaming_dataset(
            spec=spec,
            seed=seed,
            buffer_size=buffer_size,
            sample_size=sample_size,
        )
        if streaming
        else sample_local_dataset(
            spec=spec,
            seed=seed,
            sample_size=sample_size,
        )
    )


def sample_streaming_dataset(
    spec: DatasetSpec,
    *,
    seed: int,
    buffer_size: int,
    sample_size: int | None,
) -> dict[str, IterableDataset]:
    dataset_dict = load_streaming_splits(spec)
    sampled: dict[str, IterableDataset] = {}

    for split_name, spec_split in spec.splits.items():
        validate_split_exists(spec, split_name, dataset_dict)
        dataset = dataset_dict[split_name]
        split_sample_size = (
            sample_size if sample_size is not None else spec_split.sample_size
        )

        sampled[split_name] = sample_streaming_split(
            dataset=dataset,
            seed=seed,
            buffer_size=buffer_size,
            sample_size=split_sample_size,
        )

    return sampled


def sample_local_dataset(
    spec: DatasetSpec,
    *,
    seed: int,
    sample_size: int | None,
) -> dict[str, IterableDataset]:
    dataset_dict = load_local_splits(spec)
    sampled: dict[str, IterableDataset] = {}

    for split_name, spec_split in spec.splits.items():
        validate_split_exists(spec, split_name, dataset_dict)
        dataset = dataset_dict[split_name]
        split_sample_size = (
            sample_size if sample_size is not None else spec_split.sample_size
        )

        sampled[split_name] = sample_local_split(
            dataset=dataset,
            seed=seed,
            sample_size=split_sample_size,
        )

    return sampled


def validate_split_exists(
    spec: DatasetSpec,
    split_name: str,
    dataset_dict: DatasetDict | IterableDatasetDict,
) -> None:
    if split_name in dataset_dict:
        return

    available_splits = ", ".join(str(key) for key in dataset_dict)
    raise ValueError(
        f"{spec.key} does not contain split `{split_name}`. "
        f"Available splits: {available_splits}"
    )


def sample_streaming_split(
    dataset: IterableDataset,
    *,
    seed: int,
    buffer_size: int,
    sample_size: int,
) -> IterableDataset:
    """Sample one streaming split.

    HF datasets learning checkpoint:
    - shuffle this streaming split with seed and buffer_size
    - take sample_size rows
    - return the sampled iterable
    """
    dataset = dataset.shuffle(seed=seed, buffer_size=buffer_size)
    dataset = dataset.take(sample_size)
    return dataset


def sample_local_split(
    dataset: Dataset,
    *,
    seed: int,
    sample_size: int,
) -> IterableDataset:
    """Sample an Arrow-backed split, then expose it as a lazy iterable."""
    dataset = dataset.shuffle(seed=seed)
    dataset = dataset.select(range(min(sample_size, dataset.num_rows)))
    return dataset.to_iterable_dataset()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load and sample configured SVG datasets."
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=sorted(DATASETS),
        default=None,
        help=(
            "Dataset key to sample. Pass multiple times, or omit for all "
            "configured datasets."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    # Shuffle keeps a rolling buffer and samples randomly from that buffer.
    parser.add_argument("--buffer-size", type=int, default=10_000)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Override each split's raw scan size while developing.",
    )
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help=(
            "Download/cache raw dataset splits first, then sample selected rows "
            "from the local Arrow cache."
        ),
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
            streaming=not args.no_streaming,
        )

        for split_name, dataset in sampled.items():
            logger.info(f"{spec.key}/{split_name}")
            for index, row in enumerate(dataset):
                source_id = row.get(spec.id_column) if spec.id_column else None
                logger.info(f"  {index}: {source_id or '<no source id>'}")


if __name__ == "__main__":
    main()
