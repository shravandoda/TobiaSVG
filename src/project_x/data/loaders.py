from collections.abc import Callable
from logging import getLogger
from typing import cast

from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from project_x.constants import DATA_PROCESSING_SEED, MAX_SEQUENCE_LENGTH
from project_x.data.collators import (
    image2svg_collator,
    image2svg_sequence_length,
    repair_collator,
    repair_sequence_length,
    text2svg_collator,
    text2svg_sequence_length,
)

logger = getLogger(__name__)
REQUIRED_SPLITS = {"train", "test", "val"}
DATASET_COLUMN = "dataset"
STRATIFY_COLUMN = "_dataset_class"


def split_generation_train_rows(
    dataset: DatasetDict,
    seed: int = DATA_PROCESSING_SEED,
) -> tuple[DatasetDict, DatasetDict]:
    """Assign disjoint, source-balanced training rows to text and image tasks."""
    missing_splits = REQUIRED_SPLITS.difference(dataset)
    if missing_splits:
        missing = ", ".join(sorted(missing_splits))
        raise ValueError(f"Dataset is missing required splits: {missing}")

    train = dataset["train"]
    if DATASET_COLUMN not in train.column_names:
        raise ValueError(
            f"Training split is missing the stratification column: {DATASET_COLUMN}"
        )

    train = train.add_column(STRATIFY_COLUMN, train[DATASET_COLUMN])
    train = train.class_encode_column(STRATIFY_COLUMN)
    task_splits = train.train_test_split(
        test_size=0.5,
        seed=seed,
        stratify_by_column=STRATIFY_COLUMN,
    )

    text_train = task_splits["train"].remove_columns(STRATIFY_COLUMN)
    image_train = task_splits["test"].remove_columns(STRATIFY_COLUMN)
    shared_splits = {name: dataset[name] for name in ("test", "val")}

    logger.info(
        "generation task split: text_train=%s image_train=%s seed=%s",
        len(text_train),
        len(image_train),
        seed,
    )
    return (
        DatasetDict({"train": text_train, **shared_splits}),
        DatasetDict({"train": image_train, **shared_splits}),
    )


def _build_dataloader(
    dataset: Dataset,
    collate_fn,
    batch_size: int,
    num_workers: int,
    shuffle: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset=cast(TorchDataset, dataset),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )


def _build_split_dataloaders(
    dataset: DatasetDict,
    collate_fn,
    batch_size: int,
    num_workers: int,
):
    train_loader = _build_dataloader(
        dataset["train"],
        collate_fn=collate_fn,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )
    test_loader = _build_dataloader(
        dataset["test"],
        collate_fn=collate_fn,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    val_loader = _build_dataloader(
        dataset["val"],
        collate_fn=collate_fn,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    return train_loader, test_loader, val_loader


def _add_sequence_length(
    row: dict,
    sequence_length_fn: Callable[[dict], int],
) -> dict[str, int]:
    return {"sequence_length": sequence_length_fn(row)}


def _within_sequence_limit(sequence_length: int) -> bool:
    return sequence_length <= MAX_SEQUENCE_LENGTH


def _prepare_task_dataset(
    dataset: DatasetDict,
    sequence_length_fn: Callable[[dict], int],
    columns: tuple[str, ...],
    task_name: str,
) -> DatasetDict:
    missing_splits = REQUIRED_SPLITS.difference(dataset)
    if missing_splits:
        missing = ", ".join(sorted(missing_splits))
        raise ValueError(f"Dataset is missing required splits: {missing}")

    original_sizes = {split_name: len(split) for split_name, split in dataset.items()}
    with_lengths = dataset.map(
        _add_sequence_length,
        fn_kwargs={"sequence_length_fn": sequence_length_fn},
        desc=f"Measuring {task_name} sequence lengths",
    )
    filtered = with_lengths.filter(
        _within_sequence_limit,
        input_columns=["sequence_length"],
        desc=f"Filtering {task_name} sequences",
    )

    for split_name, split in filtered.items():
        logger.info(
            "prepared %s/%s: kept=%s removed=%s max_sequence_length=%s",
            task_name,
            split_name,
            len(split),
            original_sizes[split_name] - len(split),
            MAX_SEQUENCE_LENGTH,
        )

    return filtered.select_columns(list(columns))


def get_text2svg_dataloader(
    dataset: DatasetDict,
    batch_size: int,
    num_workers: int = 0,
):
    dataset = _prepare_task_dataset(
        dataset,
        sequence_length_fn=text2svg_sequence_length,
        columns=("text", "svg"),
        task_name="text2svg",
    )
    return _build_split_dataloaders(
        dataset,
        text2svg_collator,
        batch_size=batch_size,
        num_workers=num_workers,
    )


def get_img2svg_dataloader(
    dataset: DatasetDict,
    batch_size: int,
    num_workers: int = 0,
):
    dataset = _prepare_task_dataset(
        dataset,
        sequence_length_fn=image2svg_sequence_length,
        columns=("svg",),
        task_name="image2svg",
    )
    return _build_split_dataloaders(
        dataset,
        image2svg_collator,
        batch_size=batch_size,
        num_workers=num_workers,
    )


def get_repair_dataloader(
    dataset: DatasetDict,
    batch_size: int,
    num_workers: int = 0,
):
    dataset = _prepare_task_dataset(
        dataset,
        sequence_length_fn=repair_sequence_length,
        columns=("svg", "corrupted_svg"),
        task_name="repair",
    )
    return _build_split_dataloaders(
        dataset,
        repair_collator,
        batch_size=batch_size,
        num_workers=num_workers,
    )
