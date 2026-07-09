from datasets import DatasetDict
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from src.project_x.constants import DATA_PROCESSING_SEED, SPLITS
from src.project_x.data.collators import (
    image2svg_collator,
    repair_collator,
    text2svg_collator,
)
from src.project_x.data.datasets import (
    ImageToSVGDataset,
    SVGRepairDataset,
    TextToSVGDataset,
)


def _build_dataloader(
    dataset: TorchDataset,
    collate_fn,
    batch_size: int,
    num_workers: int,
    shuffle: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )


def _build_split_dataloaders(
    dataset: DatasetDict,
    dataset_class,
    collate_fn,
    batch_size: int,
    num_workers: int,
):
    train_loader = _build_dataloader(
        dataset_class(dataset["train"]),
        collate_fn=collate_fn,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )
    test_loader = _build_dataloader(
        dataset_class(dataset["test"]),
        collate_fn=collate_fn,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    val_loader = _build_dataloader(
        dataset_class(dataset["val"]),
        collate_fn=collate_fn,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    return train_loader, test_loader, val_loader


def _with_train_test_val_splits(dataset: DatasetDict) -> DatasetDict:
    if {"train", "test", "val"}.issubset(dataset):
        return dataset

    train_and_holdout = dataset["train"].train_test_split(
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


def get_text2svg_dataloader(
    dataset: DatasetDict,
    batch_size: int,
    num_workers: int = 0,
):
    return _build_split_dataloaders(
        dataset,
        TextToSVGDataset,
        text2svg_collator,
        batch_size,
        num_workers,
    )


def get_img2svg_dataloader(
    dataset: DatasetDict,
    batch_size: int,
    num_workers: int = 0,
):
    return _build_split_dataloaders(
        dataset,
        ImageToSVGDataset,
        image2svg_collator,
        batch_size,
        num_workers,
    )


def get_repair_dataloader(
    dataset: DatasetDict,
    batch_size: int,
    num_workers: int = 0,
):
    dataset = _with_train_test_val_splits(dataset)

    return _build_split_dataloaders(
        dataset,
        SVGRepairDataset,
        repair_collator,
        batch_size,
        num_workers,
    )
