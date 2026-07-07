"""Push processed SVG datasets to the Hugging Face Hub."""

import argparse
from pathlib import Path
from typing import cast

from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk
from dotenv import load_dotenv

from src.project_x.constants import DATASETS, HF_TOKEN, SEED, SPLITS, DatasetSpec

load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Push processed SVG datasets to the Hugging Face Hub."
    )
    parser.add_argument("--repo-id", required=True)
    return parser.parse_args()


def fill_caption_style(dataset: Dataset, caption_style: str) -> Dataset:
    if "caption_style" not in dataset.column_names:
        return dataset.add_column("caption_style", [caption_style] * len(dataset))

    if any(value is None for value in dataset["caption_style"]):
        return dataset.map(
            lambda row: {"caption_style": row["caption_style"] or caption_style}
        )

    return dataset


def get_dataset_dict(dataset_path: Path, spec: DatasetSpec) -> DatasetDict:
    dataset_dict = load_from_disk(str(dataset_path))
    train = fill_caption_style(dataset_dict["train"], spec.caption_style)
    test = dataset_dict.get("test")
    val = dataset_dict.get("val")

    if test is not None:
        test = fill_caption_style(test, spec.caption_style)

    if val is not None:
        val = fill_caption_style(val, spec.caption_style)

    if val is None and test is None:
        # Get the train-test split
        train_test_split = train.train_test_split(test_size=SPLITS["test"], seed=SEED)
        train, test = train_test_split["train"], train_test_split["test"]

        # Get the train-val split
        train_val_split = train.train_test_split(
            test_size=SPLITS["val"] / (1 - SPLITS["test"]), seed=SEED
        )
        train, val = train_val_split["train"], train_val_split["test"]

    elif val is None:
        train_val_split = train.train_test_split(
            test_size=SPLITS["val"] / (1 - SPLITS["test"]), seed=SEED
        )
        train, val = train_val_split["train"], train_val_split["test"]

    return DatasetDict(
        {"train": train, "test": cast(Dataset, test), "val": cast(Dataset, val)}
    )


def merge_datasets() -> DatasetDict:
    dataset_root = Path("./data/processed/datasets/")
    dataset_dicts = {}

    for dataset in DATASETS:
        spec = DATASETS[dataset]
        key = spec.key
        path = dataset_root / key
        dataset_dicts[key] = get_dataset_dict(path, spec)

    train = concatenate_datasets([ds["train"] for ds in dataset_dicts.values()])
    test = concatenate_datasets([ds["test"] for ds in dataset_dicts.values()])
    val = concatenate_datasets([ds["val"] for ds in dataset_dicts.values()])

    return DatasetDict({"train": train, "test": test, "val": val})


def main() -> None:
    args = parse_args()
    datasets_dict = merge_datasets()
    datasets_dict.push_to_hub(repo_id=args.repo_id, private=True, token=HF_TOKEN)


if __name__ == "__main__":
    main()
