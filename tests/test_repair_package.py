from datasets import Dataset, DatasetDict

from project_x.preprocessing.repair.package import split_dataset, split_source_dataset


def test_split_source_dataset_keeps_identical_svgs_together():
    source = Dataset.from_dict(
        {
            "svg": ["duplicate-svg", "unique-svg", "duplicate-svg"],
            "row_id": [1, 2, 3],
        }
    )

    result = split_source_dataset(source)

    duplicate_splits = {
        split_name
        for split_name, split in result.items()
        if "duplicate-svg" in split["svg"]
    }
    assert len(duplicate_splits) == 1
    assert sum(len(split) for split in result.values()) == 3


def test_split_dataset_keeps_repair_variants_with_their_source_split():
    source = DatasetDict(
        {
            "train": Dataset.from_dict({"svg": ["train-svg"]}),
            "test": Dataset.from_dict({"svg": ["test-svg"]}),
            "val": Dataset.from_dict({"svg": ["val-svg"]}),
        }
    )
    repair = Dataset.from_dict(
        {
            "svg": ["test-svg", "train-svg", "test-svg", "val-svg"],
            "corrupted_svg": ["test-a", "train-a", "test-b", "val-a"],
        }
    )

    result = split_dataset(repair, source)

    assert set(result["train"]["svg"]) == {"train-svg"}
    assert set(result["test"]["svg"]) == {"test-svg"}
    assert set(result["test"]["corrupted_svg"]) == {"test-a", "test-b"}
    assert set(result["val"]["svg"]) == {"val-svg"}


def test_split_dataset_prefers_train_for_source_duplicates():
    source = DatasetDict(
        {
            "train": Dataset.from_dict({"svg": ["duplicate-svg"]}),
            "test": Dataset.from_dict({"svg": ["duplicate-svg"]}),
            "val": Dataset.from_dict({"svg": []}),
        }
    )
    repair = Dataset.from_dict(
        {
            "svg": ["duplicate-svg", "duplicate-svg"],
            "corrupted_svg": ["first", "second"],
        }
    )

    result = split_dataset(repair, source)

    assert len(result["train"]) == 2
    assert len(result["test"]) == 0
    assert len(result["val"]) == 0
