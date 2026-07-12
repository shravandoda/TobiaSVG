from datasets import Dataset, DatasetDict
import torch

from project_x.data.collators import _build_training_batch
from project_x.data.loaders import (
    _prepare_task_dataset,
    split_generation_train_rows,
)


class FakeTokenizer:
    eos_token = "<eos>"
    pad_token_id = 0

    def __call__(self, text, *, add_special_tokens):
        assert text.endswith("<eos>\n")
        assert add_special_tokens is False
        return {"input_ids": [7, 8]}


class FakeProcessor:
    tokenizer = FakeTokenizer()


def build_splits() -> DatasetDict:
    split = Dataset.from_dict(
        {
            "text": ["short", "long"],
            "svg": ["<svg/>", "<svg/>"],
            "measured_length": [100, 13_000],
            "unused": [1, 2],
        }
    )
    return DatasetDict({name: split for name in ("train", "test", "val")})


def test_prepare_task_dataset_filters_and_selects_columns():
    prepared = _prepare_task_dataset(
        build_splits(),
        sequence_length_fn=lambda row: row["measured_length"],
        columns=("text", "svg"),
        task_name="test",
    )

    assert {name: len(split) for name, split in prepared.items()} == {
        "train": 1,
        "test": 1,
        "val": 1,
    }
    assert all(split.column_names == ["text", "svg"] for split in prepared.values())
    assert prepared["train"][0] == {"text": "short", "svg": "<svg/>"}


def test_prepare_task_dataset_requires_all_splits():
    dataset = DatasetDict({"train": Dataset.from_dict({"text": ["row"]})})

    try:
        _prepare_task_dataset(
            dataset,
            sequence_length_fn=lambda row: 1,
            columns=("text",),
            task_name="test",
        )
    except ValueError as error:
        assert str(error) == "Dataset is missing required splits: test, val"
    else:
        raise AssertionError("Missing dataset splits were accepted")


def test_split_generation_train_rows_is_disjoint_and_source_balanced():
    train = Dataset.from_dict(
        {
            "row_id": list(range(8)),
            "dataset": ["shapes"] * 4 + ["diagrams"] * 4,
        }
    )
    evaluation = Dataset.from_dict({"row_id": [8], "dataset": ["evaluation"]})
    dataset = DatasetDict({"train": train, "test": evaluation, "val": evaluation})

    text_dataset, image_dataset = split_generation_train_rows(dataset, seed=7)

    text_ids = set(text_dataset["train"]["row_id"])
    image_ids = set(image_dataset["train"]["row_id"])
    assert len(text_ids) == len(image_ids) == 4
    assert text_ids.isdisjoint(image_ids)
    assert text_ids | image_ids == set(range(8))
    assert text_dataset["train"]["dataset"].count("shapes") == 2
    assert image_dataset["train"]["dataset"].count("shapes") == 2
    assert text_dataset["test"][:] == evaluation[:]
    assert image_dataset["val"][:] == evaluation[:]


def test_build_training_batch_masks_prompts_and_pads_targets():
    prompt_batch = {
        "input_ids": torch.tensor([[1, 2, 0], [3, 4, 5]]),
        "attention_mask": torch.tensor([[1, 1, 0], [1, 1, 1]]),
        "mm_token_type_ids": torch.zeros((2, 3), dtype=torch.long),
    }

    batch = _build_training_batch(
        FakeProcessor(),
        prompt_batch,
        ["<svg/>", "<svg/>"],
    )

    assert batch["input_ids"].tolist() == [
        [1, 2, 7, 8, 0],
        [3, 4, 5, 7, 8],
    ]
    assert batch["labels"].tolist() == [
        [-100, -100, 7, 8, -100],
        [-100, -100, -100, 7, 8],
    ]
    assert batch["attention_mask"].tolist() == [
        [1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
    ]
