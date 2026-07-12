from types import SimpleNamespace

import torch

from project_x.training.checkpointing import sort_checkpoints
from project_x.training.train import repeat_dataloader, validate


class EpochLoader:
    def __init__(self):
        self.epoch = 0

    def __iter__(self):
        self.epoch += 1
        yield self.epoch, "first"
        yield self.epoch, "second"


class ValidationModel:
    def __init__(self):
        self.training = True

    def eval(self):
        self.training = False

    def train(self):
        self.training = True

    def __call__(self, loss):
        return SimpleNamespace(loss=torch.tensor(loss))


class ValidationAccelerator:
    device = torch.device("cpu")

    @staticmethod
    def reduce(value, reduction):
        assert reduction == "sum"
        return value


def test_repeat_dataloader_restarts_without_caching_an_epoch():
    loader = EpochLoader()
    iterator = repeat_dataloader(loader)

    assert [next(iterator) for _ in range(5)] == [
        (1, "first"),
        (1, "second"),
        (2, "first"),
        (2, "second"),
        (3, "first"),
    ]


def test_validate_averages_batches_and_tasks():
    model = ValidationModel()
    validation_loaders = (
        ("text", [{"loss": 1.0}, {"loss": 3.0}]),
        ("image", [{"loss": 2.0}, {"loss": 4.0}]),
        ("repair", [{"loss": 3.0}, {"loss": 5.0}]),
    )

    metrics = validate(
        ValidationAccelerator(),
        model,
        validation_loaders,
        max_batches=2,
    )

    assert metrics == {
        "validation/loss/text": 2.0,
        "validation/loss/image": 3.0,
        "validation/loss/repair": 4.0,
        "validation/loss": 3.0,
    }
    assert model.training is True


def test_sort_checkpoints_orders_highest_step_first(tmp_path):
    checkpoint_root = tmp_path / "checkpoints"
    first = checkpoint_root / "checkpoint_001000"
    latest = checkpoint_root / "checkpoint_002000"
    first.mkdir(parents=True)
    latest.mkdir()

    assert sort_checkpoints(tmp_path) == [(2000, latest), (1000, first)]


def test_sort_checkpoints_returns_empty_list_when_none_exist(tmp_path):
    assert sort_checkpoints(tmp_path) == []
