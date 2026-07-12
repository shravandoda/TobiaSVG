from types import SimpleNamespace

import torch

from project_x.training import checkpointing
from project_x.training.checkpointing import (
    push_folder_to_hub,
    save_checkpoint,
    sort_checkpoints,
)
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


class CheckpointModel:
    def save_pretrained(self, output_dir, safe_serialization):
        assert safe_serialization is True
        output_dir.mkdir(parents=True)
        (output_dir / "adapter_model.safetensors").touch()


class CheckpointAccelerator:
    is_main_process = True
    _optimizers = ["optimizer"]
    _schedulers = ["scheduler"]
    _dataloaders = ["dataloader"]
    state = SimpleNamespace(process_index=0)
    step = 10
    scaler = None
    project_configuration = SimpleNamespace(save_on_each_node=False)

    @staticmethod
    def unwrap_model(model):
        return model

    @staticmethod
    def wait_for_everyone():
        pass

    @staticmethod
    def print(message):
        pass


def test_save_checkpoint_skips_frozen_model_state(tmp_path, monkeypatch):
    saved_state = {}

    def fake_save_accelerator_state(**kwargs):
        saved_state.update(kwargs)

    monkeypatch.setattr(
        checkpointing,
        "save_accelerator_state",
        fake_save_accelerator_state,
    )

    save_checkpoint(
        CheckpointAccelerator(),
        CheckpointModel(),
        tmp_path,
        completed_steps=10,
    )

    checkpoint_dir = tmp_path / "checkpoints" / "checkpoint_000010"
    assert (checkpoint_dir / "adapter" / "adapter_model.safetensors").exists()
    assert saved_state["model_states"] == []
    assert saved_state["optimizers"] == ["optimizer"]
    assert not (tmp_path / "checkpoints" / ".checkpoint_000010.incomplete").exists()


def test_push_folder_to_hub_uploads_model_repo(tmp_path, monkeypatch):
    calls = {}

    class FakeApi:
        def create_repo(self, **kwargs):
            calls["create_repo"] = kwargs

        def upload_folder(self, **kwargs):
            calls["upload_folder"] = kwargs

    monkeypatch.setattr(checkpointing, "HfApi", FakeApi)

    push_folder_to_hub(
        CheckpointAccelerator(),
        tmp_path,
        "user/model",
        "checkpoints/checkpoint_005000",
        "Training checkpoint at step 5000",
    )

    assert calls["create_repo"] == {
        "repo_id": "user/model",
        "repo_type": "model",
        "private": True,
        "exist_ok": True,
    }
    assert calls["upload_folder"] == {
        "repo_id": "user/model",
        "repo_type": "model",
        "folder_path": tmp_path,
        "path_in_repo": "checkpoints/checkpoint_005000",
        "commit_message": "Training checkpoint at step 5000",
    }
