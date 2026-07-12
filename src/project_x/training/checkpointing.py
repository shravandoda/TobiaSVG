"""Save and resume adapter training checkpoints."""

import shutil
from pathlib import Path

from accelerate import Accelerator
from peft import load_peft_weights, set_peft_model_state_dict

from project_x.training.config import training_config

CHECKPOINT_PREFIX = "checkpoint_"


def register_peft_checkpoint_hooks(accelerator: Accelerator) -> None:
    """Make Accelerate checkpoint only the adapter instead of the frozen base."""

    def save_model_hook(models, weights, output_dir):
        model = accelerator.unwrap_model(models[0])
        if accelerator.is_main_process:
            model.save_pretrained(
                Path(output_dir) / "adapter",
                state_dict=weights[0],
                safe_serialization=True,
            )
        weights.clear()

    def load_model_hook(models, input_dir):
        model = accelerator.unwrap_model(models.pop())
        adapter_weights = load_peft_weights(
            str(Path(input_dir) / "adapter"),
            device=str(accelerator.device),
        )
        set_peft_model_state_dict(model, adapter_weights)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)


def sort_checkpoints(project_dir: Path) -> list[tuple[int, Path]]:
    checkpoint_root = project_dir / "checkpoints"
    if not checkpoint_root.exists():
        return []

    checkpoints = []
    for checkpoint_dir in checkpoint_root.glob(f"{CHECKPOINT_PREFIX}*"):
        step_text = checkpoint_dir.name.removeprefix(CHECKPOINT_PREFIX)
        if step_text.isdigit():
            checkpoints.append((int(step_text), checkpoint_dir))

    checkpoints.sort(reverse=True)
    return checkpoints


def resume_latest_checkpoint(
    accelerator: Accelerator,
    project_dir: Path,
) -> int:
    checkpoints = sort_checkpoints(project_dir)
    if not checkpoints:
        return 0
    _, latest_checkpoint = checkpoints[0]

    accelerator.print(f"resuming from checkpoint: {latest_checkpoint}")
    accelerator.load_state(str(latest_checkpoint))
    return int(latest_checkpoint.name.removeprefix(CHECKPOINT_PREFIX))


def save_checkpoint(
    accelerator: Accelerator,
    project_dir: Path,
    completed_steps: int,
) -> None:
    checkpoint_root = project_dir / "checkpoints"
    checkpoint_dir = checkpoint_root / f"{CHECKPOINT_PREFIX}{completed_steps:06d}"

    if checkpoint_dir.exists():
        accelerator.print(f"checkpoint already exists: {checkpoint_dir}")
        return

    accelerator.save_state(str(checkpoint_dir))
    accelerator.print(f"saved checkpoint: {checkpoint_dir}")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        checkpoints = sort_checkpoints(project_dir)
        to_delete = checkpoints[training_config.KEEP_LAST_CHECKPOINTS :]
        for _, checkpoint_dir in to_delete:
            shutil.rmtree(checkpoint_dir)
    accelerator.wait_for_everyone()


def save_final_adapter(
    accelerator: Accelerator,
    model,
    project_dir: Path,
) -> None:
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.unwrap_model(model).save_pretrained(
            project_dir / "final_adapter",
            safe_serialization=True,
        )
    accelerator.wait_for_everyone()
