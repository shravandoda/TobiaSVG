"""Save and resume adapter training checkpoints."""

import shutil
from pathlib import Path

from accelerate import Accelerator
from accelerate.checkpointing import save_accelerator_state
from peft import load_peft_weights, set_peft_model_state_dict

from project_x.training.config import training_config

CHECKPOINT_PREFIX = "checkpoint_"


def register_peft_load_hook(accelerator: Accelerator) -> None:
    """Load adapter weights before Accelerate restores the remaining state."""

    def load_model_hook(models, input_dir):
        model = accelerator.unwrap_model(models.pop())
        adapter_weights = load_peft_weights(
            str(Path(input_dir) / "adapter"),
            device=str(accelerator.device),
        )
        set_peft_model_state_dict(model, adapter_weights)

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
    model,
    project_dir: Path,
    completed_steps: int,
) -> None:
    checkpoint_root = project_dir / "checkpoints"
    checkpoint_dir = checkpoint_root / f"{CHECKPOINT_PREFIX}{completed_steps:06d}"
    incomplete_dir = checkpoint_root / f".{checkpoint_dir.name}.incomplete"

    if checkpoint_dir.exists():
        accelerator.print(f"checkpoint already exists: {checkpoint_dir}")
        return

    if accelerator.is_main_process:
        if incomplete_dir.exists():
            shutil.rmtree(incomplete_dir)
        incomplete_dir.mkdir(parents=True)
        accelerator.unwrap_model(model).save_pretrained(
            incomplete_dir / "adapter",
            safe_serialization=True,
        )
    accelerator.wait_for_everyone()

    # Accelerator.save_state() first materializes the entire frozen base model.
    # Save the non-model state directly after writing the small PEFT adapter.
    save_accelerator_state(
        output_dir=str(incomplete_dir),
        model_states=[],
        optimizers=accelerator._optimizers,
        schedulers=accelerator._schedulers,
        dataloaders=accelerator._dataloaders,
        process_index=accelerator.state.process_index,
        step=accelerator.step,
        scaler=accelerator.scaler,
        save_on_each_node=accelerator.project_configuration.save_on_each_node,
        safe_serialization=True,
    )

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        incomplete_dir.rename(checkpoint_dir)
        checkpoints = sort_checkpoints(project_dir)
        to_delete = checkpoints[training_config.KEEP_LAST_CHECKPOINTS :]
        for _, checkpoint_dir in to_delete:
            shutil.rmtree(checkpoint_dir)
    accelerator.wait_for_everyone()
    accelerator.print(f"saved checkpoint: {checkpoint_dir}")


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
