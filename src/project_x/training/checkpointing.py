"""Save and resume adapter training checkpoints."""

import shutil
from pathlib import Path

from accelerate import Accelerator
from accelerate.checkpointing import save_accelerator_state
from huggingface_hub import HfApi
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
    checkpoint_path: Path | None = None,
) -> int:
    if checkpoint_path is None:
        checkpoints = sort_checkpoints(project_dir)
        if not checkpoints:
            return 0
        _, checkpoint_path = checkpoints[0]

    step_text = checkpoint_path.name.removeprefix(CHECKPOINT_PREFIX)
    if not checkpoint_path.is_dir() or not step_text.isdigit():
        raise ValueError(f"Invalid checkpoint directory: {checkpoint_path}")

    accelerator.print(f"resuming from checkpoint: {checkpoint_path}")
    accelerator.load_state(str(checkpoint_path))
    return int(step_text)


def save_checkpoint(
    accelerator: Accelerator,
    model,
    project_dir: Path,
    completed_steps: int,
) -> Path:
    checkpoint_root = project_dir / "checkpoints"
    checkpoint_dir = checkpoint_root / f"{CHECKPOINT_PREFIX}{completed_steps:06d}"
    incomplete_dir = checkpoint_root / f".{checkpoint_dir.name}.incomplete"

    if checkpoint_dir.exists():
        accelerator.print(f"checkpoint already exists: {checkpoint_dir}")
        return checkpoint_dir

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
        for _, old_checkpoint_dir in to_delete:
            shutil.rmtree(old_checkpoint_dir)
    accelerator.wait_for_everyone()
    accelerator.print(f"saved checkpoint: {checkpoint_dir}")
    return checkpoint_dir


def save_final_adapter(
    accelerator: Accelerator,
    model,
    project_dir: Path,
) -> Path:
    adapter_dir = project_dir / "final_adapter"
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.unwrap_model(model).save_pretrained(
            adapter_dir,
            safe_serialization=True,
        )
    accelerator.wait_for_everyone()
    return adapter_dir


def push_folder_to_hub(
    accelerator: Accelerator,
    folder: Path,
    repo_id: str,
    path_in_repo: str,
    commit_message: str,
) -> None:
    """Upload training artifacts without interrupting training on Hub failures."""
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        try:
            api = HfApi()
            api.create_repo(
                repo_id=repo_id,
                repo_type="model",
                private=True,
                exist_ok=True,
            )
            api.upload_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=folder,
                path_in_repo=path_in_repo,
                commit_message=commit_message,
            )
            accelerator.print(f"pushed training artifacts to Hub: {repo_id}")
        except Exception as error:
            accelerator.print(f"Hub push failed: {error}")
    accelerator.wait_for_everyone()
