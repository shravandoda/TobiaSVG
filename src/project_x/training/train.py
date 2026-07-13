"""Fine-tune the SVG model across text, image, and repair tasks."""

from dataclasses import asdict
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import GradientAccumulationPlugin, set_seed
from liger_kernel.transformers import apply_liger_kernel_to_qwen3_5
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

from project_x.constants import WANDB_ENTITY, WANDB_PROJECT_NAME
from project_x.data.datasets import get_tobias_dataset, get_tobias_repair_dataset
from project_x.data.loaders import (
    get_img2svg_dataloader,
    get_repair_dataloader,
    get_text2svg_dataloader,
    split_generation_train_rows,
)
from project_x.modeling.loading import get_model
from project_x.training.checkpointing import (
    push_folder_to_hub,
    register_peft_load_hook,
    resume_latest_checkpoint,
    save_checkpoint,
    save_final_adapter,
)
from project_x.training.config import training_config


def build_accelerator(max_train_steps: int) -> Accelerator:
    accumulation = GradientAccumulationPlugin(
        num_steps=training_config.GRADIENT_ACCUMULATION_STEPS,
        sync_with_dataloader=False,
    )
    accelerator = Accelerator(
        gradient_accumulation_plugin=accumulation,
        log_with="wandb",
        step_scheduler_with_optimizer=False,
    )

    tracker_config = asdict(training_config)
    tracker_config["MAX_TRAIN_STEPS"] = max_train_steps
    accelerator.init_trackers(
        project_name=WANDB_PROJECT_NAME,
        config=tracker_config,
        init_kwargs={"wandb": {"entity": WANDB_ENTITY}},
    )
    return accelerator


def build_model():
    apply_liger_kernel_to_qwen3_5(
        rms_norm=True,
        swiglu=True,
        cross_entropy=False,
        fused_linear_cross_entropy=True,
    )

    lora_config = LoraConfig(
        r=training_config.LORA_RANK,
        lora_alpha=training_config.LORA_ALPHA,
        lora_dropout=training_config.LORA_DROPOUT,
        target_modules=training_config.LORA_TARGET_MODULES,  # type: ignore
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_model()
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def build_training_loaders():
    dataset = get_tobias_dataset()
    repair_dataset = get_tobias_repair_dataset()
    text_dataset, image_dataset = split_generation_train_rows(
        dataset,
        seed=training_config.SEED,
    )

    text_train, _, text_val = get_text2svg_dataloader(
        text_dataset,
        batch_size=training_config.MICRO_BATCH_SIZE,
        preprocessing_workers=training_config.PREPROCESSING_WORKERS,
    )
    image_train, _, image_val = get_img2svg_dataloader(
        image_dataset,
        batch_size=training_config.MICRO_BATCH_SIZE,
        preprocessing_workers=training_config.PREPROCESSING_WORKERS,
    )
    repair_train, _, repair_val = get_repair_dataloader(
        repair_dataset,
        batch_size=training_config.MICRO_BATCH_SIZE,
        preprocessing_workers=training_config.PREPROCESSING_WORKERS,
    )

    return (
        text_train,
        image_train,
        repair_train,
        text_val,
        image_val,
        repair_val,
    )


def repeat_dataloader(dataloader):
    while True:
        yield from dataloader


def restart_scheduler(
    scheduler,
    optimizer,
    learning_rate: float,
    warmup_steps: int,
    training_steps: int,
) -> None:
    """Start a fresh schedule while retaining the resumed optimizer state."""
    if training_steps <= 0:
        raise ValueError("The restarted scheduler requires remaining training steps")

    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate
        param_group["initial_lr"] = learning_rate

    scheduler.scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=training_steps,
    )


@torch.inference_mode()
def validate(
    accelerator,
    model,
    validation_loaders,
    max_batches: int = 50,
):
    model.eval()
    metrics = {}

    for task_name, dataloader in validation_loaders:
        loss_sum = torch.zeros((), device=accelerator.device)
        batch_count = torch.zeros((), device=accelerator.device)

        for batch_index, batch in enumerate(dataloader):
            if batch_index >= max_batches:
                break

            # Gives me the mean loss across the batch and the non-masked tokens
            output = model(**batch)

            # Sums the loss for the given task_name across all the batches
            loss_sum += output.loss.detach()
            batch_count += 1

        loss_sum = accelerator.reduce(loss_sum, reduction="sum")
        batch_count = accelerator.reduce(batch_count, reduction="sum")
        metrics[f"validation/loss/{task_name}"] = (
            loss_sum / batch_count.clamp_min(1)
        ).item()

    metrics["validation/loss"] = sum(metrics.values()) / len(validation_loaders)
    model.train()
    return metrics


def train(
    accelerator,
    model,
    optimizer,
    scheduler,
    train_loaders,
    validation_loaders,
    completed_steps,
    max_train_steps,
    project_dir,
):
    model.train()
    while completed_steps < max_train_steps:
        losses = {}

        # One optimizer update uses one batch from each task.
        for task_name, dataloader in train_loaders:
            batch = next(dataloader)

            with accelerator.accumulate(model):
                output = model(**batch)

                # Mean loss over non-masked target tokens in this GPU's local batch.
                loss = output.loss

                # Keep the scalar for logging without retaining its computation graph.
                losses[task_name] = loss.detach()

                # Scale by 1/3 (accumulation steps = 3) and compute gradients.
                # On the third task, DDP averages the accumulated gradients
                # across GPUs.
                accelerator.backward(loss)

                # This is true after all three task gradients are accumulated.
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    completed_steps += 1

        should_log = (
            completed_steps == 1
            or completed_steps % training_config.LOG_EVERY_STEPS == 0
            or completed_steps == max_train_steps
        )
        if should_log:
            metrics = {
                f"loss/{task_name}": accelerator.reduce(
                    loss,
                    reduction="mean",
                ).item()
                for task_name, loss in losses.items()
            }
            metrics["train/learning_rate"] = scheduler.get_last_lr()[0]
            accelerator.log(metrics, step=completed_steps)
            accelerator.print(
                f"step {completed_steps}/{max_train_steps} "
                f"text={metrics['loss/text']:.4f} "
                f"image={metrics['loss/image']:.4f} "
                f"repair={metrics['loss/repair']:.4f}"
            )

        if completed_steps % training_config.VALIDATE_EVERY_STEPS == 0:
            validation_metrics = validate(
                accelerator,
                model,
                validation_loaders,
                max_batches=training_config.MAX_VALIDATION_BATCHES,
            )
            accelerator.log(validation_metrics, step=completed_steps)
            accelerator.print(
                f"validation step {completed_steps}: "
                f"loss={validation_metrics['validation/loss']:.4f}"
            )

        if completed_steps % training_config.SAVE_EVERY_STEPS == 0:
            checkpoint_dir = save_checkpoint(
                accelerator,
                model,
                project_dir,
                completed_steps,
            )
            if (
                training_config.HUB_REPO_ID
                and completed_steps % training_config.PUSH_TO_HUB_EVERY_STEPS == 0
            ):
                push_folder_to_hub(
                    accelerator,
                    checkpoint_dir,
                    training_config.HUB_REPO_ID,
                    path_in_repo=f"checkpoints/{checkpoint_dir.name}",
                    commit_message=f"Training checkpoint at step {completed_steps}",
                )


def main():
    project_dir = Path(training_config.PROJECT_DIR)

    accelerator = build_accelerator(training_config.MAX_TRAIN_STEPS)
    set_seed(training_config.SEED)

    with accelerator.main_process_first():
        (
            text_train,
            image_train,
            repair_train,
            text_val,
            image_val,
            repair_val,
        ) = build_training_loaders()

    model = build_model()

    optimizer = AdamW(
        model.parameters(),
        lr=training_config.LR,
        weight_decay=training_config.WEIGHT_DECAY,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=training_config.WARMUP_STEPS,
        num_training_steps=training_config.MAX_TRAIN_STEPS,
    )

    (
        model,
        optimizer,
        scheduler,
        text_train,
        image_train,
        repair_train,
        text_val,
        image_val,
        repair_val,
    ) = accelerator.prepare(
        model,
        optimizer,
        scheduler,
        text_train,
        image_train,
        repair_train,
        text_val,
        image_val,
        repair_val,
    )

    train_loaders = (
        ("text", repeat_dataloader(text_train)),
        ("image", repeat_dataloader(image_train)),
        ("repair", repeat_dataloader(repair_train)),
    )
    validation_loaders = (
        ("text", text_val),
        ("image", image_val),
        ("repair", repair_val),
    )

    register_peft_load_hook(accelerator)
    resume_checkpoint = (
        Path(training_config.RESUME_CHECKPOINT)
        if training_config.RESUME_CHECKPOINT
        else None
    )
    completed_steps = resume_latest_checkpoint(
        accelerator,
        project_dir,
        checkpoint_path=resume_checkpoint,
    )
    if completed_steps and training_config.RESET_SCHEDULER_ON_RESUME:
        remaining_steps = training_config.MAX_TRAIN_STEPS - completed_steps
        restart_scheduler(
            scheduler,
            optimizer,
            learning_rate=training_config.LR,
            warmup_steps=training_config.WARMUP_STEPS,
            training_steps=remaining_steps,
        )
        accelerator.print(
            "restarted scheduler: "
            f"lr={training_config.LR:g} "
            f"warmup_steps={training_config.WARMUP_STEPS} "
            f"training_steps={remaining_steps}"
        )
    train(
        accelerator,
        model,
        optimizer,
        scheduler,
        train_loaders,
        validation_loaders,
        completed_steps,
        training_config.MAX_TRAIN_STEPS,
        project_dir,
    )
    final_adapter_dir = save_final_adapter(accelerator, model, project_dir)
    if training_config.HUB_REPO_ID:
        push_folder_to_hub(
            accelerator,
            final_adapter_dir,
            training_config.HUB_REPO_ID,
            path_in_repo="",
            commit_message="Final adapter",
        )
    accelerator.end_training()


if __name__ == "__main__":
    main()
