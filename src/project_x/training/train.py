"""Run fine-tuning using configured datasets, curriculum weights, and adapters."""

import os
import time
from dataclasses import asdict

import torch
from accelerate import Accelerator
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

from project_x.constants import WANDB_ENTITY, WANDB_PROJECT_NAME
from project_x.data.datasets import get_tobias_dataset, get_tobias_repair_dataset
from project_x.data.loaders import (
    get_img2svg_dataloader,
    get_repair_dataloader,
    get_text2svg_dataloader,
)
from project_x.modeling.loading import get_model
from project_x.training.config import training_config


def build_peft_model():
    peft_config = LoraConfig(
        r=training_config.LORA_RANK,
        lora_alpha=training_config.LORA_ALPHA,
        lora_dropout=training_config.LORA_DROPOUT,
        target_modules=training_config.LORA_TARGET_MODULES,  # type: ignore
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # Model Initialization
    model = get_model()
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


def batch_stats(batch):
    seq_len = batch["input_ids"].shape[1]
    label_tokens = int((batch["labels"] != -100).sum().item())
    visual_tokens = 0

    if "image_grid_thw" in batch:
        visual_tokens = int(batch["image_grid_thw"][:, 1:].prod(dim=1).sum().item())

    return seq_len, label_tokens, visual_tokens


def repeat_dataloader(dataloader):
    while True:
        yield from dataloader


def main():
    max_train_steps = int(
        os.environ.get("PROJECT_X_MAX_TRAIN_STEPS", training_config.MAX_TRAIN_STEPS)
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=training_config.GRADIENT_ACCUMULATION_STEPS,
        log_with="wandb",
    )
    tracker_config = asdict(training_config)
    tracker_config["MAX_TRAIN_STEPS"] = max_train_steps
    accelerator.init_trackers(
        project_name=WANDB_PROJECT_NAME,
        config=tracker_config,
        init_kwargs={"wandb": {"entity": WANDB_ENTITY}},
    )

    # Dataloaders
    tobias_dataset = get_tobias_dataset()
    repair_dataset = get_tobias_repair_dataset()
    text_train, text_test, text_val = get_text2svg_dataloader(
        tobias_dataset, batch_size=training_config.MICRO_BATCH_SIZE
    )
    image_train, image_test, image_val = get_img2svg_dataloader(
        tobias_dataset, batch_size=training_config.MICRO_BATCH_SIZE
    )
    repair_train, repair_test, repair_val = get_repair_dataloader(
        repair_dataset, batch_size=training_config.MICRO_BATCH_SIZE
    )

    model = build_peft_model()

    # Optimizer
    optimizer = AdamW(
        params=model.parameters(),
        lr=training_config.LR,
        weight_decay=training_config.WEIGHT_DECAY,
    )

    # Scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=training_config.WARMUP_STEPS,
        num_training_steps=max_train_steps,
    )

    (
        model,
        optimizer,
        text_train,
        text_test,
        text_val,
        image_train,
        image_test,
        image_val,
        repair_train,
        repair_test,
        repair_val,
        scheduler,
    ) = accelerator.prepare(
        model,
        optimizer,
        text_train,
        text_test,
        text_val,
        image_train,
        image_test,
        image_val,
        repair_train,
        repair_test,
        repair_val,
        scheduler,
    )

    train_loaders = [
        ("text", repeat_dataloader(text_train)),
        ("image", repeat_dataloader(image_train)),
        ("repair", repeat_dataloader(repair_train)),
    ]
    if len(train_loaders) != training_config.GRADIENT_ACCUMULATION_STEPS:
        raise ValueError(
            "GRADIENT_ACCUMULATION_STEPS must match the number of training tasks."
        )

    completed_steps = 0
    step_metrics: dict[str, float | int] = {}
    step_peak_vram = 0.0
    step_start_time = time.perf_counter()

    while completed_steps < max_train_steps:
        for (
            task_name,
            dataloader,
        ) in train_loaders:  # This steps over each of the three datasets
            batch = next(dataloader)
            seq_len, label_tokens, visual_tokens = batch_stats(batch)

            torch.cuda.reset_peak_memory_stats()

            with accelerator.accumulate(model):
                output = model(**batch)
                loss = output.loss
                accelerator.backward(loss)

                step_metrics[f"loss/{task_name}"] = loss.item()
                step_metrics[f"batch/{task_name}_sequence_length"] = seq_len
                step_metrics[f"batch/{task_name}_label_tokens"] = label_tokens
                step_metrics[f"batch/{task_name}_visual_tokens"] = visual_tokens

                peak_memory = torch.cuda.max_memory_allocated() / 1024**3
                step_peak_vram = max(step_peak_vram, peak_memory)

                if accelerator.sync_gradients:
                    gradient_norm = accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    completed_steps += 1
                    step_metrics["train/learning_rate"] = scheduler.get_last_lr()[0]
                    step_metrics["train/step_time_seconds"] = (
                        time.perf_counter() - step_start_time
                    )
                    step_metrics["train/gradient_norm"] = float(gradient_norm)
                    step_metrics["system/pytorch_peak_vram_gb"] = step_peak_vram

                    accelerator.log(step_metrics, step=completed_steps)

                    should_print = (
                        completed_steps == 1
                        or completed_steps % training_config.LOG_EVERY_STEPS == 0
                        or completed_steps == max_train_steps
                    )
                    if should_print:
                        accelerator.print(
                            f"step {completed_steps}/{max_train_steps} "
                            f"text={step_metrics['loss/text']:.4f} "
                            f"image={step_metrics['loss/image']:.4f} "
                            f"repair={step_metrics['loss/repair']:.4f}"
                        )

                    step_metrics = {}
                    step_peak_vram = 0.0
                    step_start_time = time.perf_counter()
                    if completed_steps >= max_train_steps:
                        break

    accelerator.end_training()


if __name__ == "__main__":
    main()
