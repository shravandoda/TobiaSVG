"""Run fine-tuning using configured datasets, curriculum weights, and adapters."""

from itertools import cycle

from accelerate import Accelerator
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

from src.project_x.data.datasets import get_tobias_dataset, get_tobias_repair_dataset
from src.project_x.data.loaders import (
    get_img2svg_dataloader,
    get_repair_dataloader,
    get_text2svg_dataloader,
)
from src.project_x.modeling.loading import get_model
from src.project_x.training.config import training_config

accelerator = Accelerator(
    gradient_accumulation_steps=training_config.GRADIENT_ACCUMULATION_STEPS
)


def build_peft_model():
    peft_config = LoraConfig(
        r=training_config.LORA_RANK,
        lora_alpha=training_config.LORA_ALPHA,
        lora_dropout=training_config.LORA_DROPOUT,
        target_modules=training_config.LORA_TARGET_MODULES,
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


def main():
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
        num_training_steps=training_config.TOTAL_STEPS,
    )

    # Dataloaders
    text_train, text_test, text_val = get_text2svg_dataloader(
        get_tobias_dataset(), batch_size=training_config.MICRO_BATCH_SIZE
    )
    image_train, image_test, image_val = get_img2svg_dataloader(
        get_tobias_dataset(), batch_size=training_config.MICRO_BATCH_SIZE
    )
    repair_train, repair_test, repair_val = get_repair_dataloader(
        get_tobias_repair_dataset(), batch_size=training_config.MICRO_BATCH_SIZE
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
        ("text", cycle(text_train)),
        ("image", cycle(image_train)),
        ("repair", cycle(repair_train)),
    ]
    completed_steps = 0

    while completed_steps < training_config.MAX_TRAIN_STEPS:
        for task_name, dataloader in train_loaders:
            batch = next(dataloader)
            with accelerator.accumulate(model):
                output = model(**batch)
                loss = output.loss
                accelerator.backward(loss)

                accelerator.print(f"{task_name} loss: {loss.item():.4f}")

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                if accelerator.sync_gradients:
                    completed_steps += 1
                    accelerator.print(f"completed step: {completed_steps}")
                    if completed_steps >= training_config.MAX_TRAIN_STEPS:
                        break


if __name__ == "__main__":
    main()
