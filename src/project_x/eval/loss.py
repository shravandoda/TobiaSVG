"""Measure teacher-forced validation loss for a base model or LoRA adapter."""

import argparse
import json
from pathlib import Path

from accelerate import Accelerator
from datasets import DatasetDict
from liger_kernel.transformers import apply_liger_kernel_to_qwen3_5
from peft import PeftModel

from project_x.data.datasets import get_tobias_dataset, get_tobias_repair_dataset
from project_x.data.loaders import (
    get_img2svg_dataloader,
    get_repair_dataloader,
    get_text2svg_dataloader,
)
from project_x.modeling.loading import get_model
from project_x.training.config import training_config
from project_x.training.train import validate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adapter-path",
        type=Path,
        help="Saved PEFT adapter directory. Omit to evaluate the base model.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=500,
        help="Maximum validation batches per task and process.",
    )
    return parser.parse_args()


def load_evaluation_model(adapter_path: Path | None):
    apply_liger_kernel_to_qwen3_5(
        rms_norm=True,
        swiglu=True,
        cross_entropy=False,
        fused_linear_cross_entropy=True,
    )

    model = get_model()
    model.config.use_cache = False
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, adapter_path)
    return model


def validation_only(dataset: DatasetDict) -> DatasetDict:
    validation = dataset["val"]
    placeholder = validation.select(range(min(1, len(validation))))
    return DatasetDict({"train": placeholder, "test": placeholder, "val": validation})


def build_validation_loaders():
    dataset = validation_only(get_tobias_dataset())
    repair_dataset = validation_only(get_tobias_repair_dataset())

    _, _, text_val = get_text2svg_dataloader(
        dataset,
        batch_size=training_config.MICRO_BATCH_SIZE,
        preprocessing_workers=training_config.PREPROCESSING_WORKERS,
    )
    _, _, image_val = get_img2svg_dataloader(
        dataset,
        batch_size=training_config.MICRO_BATCH_SIZE,
        preprocessing_workers=training_config.PREPROCESSING_WORKERS,
    )
    _, _, repair_val = get_repair_dataloader(
        repair_dataset,
        batch_size=training_config.MICRO_BATCH_SIZE,
        preprocessing_workers=training_config.PREPROCESSING_WORKERS,
    )
    return text_val, image_val, repair_val


def main() -> None:
    args = parse_args()
    if args.max_batches <= 0:
        raise ValueError("--max-batches must be positive")

    accelerator = Accelerator()
    with accelerator.main_process_first():
        text_val, image_val, repair_val = build_validation_loaders()

    model = load_evaluation_model(args.adapter_path)
    model, text_val, image_val, repair_val = accelerator.prepare(
        model,
        text_val,
        image_val,
        repair_val,
    )

    metrics = validate(
        accelerator,
        model,
        (
            ("text", text_val),
            ("image", image_val),
            ("repair", repair_val),
        ),
        max_batches=args.max_batches,
    )
    accelerator.print(json.dumps(metrics, indent=2, sort_keys=True))
    accelerator.end_training()


if __name__ == "__main__":
    main()
