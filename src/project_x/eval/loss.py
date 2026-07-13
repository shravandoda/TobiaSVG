"""Measure teacher-forced validation loss for a base model or LoRA adapter."""

import argparse
import json
from pathlib import Path

from accelerate import Accelerator
from liger_kernel.transformers import apply_liger_kernel_to_qwen3_5
from peft import PeftModel

from project_x.modeling.loading import get_model
from project_x.training.train import build_training_loaders, validate


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


def main() -> None:
    args = parse_args()
    if args.max_batches <= 0:
        raise ValueError("--max-batches must be positive")

    accelerator = Accelerator()
    with accelerator.main_process_first():
        (
            _,
            _,
            _,
            text_val,
            image_val,
            repair_val,
        ) = build_training_loaders()

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
