"""Generate and inspect SVG predictions from a trained adapter."""

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import Qwen3VLProcessor

from project_x.constants import DATA_PROCESSING_SEED, MAX_SEQUENCE_LENGTH
from project_x.data.collators import (
    image2svg_sequence_length,
    repair_sequence_length,
    text2svg_sequence_length,
)
from project_x.data.datasets import get_tobias_dataset, get_tobias_repair_dataset
from project_x.eval.metrics import is_valid_svg, normalized_pixel_mse
from project_x.eval.render import extract_svg, render_svg
from project_x.modeling.chat import (
    get_image2svg_prompt,
    get_repair_prompt,
    get_text2svg_prompt,
    serialize_prompt,
)
from project_x.modeling.loading import get_model, get_processor
from project_x.utils.svg import svg2pil

SEQUENCE_LENGTH_FUNCTIONS = {
    "text": text2svg_sequence_length,
    "image": image2svg_sequence_length,
    "repair": repair_sequence_length,
}
TASKS = tuple(SEQUENCE_LENGTH_FUNCTIONS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adapter-path",
        type=Path,
        required=True,
        help="Path to a saved PEFT adapter directory.",
    )
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--num-examples", type=int, default=10)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=MAX_SEQUENCE_LENGTH,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/evaluation"),
    )
    return parser.parse_args()


def load_model_and_processor(adapter_path: Path):
    """Load the base model, processor, and saved PEFT adapter."""
    processor = get_processor()
    base_model = get_model()

    model = PeftModel.from_pretrained(base_model, adapter_path)

    model = model.to("cuda")
    model.eval()

    return model, processor


def load_evaluation_rows(task: str, num_examples: int):
    """Load stable test rows that fit the task's training context limit."""
    if num_examples < 0:
        raise ValueError("num_examples must be non-negative")

    if task in {"text", "image"}:
        dataset = get_tobias_dataset()
    elif task == "repair":
        dataset = get_tobias_repair_dataset()
    else:
        raise ValueError(f"Unknown evaluation task: {task}")

    rows = dataset["test"].shuffle(seed=DATA_PROCESSING_SEED)
    if num_examples == 0:
        return rows.select([])

    sequence_length_fn = SEQUENCE_LENGTH_FUNCTIONS[task]
    valid_indices = []

    for index, row in enumerate(rows):
        if sequence_length_fn(row) <= MAX_SEQUENCE_LENGTH:
            valid_indices.append(index)
        if len(valid_indices) == num_examples:
            break

    return rows.select(valid_indices)


def generate_svg(
    model,
    processor: Qwen3VLProcessor,
    row: dict[str, Any],
    task: str,
    max_new_tokens: int,
) -> str:
    """Generate raw model text from prompt-only inputs for one row."""
    image = None

    #  Build the prompt
    if task == "text":
        prompt = get_text2svg_prompt(row["text"])
    elif task == "image":
        svg = row["svg"]
        image = svg2pil(svg)
        prompt = get_image2svg_prompt(image)
    elif task == "repair":
        image = svg2pil(row["svg"])
        prompt = get_repair_prompt(image, row["corrupted_svg"])
    else:
        raise ValueError(f"Unknown evaluation task: {task}")

    processed_prompt = serialize_prompt(processor, prompt)
    kwargs = {
        "text": [processed_prompt],
        "padding": True,
        "return_tensors": "pt",
    }

    if image is not None:
        kwargs["images"] = [image]

    inputs = processor(**kwargs).to(model.device)

    prompt_length = inputs["input_ids"].shape[1]
    available_tokens = MAX_SEQUENCE_LENGTH - prompt_length
    if available_tokens <= 0:
        raise ValueError("Prompt exceeds MAX_SEQUENCE_LENGTH")

    generation_length = min(max_new_tokens, available_tokens)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=generation_length,
            do_sample=False,
        )

    generated_ids = output[:, prompt_length:]
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


def save_example(
    output_dir: Path,
    index: int,
    row: dict[str, Any],
    generated_text: str,
) -> dict[str, Any]:
    """Save raw output, extracted SVGs, renders, and basic status metadata."""
    example_dir = output_dir / f"{index:04d}"
    example_dir.mkdir(parents=True, exist_ok=True)

    target_svg = row["svg"]
    target_path = example_dir / "target.png"
    prediction_path = example_dir / "prediction.png"
    (example_dir / "target.svg").write_text(target_svg, encoding="utf-8")
    (example_dir / "prediction.txt").write_text(generated_text, encoding="utf-8")

    result: dict[str, Any] = {
        "index": index,
        "filename": row.get("filename"),
        "prediction_has_svg": False,
        "prediction_is_valid_svg": False,
        "prediction_rendered": False,
    }

    try:
        predicted_svg = extract_svg(generated_text)
        result["prediction_has_svg"] = True
        result["prediction_is_valid_svg"] = is_valid_svg(predicted_svg)
        (example_dir / "prediction.svg").write_text(
            predicted_svg,
            encoding="utf-8",
        )
        render_svg(predicted_svg, prediction_path)
        result["prediction_rendered"] = True
    except Exception as error:
        result["prediction_error"] = str(error)

    try:
        render_svg(target_svg, target_path)
        result["target_rendered"] = True
    except Exception as error:
        result["target_rendered"] = False
        result["target_error"] = str(error)

    if result["prediction_rendered"] and result["target_rendered"]:
        result["pixel_mse"] = normalized_pixel_mse(target_path, prediction_path)

    (example_dir / "result.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    return result


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate validity, rendering, and pixel-error metrics."""
    total = len(results)
    mse_values = [result["pixel_mse"] for result in results if "pixel_mse" in result]

    def rate(key: str) -> float:
        return sum(bool(result[key]) for result in results) / total if total else 0.0

    return {
        "num_examples": total,
        "svg_extraction_rate": rate("prediction_has_svg"),
        "valid_svg_rate": rate("prediction_is_valid_svg"),
        "render_success_rate": rate("prediction_rendered"),
        "pixel_mse_examples": len(mse_values),
        "mean_pixel_mse": sum(mse_values) / len(mse_values) if mse_values else None,
    }


def main() -> None:
    args = parse_args()
    rows = load_evaluation_rows(args.task, args.num_examples)
    model, processor = load_model_and_processor(args.adapter_path)
    run_dir = args.output_dir / args.task / "test"

    results = []
    for index, row in enumerate(rows):
        generated_text = generate_svg(
            model,
            processor,
            row,
            args.task,
            args.max_new_tokens,
        )
        results.append(save_example(run_dir, index, row, generated_text))

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "results.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps(summarize_results(results), indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
