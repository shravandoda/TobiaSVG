"""Compare image-to-SVG renders from local and external models."""

import argparse
import csv
import json
import statistics
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

MODEL_LABELS = {
    "base": "Base",
    "fine_tuned": "Fine-tuned",
    "gpt_5_3_codex": "GPT-5.3-Codex",
    "gemini_3_1_flash_lite": "Gemini 3.1 Flash-Lite",
}
CELL_SIZE = (300, 230)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/evaluation/comparison-sets/base-vs-tuned-overlap.json"),
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("artifacts/evaluation/final30-bounded/base/image/test"),
    )
    parser.add_argument(
        "--fine-tuned-dir",
        type=Path,
        default=Path("artifacts/evaluation/repetition-final-p105/tuned/image/test"),
    )
    parser.add_argument(
        "--external-dir",
        type=Path,
        default=Path("artifacts/evaluation/external-image-to-svg"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/evaluation/external-image-to-svg/comparison"),
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def render_cell(image_path: Path, label: str, mse: float | None) -> Image.Image:
    cell = Image.new("RGB", CELL_SIZE, "white")
    draw = ImageDraw.Draw(cell)
    draw.text((10, 8), label, fill="black", font=load_font(17))

    with Image.open(image_path) as source:
        image = ImageOps.contain(source.convert("RGBA"), (280, 175))
    x = (CELL_SIZE[0] - image.width) // 2
    y = 38 + (175 - image.height) // 2
    cell.paste(image, (x, y), image)

    if mse is not None:
        draw.text((10, 207), f"MSE {mse:.4f}", fill="black", font=load_font(14))
    return cell


def save_gallery_row(
    output_path: Path,
    filename: str,
    target_path: Path,
    model_data: dict[str, tuple[Path, float]],
) -> None:
    header_height = 38
    row = Image.new(
        "RGB",
        (CELL_SIZE[0] * (len(model_data) + 1), CELL_SIZE[1] + header_height),
        "white",
    )
    draw = ImageDraw.Draw(row)
    draw.text((10, 8), filename, fill="black", font=load_font(16))

    cells = [render_cell(target_path, "Target", None)]
    cells.extend(
        render_cell(image_path, MODEL_LABELS[model], mse)
        for model, (image_path, mse) in model_data.items()
    )
    for index, cell in enumerate(cells):
        row.paste(cell, (index * CELL_SIZE[0], header_height))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    row.save(output_path)


def main() -> None:
    args = parse_args()
    image_examples = load_json(args.manifest)["tasks"]["image"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gallery_dir = args.output_dir / "gallery"
    rows = []
    gallery_rows = []

    for output_index, item in enumerate(image_examples):
        evaluation_index = item["evaluation_index"]
        local_name = f"{evaluation_index:04d}"
        external_name = f"{output_index:04d}"
        result_paths = {
            "base": args.base_dir / local_name / "result.json",
            "fine_tuned": args.fine_tuned_dir / local_name / "result.json",
            "gpt_5_3_codex": (
                args.external_dir
                / "openai"
                / "image"
                / "test"
                / external_name
                / "result.json"
            ),
            "gemini_3_1_flash_lite": (
                args.external_dir
                / "gemini"
                / "image"
                / "test"
                / external_name
                / "result.json"
            ),
        }
        results = {model: load_json(path) for model, path in result_paths.items()}
        mse_values = {model: result["pixel_mse"] for model, result in results.items()}
        winner = min(mse_values, key=mse_values.get)
        row = {
            "output_index": output_index,
            "evaluation_index": evaluation_index,
            "filename": item["filename"],
            **{f"{model}_mse": mse for model, mse in mse_values.items()},
            "winner": winner,
        }
        rows.append(row)

        model_data = {
            "base": (
                args.base_dir / local_name / "prediction.png",
                mse_values["base"],
            ),
            "fine_tuned": (
                args.fine_tuned_dir / local_name / "prediction.png",
                mse_values["fine_tuned"],
            ),
            "gpt_5_3_codex": (
                args.external_dir
                / "openai"
                / "image"
                / "test"
                / external_name
                / "prediction.png",
                mse_values["gpt_5_3_codex"],
            ),
            "gemini_3_1_flash_lite": (
                args.external_dir
                / "gemini"
                / "image"
                / "test"
                / external_name
                / "prediction.png",
                mse_values["gemini_3_1_flash_lite"],
            ),
        }
        gallery_path = gallery_dir / f"{output_index:04d}.png"
        save_gallery_row(
            gallery_path,
            item["filename"],
            args.base_dir / local_name / "target.png",
            model_data,
        )
        gallery_rows.append(gallery_path)

    model_summary = {}
    for model in MODEL_LABELS:
        values = [row[f"{model}_mse"] for row in rows]
        model_summary[model] = {
            "mean_pixel_mse": sum(values) / len(values),
            "median_pixel_mse": statistics.median(values),
            "wins": sum(row["winner"] == model for row in rows),
        }

    pairwise_wins = {}
    for first_model in MODEL_LABELS:
        for second_model in MODEL_LABELS:
            if first_model >= second_model:
                continue
            pairwise_wins[f"{first_model}_over_{second_model}"] = sum(
                row[f"{first_model}_mse"] < row[f"{second_model}_mse"] for row in rows
            )
    summary = {
        "num_examples": len(rows),
        "models": model_summary,
        "pairwise_wins": pairwise_wins,
    }

    with (args.output_dir / "results.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    markdown = [
        "# Image-to-SVG Model Comparison",
        "",
        "All models are compared on the 13 image examples rendered successfully by "
        "both the base and fine-tuned local models.",
        "",
        "| Model | Mean MSE | Median MSE | Best MSE |",
        "| --- | ---: | ---: | ---: |",
    ]
    for model, label in MODEL_LABELS.items():
        metrics = model_summary[model]
        markdown.append(
            f"| {label} | {metrics['mean_pixel_mse']:.4f} | "
            f"{metrics['median_pixel_mse']:.4f} | {metrics['wins']}/13 |"
        )
    markdown += [
        "",
        "Pixel MSE measures visual reconstruction, not semantic correctness. Sparse "
        "or incomplete drawings can occasionally receive deceptively low scores. "
        "Use `gallery.png` and the individual gallery rows with this table.",
    ]
    (args.output_dir / "RESULTS.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )

    with Image.open(gallery_rows[0]) as first_row:
        width, row_height = first_row.size
    gallery = Image.new("RGB", (width, row_height * len(gallery_rows)), "white")
    for index, path in enumerate(gallery_rows):
        with Image.open(path) as row_image:
            gallery.paste(row_image, (0, index * row_height))
    gallery.save(args.output_dir / "gallery.png")


if __name__ == "__main__":
    main()
