"""Evaluate external multimodal models on the image-to-SVG intersection set."""

import argparse
import asyncio
import base64
import json
import os
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types as genai_types
from openai import AsyncOpenAI
from PIL import Image

from project_x.eval.evaluate import save_example, summarize_results

OPENAI_MODEL = "gpt-5.3-codex"
GEMINI_MODEL = "gemini-3.1-flash-lite"
IMAGE_TO_SVG_PROMPT = (
    "Recreate the provided image as a complete SVG document. Preserve the visible "
    "geometry, layout, text, colors, fills, strokes, and relative positioning as "
    "faithfully as possible. Use only vector SVG elements: do not embed raster "
    "images, data URLs, scripts, or external resources. Return only SVG markup "
    "beginning with <svg and ending with </svg>."
)


@dataclass(frozen=True)
class BenchmarkExample:
    output_index: int
    evaluation_index: int
    filename: str
    image_path: Path
    svg_path: Path


@dataclass(frozen=True)
class GeneratedOutput:
    provider: str
    model: str
    example: BenchmarkExample
    text: str | None
    latency_seconds: float
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/evaluation/comparison-sets/base-vs-tuned-overlap.json"),
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("artifacts/evaluation/final30-bounded/base/image/test"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/evaluation/external-image-to-svg"),
    )
    parser.add_argument("--openai-model", default=OPENAI_MODEL)
    parser.add_argument("--gemini-model", default=GEMINI_MODEL)
    parser.add_argument("--openai-concurrency", type=int, default=4)
    parser.add_argument("--gemini-concurrency", type=int, default=4)
    parser.add_argument("--max-output-tokens", type=int, default=32_768)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate examples whose result files already exist.",
    )
    return parser.parse_args()


def load_examples(manifest_path: Path, source_dir: Path) -> list[BenchmarkExample]:
    """Load the image-task intersection and its existing target artifacts."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    examples = []

    for output_index, item in enumerate(manifest["tasks"]["image"]):
        evaluation_index = item["evaluation_index"]
        example_dir = source_dir / f"{evaluation_index:04d}"
        image_path = example_dir / "target.png"
        svg_path = example_dir / "target.svg"

        if not image_path.is_file() or not svg_path.is_file():
            raise FileNotFoundError(
                f"Missing target artifacts for evaluation index {evaluation_index}."
            )

        examples.append(
            BenchmarkExample(
                output_index=output_index,
                evaluation_index=evaluation_index,
                filename=item["filename"],
                image_path=image_path,
                svg_path=svg_path,
            )
        )

    return examples


def encode_data_url(image_path: Path) -> str:
    image_data = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{image_data}"


async def request_openai_svg(
    client: AsyncOpenAI,
    example: BenchmarkExample,
    model: str,
    max_output_tokens: int,
) -> str:
    response = await client.responses.create(
        model=model,
        reasoning={"effort": "high"},
        max_output_tokens=max_output_tokens,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": IMAGE_TO_SVG_PROMPT},
                    {
                        "type": "input_image",
                        "image_url": encode_data_url(example.image_path),
                        "detail": "high",
                    },
                ],
            }
        ],
    )
    if not response.output_text:
        raise RuntimeError("OpenAI returned no text output.")
    return response.output_text


async def request_gemini_svg(
    client: genai.Client,
    example: BenchmarkExample,
    model: str,
    max_output_tokens: int,
) -> str:
    with Image.open(example.image_path) as image:
        input_image = image.copy()

    response = await client.aio.models.generate_content(
        model=model,
        contents=[IMAGE_TO_SVG_PROMPT, input_image],
        config=genai_types.GenerateContentConfig(
            temperature=0,
            max_output_tokens=max_output_tokens,
        ),
    )
    if not response.text:
        raise RuntimeError("Gemini returned no text output.")
    return response.text


async def call_model(
    provider: str,
    model: str,
    example: BenchmarkExample,
    semaphore: asyncio.Semaphore,
    request: Callable[[BenchmarkExample], Awaitable[str]],
) -> GeneratedOutput:
    try:
        async with semaphore:
            start_time = time.monotonic()
            text = await request(example)
        return GeneratedOutput(
            provider=provider,
            model=model,
            example=example,
            text=text,
            latency_seconds=time.monotonic() - start_time,
        )
    except Exception as error:
        return GeneratedOutput(
            provider=provider,
            model=model,
            example=example,
            text=None,
            latency_seconds=time.monotonic() - start_time,
            error=f"{type(error).__name__}: {error}",
        )


def result_path(output_dir: Path, provider: str, output_index: int) -> Path:
    return (
        output_dir / provider / "image" / "test" / f"{output_index:04d}" / "result.json"
    )


def load_completed_results(
    output_dir: Path,
    provider: str,
    examples: list[BenchmarkExample],
) -> dict[int, dict[str, Any]]:
    completed = {}
    for example in examples:
        path = result_path(output_dir, provider, example.output_index)
        if path.is_file():
            completed[example.output_index] = json.loads(
                path.read_text(encoding="utf-8")
            )
    return completed


def save_generated_output(output_dir: Path, output: GeneratedOutput) -> dict[str, Any]:
    example = output.example
    run_dir = output_dir / output.provider / "image" / "test"
    row = {
        "filename": example.filename,
        "svg": example.svg_path.read_text(encoding="utf-8"),
    }

    result = save_example(run_dir, example.output_index, row, output.text or "")
    if output.error is not None:
        result["request_error"] = output.error

    result.update(
        {
            "provider": output.provider,
            "model": output.model,
            "source_evaluation_index": example.evaluation_index,
            "latency_seconds": output.latency_seconds,
        }
    )
    path = result_path(output_dir, output.provider, example.output_index)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def write_provider_summary(
    output_dir: Path,
    provider: str,
    model: str,
    results: list[dict[str, Any]],
) -> None:
    run_dir = output_dir / provider / "image" / "test"
    summary = summarize_results(results)
    summary.update(
        {
            "provider": provider,
            "model": model,
            "mean_latency_seconds": sum(result["latency_seconds"] for result in results)
            / len(results),
            "coverage_penalized_mse": sum(
                result.get("pixel_mse", 1.0) for result in results
            )
            / len(results),
        }
    )
    (run_dir / "results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


async def run_benchmark(args: argparse.Namespace) -> None:
    if args.openai_concurrency <= 0 or args.gemini_concurrency <= 0:
        raise ValueError("Provider concurrency must be positive.")
    if args.max_output_tokens <= 0:
        raise ValueError("--max-output-tokens must be positive.")

    openai_api_key = os.environ.get("OPENAI_API_KEY")
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if not openai_api_key or not gemini_api_key:
        raise RuntimeError("OPENAI_API_KEY and GEMINI_API_KEY are required.")

    examples = load_examples(args.manifest, args.source_dir)
    openai_client = AsyncOpenAI(api_key=openai_api_key)
    gemini_client = genai.Client(api_key=gemini_api_key)
    provider_configs = {
        "openai": (
            args.openai_model,
            args.openai_concurrency,
            lambda example: request_openai_svg(
                openai_client,
                example,
                args.openai_model,
                args.max_output_tokens,
            ),
        ),
        "gemini": (
            args.gemini_model,
            args.gemini_concurrency,
            lambda example: request_gemini_svg(
                gemini_client,
                example,
                args.gemini_model,
                args.max_output_tokens,
            ),
        ),
    }

    completed_by_provider = {
        provider: {}
        if args.overwrite
        else load_completed_results(args.output_dir, provider, examples)
        for provider in provider_configs
    }
    calls = []
    for provider, (model, concurrency, request) in provider_configs.items():
        semaphore = asyncio.Semaphore(concurrency)
        for example in examples:
            if example.output_index in completed_by_provider[provider]:
                continue
            calls.append(call_model(provider, model, example, semaphore, request))

    # Results are saved as calls finish. Cairo rendering stays sequential because
    # concurrent rendering is unstable, while API requests remain asynchronous.
    for completed_call in asyncio.as_completed(calls):
        output = await completed_call
        result = save_generated_output(args.output_dir, output)
        completed_by_provider[output.provider][output.example.output_index] = result
        status = "rendered" if result["prediction_rendered"] else "failed"
        print(
            f"{output.provider}: {output.example.output_index + 1}/{len(examples)} "
            f"{status} ({output.latency_seconds:.1f}s)",
            flush=True,
        )

    for provider, (model, _, _) in provider_configs.items():
        results = [
            completed_by_provider[provider][example.output_index]
            for example in examples
        ]
        write_provider_summary(args.output_dir, provider, model, results)


def main() -> None:
    asyncio.run(run_benchmark(parse_args()))


if __name__ == "__main__":
    main()
