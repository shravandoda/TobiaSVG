import argparse
import json
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cairosvg import svg2png
from datasets import IterableDataset, IterableDatasetDict, load_dataset


SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    path: str
    config_name: str | None
    svg_column: str
    id_column: str | None
    sample_size: int


DATASETS = {
    "starvector": DatasetSpec(
        key="starvector",
        path="starvector/svg-diagrams",
        config_name=None,
        svg_column="Svg",
        id_column="Filename",
        sample_size=17_000,
    ),
    "vfig_shapes": DatasetSpec(
        key="vfig_shapes",
        path="QijiaHe/VFIG-Data",
        config_name="VFIG-Data-Shapes-and-Arrows",
        svg_column="svg",
        id_column="filename",
        sample_size=6_545,
    ),
    "vfig_complex": DatasetSpec(
        key="vfig_complex",
        path="QijiaHe/VFIG-Data",
        config_name="VFIG-Data-Complex-Diagrams",
        svg_column="svg",
        id_column="filename",
        sample_size=17_000,
    ),
}


def load_env_file(path: Path = Path(".env")) -> None:
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()

        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def repair_missing_svg_closings(svg_text: str) -> str:
    opening_svg_count = len(re.findall(r"<svg(?:\s|>|/)", svg_text))
    closing_svg_count = len(re.findall(r"</svg\s*>", svg_text))
    missing = opening_svg_count - closing_svg_count

    if missing > 0:
        svg_text = svg_text.rstrip() + ("\n</svg>" * missing)

    return svg_text


def maybe_decode_escaped_svg(svg_text: str) -> str:
    if "<svg" in svg_text:
        return svg_text

    try:
        decoded = svg_text.encode("utf-8").decode("unicode_escape")
    except UnicodeDecodeError:
        return svg_text

    return decoded if "<svg" in decoded else svg_text


def clean_svg(svg_text: str) -> str:
    svg_text = maybe_decode_escaped_svg(svg_text)
    svg_text = repair_missing_svg_closings(svg_text)

    root = ET.fromstring(svg_text)

    inner_svg = None
    for child in root:
        if child.tag == f"{{{SVG_NS}}}svg" or child.tag == "svg":
            inner_svg = child
            break

    if inner_svg is None:
        return ET.tostring(root, encoding="unicode")

    for key, value in root.attrib.items():
        if key not in inner_svg.attrib:
            inner_svg.set(key, value)

    return ET.tostring(inner_svg, encoding="unicode")


def load_streaming_splits(spec: DatasetSpec) -> IterableDatasetDict:
    token = os.environ.get("HF_TOKEN") or None
    dataset = load_dataset(
        spec.path,
        spec.config_name,
        streaming=True,
        token=token,
    )

    if isinstance(dataset, IterableDatasetDict):
        return dataset

    return IterableDatasetDict({"train": dataset})


def sample_dataset(
    spec: DatasetSpec,
    *,
    seed: int,
    buffer_size: int,
    sample_size: int | None = None,
) -> IterableDatasetDict:
    sampled = {}
    dataset_dict = load_streaming_splits(spec)

    for split_name, dataset in dataset_dict.items():
        split_sample_size = sample_size or spec.sample_size
        sampled[split_name] = (
            dataset.shuffle(seed=seed, buffer_size=buffer_size).take(split_sample_size)
        )

    return IterableDatasetDict(sampled)


def get_source_id(example: dict[str, Any], spec: DatasetSpec, fallback_id: str) -> str:
    if spec.id_column and spec.id_column in example:
        return str(example[spec.id_column])

    return fallback_id


def rasterize_svg(svg_text: str, image_path: Path) -> None:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    svg2png(bytestring=svg_text.encode("utf-8"), write_to=str(image_path))


def iter_manifest_rows(
    spec: DatasetSpec,
    dataset: IterableDataset,
    *,
    split_name: str,
    image_root: Path,
) -> Any:
    for index, example in enumerate(dataset):
        example_id = f"{spec.key}_{split_name}_{index:06d}"
        image_path = image_root / spec.key / split_name / f"{example_id}.png"

        row = {
            "id": example_id,
            "source": spec.key,
            "dataset": spec.path,
            "dataset_config": spec.config_name,
            "split": split_name,
            "source_id": get_source_id(example, spec, example_id),
            "image_path": str(image_path),
            "svg": None,
            "status": "sampled",
            "error": None,
        }

        try:
            svg_text = clean_svg(str(example[spec.svg_column]))
            rasterize_svg(svg_text, image_path)
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = f"{type(exc).__name__}: {exc}"
        else:
            row["svg"] = svg_text
            row["status"] = "rasterized"

        yield row


def write_manifest(
    selected_datasets: list[str],
    *,
    manifest_path: Path,
    image_root: Path,
    seed: int,
    buffer_size: int,
    sample_size: int | None,
    append: bool,
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"

    with manifest_path.open(mode, encoding="utf-8") as manifest_file:
        for dataset_key in selected_datasets:
            spec = DATASETS[dataset_key]
            sampled = sample_dataset(
                spec,
                seed=seed,
                buffer_size=buffer_size,
                sample_size=sample_size,
            )

            for split_name, dataset in sampled.items():
                for row in iter_manifest_rows(
                    spec,
                    dataset,
                    split_name=split_name,
                    image_root=image_root,
                ):
                    manifest_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                    manifest_file.flush()
                    print(f"{row['status']}: {row['id']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream SVG datasets, sample examples, rasterize PNGs, and write a manifest."
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=sorted(DATASETS),
        default=None,
        help="Dataset key to sample. Pass multiple times for multiple datasets.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--buffer-size", type=int, default=10_000)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Override each dataset's default sample size.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("data/manifests/sampled.jsonl"),
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=Path("data/processed/images"),
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to the manifest instead of overwriting it.",
    )
    return parser.parse_args()


def main() -> None:
    load_env_file()

    args = parse_args()
    selected_datasets = args.dataset or ["starvector"]

    write_manifest(
        selected_datasets,
        manifest_path=args.manifest_path,
        image_root=args.image_root,
        seed=args.seed,
        buffer_size=args.buffer_size,
        sample_size=args.sample_size,
        append=args.append,
    )


if __name__ == "__main__":
    main()
