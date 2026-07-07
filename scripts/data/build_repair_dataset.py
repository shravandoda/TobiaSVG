import argparse
import random
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Callable
from pathlib import Path
from typing import Any

from datasets import Dataset, Features, Value, load_from_disk
from PIL import Image

from scripts.data.process_svg import SVG_NS, render_svg_to_png
from scripts.data.sample_dataset import DATASETS, DatasetSpec
from scripts.data.utils import OUTPUT_ROOT, build_split_output_path

REPAIR_OUTPUT_ROOT = Path("data/processed/repair_datasets")
INSPECT_ROOT = Path("data/processed/inspect/repair_corruptions")
ET.register_namespace("", SVG_NS)

GEOMETRIC_TAGS = {
    "path",
    "polygon",
    "polyline",
    "rect",
    "circle",
    "ellipse",
    "line",
    "text",
}
STYLE_ATTRIBUTES = ["fill", "stroke", "opacity", "stroke-width"]
GEOMETRY_ATTRIBUTES = [
    "x",
    "y",
    "cx",
    "cy",
    "r",
    "rx",
    "ry",
    "width",
    "height",
    "x1",
    "y1",
    "x2",
    "y2",
]
COLORS = ["#000000", "#ffffff", "#ff0000", "#00aa55", "#0066ff", "#ffaa00"]
NON_SCENE_CONTAINER_TAGS = {
    "clipPath",
    "defs",
    "filter",
    "linearGradient",
    "marker",
    "mask",
    "metadata",
    "pattern",
    "radialGradient",
    "style",
    "symbol",
}

REPAIR_FEATURES = Features(
    {
        "filename": Value("string"),
        "svg": Value("string"),
        "corrupted_svg": Value("string"),
        "corruption_type": Value("string"),
        "corruption_count": Value("int64"),
        "corruption_seed": Value("int64"),
        "corruption_level": Value("string"),
        "render_mse": Value("float64"),
        "changed_pixel_ratio": Value("float64"),
        "text": Value("string"),
        "caption_style": Value("string"),
        "dataset": Value("string"),
    }
)


class CorruptionError(ValueError):
    pass


def local_tag_name(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1].lower()


def qualify_tag(root: ET.Element, tag_name: str) -> str:
    if root.tag.startswith("{"):
        namespace = root.tag.split("}", 1)[0][1:]
        return f"{{{namespace}}}{tag_name}"

    return tag_name


def parent_map(root: ET.Element) -> dict[ET.Element, ET.Element]:
    return {child: parent for parent in root.iter() for child in parent}


def has_ancestor_with_tag(
    element: ET.Element,
    parents: dict[ET.Element, ET.Element],
    tag_names: set[str],
) -> bool:
    parent = parents.get(element)
    while parent is not None:
        if local_tag_name(parent) in tag_names:
            return True
        parent = parents.get(parent)

    return False


def is_non_scene_element(
    element: ET.Element,
    parents: dict[ET.Element, ET.Element],
) -> bool:
    return local_tag_name(element) in NON_SCENE_CONTAINER_TAGS or has_ancestor_with_tag(
        element,
        parents,
        NON_SCENE_CONTAINER_TAGS,
    )


def element_candidates(root: ET.Element) -> list[ET.Element]:
    return [
        element
        for element in root.iter()
        if element is not root and local_tag_name(element) in GEOMETRIC_TAGS
    ]


def parse_svg(svg: str) -> ET.Element:
    try:
        return ET.fromstring(svg)
    except ET.ParseError as exc:
        raise CorruptionError(f"clean SVG parse failed: {exc}") from exc


def serialize_svg(root: ET.Element) -> str:
    return ET.tostring(root, encoding="unicode")


def ensure_changed(clean_svg: str, corrupted_svg: str) -> str:
    if corrupted_svg == clean_svg:
        raise CorruptionError("corruption did not change SVG")

    return corrupted_svg


def corrupt_missing_objects(svg: str, rng: random.Random, level: str) -> str:
    root = parse_svg(svg)
    parents = parent_map(root)
    candidates = [
        element
        for element in element_candidates(root)
        if element in parents and not is_non_scene_element(element, parents)
    ]
    if not candidates:
        raise CorruptionError("no removable SVG elements found")

    remove_count = {"easy": 1, "medium": 2, "hard": 3}[level]
    for element in rng.sample(candidates, k=min(remove_count, len(candidates))):
        parent = parents.get(element)
        if parent is not None:
            parent.remove(element)

    return ensure_changed(svg, serialize_svg(root))


def corrupt_wrong_z_order(svg: str, rng: random.Random, level: str) -> str:
    root = parse_svg(svg)
    update_count = {"easy": 1, "medium": 2, "hard": 3}[level]

    for _ in range(update_count):
        parents_by_child = parent_map(root)
        parents = [
            element
            for element in root.iter()
            if not is_non_scene_element(element, parents_by_child)
            and len(scene_reorderable_child_indices(element, parents_by_child)) >= 2
        ]
        if not parents:
            raise CorruptionError("no sibling elements found for z-order corruption")

        parent = rng.choice(parents)
        children = list(parent)
        order = scene_reorderable_child_indices(parent, parents_by_child)

        for _ in range(10):
            shuffled = order[:]
            rng.shuffle(shuffled)
            if shuffled != order:
                break
        else:
            raise CorruptionError("could not shuffle z-order")

        reordered_children = children[:]
        for target_index, source_index in zip(order, shuffled, strict=True):
            reordered_children[target_index] = children[source_index]
        parent[:] = reordered_children

    return ensure_changed(svg, serialize_svg(root))


def scene_reorderable_child_indices(
    element: ET.Element,
    parents: dict[ET.Element, ET.Element],
) -> list[int]:
    return [
        index
        for index, child in enumerate(list(element))
        if local_tag_name(child) in GEOMETRIC_TAGS | {"g"}
        and not is_non_scene_element(child, parents)
    ]


def corrupt_style(svg: str, rng: random.Random, level: str) -> str:
    root = parse_svg(svg)
    parents = parent_map(root)
    candidates = [
        element
        for element in element_candidates(root)
        if not is_non_scene_element(element, parents)
        and local_tag_name(element) != "text"
        and (
            has_direct_paint(element)
            or has_direct_stroke_width(element)
            or has_direct_opacity(element)
        )
    ]
    if not candidates:
        raise CorruptionError("no style candidates found")

    update_count = {"easy": 1, "medium": 2, "hard": 3}[level]
    for element in rng.sample(candidates, k=min(update_count, len(candidates))):
        attribute = rng.choice(visible_style_attributes(element))
        if attribute == "opacity":
            element.set(attribute, f"{rng.uniform(0.15, 0.85):.2f}")
        elif attribute == "stroke-width":
            element.set(attribute, str(rng.choice([1, 2, 4, 8])))
        else:
            element.set(attribute, choose_different_color(element.attrib[attribute], rng))

    return ensure_changed(svg, serialize_svg(root))


def has_direct_paint(element: ET.Element) -> bool:
    return any(
        attribute in element.attrib
        and element.attrib[attribute].strip()
        and not element.attrib[attribute].startswith("url(")
        and element.attrib[attribute] != "none"
        for attribute in ["fill", "stroke"]
    )


def has_direct_stroke_width(element: ET.Element) -> bool:
    return "stroke-width" in element.attrib and element.attrib.get("stroke") not in {
        None,
        "none",
    }


def has_direct_opacity(element: ET.Element) -> bool:
    return "opacity" in element.attrib


def visible_style_attributes(element: ET.Element) -> list[str]:
    attributes: list[str] = []

    for attribute in ["fill", "stroke"]:
        value = element.attrib.get(attribute)
        if value and value != "none" and not value.startswith("url("):
            attributes.append(attribute)

    if has_direct_stroke_width(element):
        attributes.append("stroke-width")

    if has_direct_opacity(element):
        attributes.append("opacity")

    return attributes


def choose_different_color(current_color: str, rng: random.Random) -> str:
    candidates = [color for color in COLORS if color.lower() != current_color.lower()]
    return rng.choice(candidates or COLORS)


def corrupt_text(svg: str, rng: random.Random, level: str) -> str:
    root = parse_svg(svg)
    parents = parent_map(root)
    text_path_tag = qualify_tag(root, "path")
    candidates = [
        element
        for element in root.iter()
        if local_tag_name(element) == "text" and element.text and element.text.strip()
        and not is_non_scene_element(element, parents)
    ]
    if not candidates:
        raise CorruptionError("no text elements found")

    update_count = {"easy": 1, "medium": 2, "hard": 3}[level]
    for element in rng.sample(candidates, k=min(update_count, len(candidates))):
        text = element.text or ""
        if rng.random() < 0.5:
            element.text = ""
            continue

        x = first_numeric_value(element.attrib.get("x", "0"), default=0.0)
        y = first_numeric_value(element.attrib.get("y", "0"), default=0.0)
        width = max(8.0, min(120.0, len(text) * 7.0))
        stroke = element.attrib.get("fill") or element.attrib.get("stroke") or "#000000"

        element.clear()
        element.tag = text_path_tag
        element.set("d", f"M {x:g} {y:g} h {width:g}")
        element.set("fill", "none")
        element.set("stroke", stroke)
        element.set("stroke-width", "1")

    return ensure_changed(svg, serialize_svg(root))


def first_numeric_value(value: str, default: float) -> float:
    match = re.search(r"-?\d+(?:\.\d+)?", value)
    if match is None:
        return default

    return float(match.group(0))


def perturb_number(value: str, rng: random.Random, level: str) -> str:
    try:
        number = float(value)
    except ValueError as exc:
        raise CorruptionError(f"not a plain numeric value: {value}") from exc

    scale = {"easy": 0.05, "medium": 0.12, "hard": 0.25}[level]
    delta = max(abs(number), 10.0) * rng.uniform(-scale, scale)
    perturbed = number + delta

    if "." not in value and value.isdigit():
        return str(max(0, round(perturbed)))

    return f"{perturbed:.2f}".rstrip("0").rstrip(".")


def perturb_path_data(path_data: str, rng: random.Random, level: str) -> str:
    numbers = list(re.finditer(r"-?\d+(?:\.\d+)?", path_data))
    if not numbers:
        raise CorruptionError("path has no numeric coordinates")

    update_count = min({"easy": 1, "medium": 3, "hard": 6}[level], len(numbers))
    selected = set(rng.sample(range(len(numbers)), k=update_count))
    parts: list[str] = []
    last_end = 0

    for index, match in enumerate(numbers):
        parts.append(path_data[last_end : match.start()])
        value = match.group(0)
        parts.append(perturb_number(value, rng, level) if index in selected else value)
        last_end = match.end()

    parts.append(path_data[last_end:])
    return "".join(parts)


def corrupt_geometry(svg: str, rng: random.Random, level: str) -> str:
    root = parse_svg(svg)
    parents = parent_map(root)
    candidates = [
        element
        for element in element_candidates(root)
        if not is_non_scene_element(element, parents)
        and (
            any(attribute in element.attrib for attribute in GEOMETRY_ATTRIBUTES)
            or "d" in element.attrib
        )
    ]
    if not candidates:
        raise CorruptionError("no geometry candidates found")

    update_count = {"easy": 1, "medium": 2, "hard": 3}[level]
    changed = False

    for element in rng.sample(candidates, k=min(update_count, len(candidates))):
        numeric_attributes = [
            attribute
            for attribute in GEOMETRY_ATTRIBUTES
            if attribute in element.attrib
        ]

        if "d" in element.attrib and (not numeric_attributes or rng.random() < 0.5):
            element.set("d", perturb_path_data(element.attrib["d"], rng, level))
            changed = True
        elif numeric_attributes:
            attribute = rng.choice(numeric_attributes)
            element.set(
                attribute,
                perturb_number(element.attrib[attribute], rng, level),
            )
            changed = True

    if not changed:
        raise CorruptionError("selected elements have no perturbable geometry")

    return ensure_changed(svg, serialize_svg(root))


def corrupt_primitive_degradation(svg: str, rng: random.Random, level: str) -> str:
    root = parse_svg(svg)
    parents = parent_map(root)
    candidates = [
        element
        for element in element_candidates(root)
        if local_tag_name(element) in {"path", "polygon", "polyline"}
        and not is_non_scene_element(element, parents)
    ]
    if not candidates:
        raise CorruptionError("no complex primitives found")

    update_count = {"easy": 1, "medium": 2, "hard": 3}[level]
    polyline_tag = qualify_tag(root, "polyline")

    for element in rng.sample(candidates, k=min(update_count, len(candidates))):
        original_tag = local_tag_name(element)
        element.attrib.clear()

        if original_tag == "path":
            element.set("d", "M 0 0 L 10 10")
        else:
            element.tag = polyline_tag
            element.set("points", "0,0 10,10 20,0")

        element.set("fill", "none")
        element.set("stroke", "#000000")

    return ensure_changed(svg, serialize_svg(root))


def corrupt_group_flattening(svg: str, rng: random.Random, level: str) -> str:
    root = parse_svg(svg)
    update_count = {"easy": 1, "medium": 2, "hard": 3}[level]

    for _ in range(update_count):
        parents = parent_map(root)
        groups = [
            element
            for element in root.iter()
            if local_tag_name(element) == "g"
            and element in parents
            and len(list(element))
            and not is_non_scene_element(element, parents)
        ]
        if not groups:
            raise CorruptionError("no groups found")

        group = rng.choice(groups)
        parent = parents[group]
        children = list(parent)
        group_index = children.index(group)
        parent.remove(group)

        for offset, child in enumerate(list(group)):
            parent.insert(group_index + offset, child)

    return ensure_changed(svg, serialize_svg(root))


def corrupt_truncation(svg: str, rng: random.Random, level: str) -> str:
    max_fraction = {"easy": 0.02, "medium": 0.05, "hard": 0.10}[level]
    min_chars = min(8, max(1, len(svg) - 1))
    max_chars = max(min_chars, int(len(svg) * max_fraction))
    remove_count = rng.randint(min_chars, max_chars)
    return ensure_changed(svg, svg[:-remove_count].rstrip())


CORRUPTIONS: dict[str, Callable[[str, random.Random, str], str]] = {
    "missing_objects": corrupt_missing_objects,
    "wrong_z_order": corrupt_wrong_z_order,
    "primitive_degradation": corrupt_primitive_degradation,
    "text_corruption": corrupt_text,
    "style_corruption": corrupt_style,
    "geometry_perturbation": corrupt_geometry,
    "group_flattening": corrupt_group_flattening,
    "truncation": corrupt_truncation,
}


def resolve_split_path(
    spec: DatasetSpec,
    dataset_key: str,
    split_name: str,
    input_root: Path,
) -> Path:
    canonical_path = build_split_output_path(
        spec=spec,
        split_name=split_name,
        output_root=input_root,
    )
    if canonical_path.exists():
        return canonical_path

    legacy_path = input_root / dataset_key / split_name
    if legacy_path.exists():
        return legacy_path

    raise FileNotFoundError(
        f"Could not find processed split at {canonical_path} or {legacy_path}"
    )


def choose_corruptions(
    corruption_types: list[str],
    per_row: int,
    rng: random.Random,
) -> list[str]:
    if per_row >= len(corruption_types):
        selected = corruption_types[:]
        rng.shuffle(selected)
        return selected

    return rng.sample(corruption_types, k=per_row)


def choose_corruption_count(rng: random.Random) -> int:
    roll = rng.random()
    if roll < 0.50:
        return 1
    if roll < 0.75:
        return 2
    return 3


def order_corruptions_for_application(corruption_types: list[str]) -> list[str]:
    return sorted(corruption_types, key=lambda corruption_type: corruption_type == "truncation")


def apply_corruption_stack(
    svg: str,
    *,
    corruption_types: list[str],
    corruption_level: str,
    corruption_seed: int,
) -> tuple[str, list[str]]:
    corrupted_svg = svg
    applied_corruptions: list[str] = []

    for offset, corruption_type in enumerate(
        order_corruptions_for_application(corruption_types)
    ):
        corruption_rng = random.Random(corruption_seed * 100 + offset)
        corrupted_svg = CORRUPTIONS[corruption_type](
            corrupted_svg,
            corruption_rng,
            corruption_level,
        )
        applied_corruptions.append(corruption_type)

    return corrupted_svg, applied_corruptions


def render_difference_metrics(
    clean_svg: str,
    corrupted_svg: str,
    temp_dir: Path,
) -> tuple[float, float]:
    clean_path = temp_dir / "clean.png"
    corrupted_path = temp_dir / "corrupted.png"

    render_svg_to_png(clean_svg, clean_path)
    render_svg_to_png(corrupted_svg, corrupted_path)

    with Image.open(clean_path) as clean_image:
        clean_rgba = clean_image.convert("RGBA")
        clean_pixels = image_pixels(clean_rgba)
        size = clean_image.size

    with Image.open(corrupted_path) as corrupted_image:
        corrupted = corrupted_image.convert("RGBA")
        if corrupted.size != size:
            corrupted = corrupted.resize(size)
        corrupted_pixels = image_pixels(corrupted)

    total_squared_error = 0.0
    changed_pixels = 0
    total_channels = len(clean_pixels) * 4

    for clean_pixel, corrupted_pixel in zip(clean_pixels, corrupted_pixels):
        pixel_changed = False
        for clean_channel, corrupted_channel in zip(clean_pixel, corrupted_pixel):
            difference = abs(clean_channel - corrupted_channel) / 255.0
            total_squared_error += difference * difference
            if difference > 0.03:
                pixel_changed = True

        if pixel_changed:
            changed_pixels += 1

    mse = total_squared_error / total_channels
    changed_pixel_ratio = changed_pixels / len(clean_pixels)
    return mse, changed_pixel_ratio


def image_pixels(image: Image.Image) -> list[tuple[int, int, int, int]]:
    if hasattr(image, "get_flattened_data"):
        return list(image.get_flattened_data())

    return list(image.getdata())


def validate_render_difference(
    clean_svg: str,
    corrupted_svg: str,
    *,
    min_render_mse: float,
    min_changed_pixel_ratio: float,
    temp_dir: Path,
) -> tuple[float, float]:
    if min_render_mse <= 0 and min_changed_pixel_ratio <= 0:
        return 0.0, 0.0

    try:
        render_mse, changed_pixel_ratio = render_difference_metrics(
            clean_svg,
            corrupted_svg,
            temp_dir,
        )
    except Exception as exc:
        raise CorruptionError(f"render diff failed: {exc}") from exc

    if render_mse < min_render_mse:
        raise CorruptionError(
            f"render MSE {render_mse:.6f} < {min_render_mse:.6f}"
        )

    if changed_pixel_ratio < min_changed_pixel_ratio:
        raise CorruptionError(
            "changed pixel ratio "
            f"{changed_pixel_ratio:.6f} < {min_changed_pixel_ratio:.6f}"
        )

    return render_mse, changed_pixel_ratio


def build_repair_rows(
    dataset: Dataset,
    *,
    dataset_key: str,
    split_name: str,
    corruption_types: list[str],
    corruption_level: str,
    per_row: int,
    seed: int,
    start_index: int,
    sample_size: int | None,
    min_render_mse: float,
    min_changed_pixel_ratio: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    start_index = max(start_index, 0)
    stop_index = len(dataset)
    if sample_size is not None:
        stop_index = min(start_index + sample_size, len(dataset))

    with tempfile.TemporaryDirectory(prefix="project_x_repair_diff_") as temp_root:
        temp_dir = Path(temp_root)
        for row_index in range(start_index, stop_index):
            row = dataset[row_index]
            row_seed = seed + row_index
            for repair_offset in range(per_row):
                base_seed = row_seed * 100 + repair_offset
                last_selected_corruptions: list[str] = []
                last_error: CorruptionError | None = None

                for retry_offset in range(8):
                    corruption_seed = base_seed * 10 + retry_offset
                    corruption_rng = random.Random(corruption_seed)
                    corruption_count = min(
                        choose_corruption_count(corruption_rng),
                        len(corruption_types),
                    )
                    selected_corruptions = choose_corruptions(
                        corruption_types,
                        corruption_count,
                        corruption_rng,
                    )
                    last_selected_corruptions = selected_corruptions

                    try:
                        corrupted_svg, applied_corruptions = apply_corruption_stack(
                            row["svg"],
                            corruption_types=selected_corruptions,
                            corruption_level=corruption_level,
                            corruption_seed=corruption_seed,
                        )
                        render_mse, changed_pixel_ratio = validate_render_difference(
                            row["svg"],
                            corrupted_svg,
                            min_render_mse=min_render_mse,
                            min_changed_pixel_ratio=min_changed_pixel_ratio,
                            temp_dir=temp_dir,
                        )
                        break
                    except CorruptionError as exc:
                        last_error = exc
                else:
                    print(
                        "skipped: "
                        f"{dataset_key}/{split_name}/{row_index} "
                        f"{'+'.join(last_selected_corruptions)} ({last_error})"
                    )
                    continue

                rows.append(
                    {
                        "filename": str(row["filename"]),
                        "svg": str(row["svg"]),
                        "corrupted_svg": corrupted_svg,
                        "corruption_type": "+".join(applied_corruptions),
                        "corruption_count": len(applied_corruptions),
                        "corruption_seed": corruption_seed,
                        "corruption_level": corruption_level,
                        "render_mse": render_mse,
                        "changed_pixel_ratio": changed_pixel_ratio,
                        "text": str(row.get("text") or ""),
                        "caption_style": str(row.get("caption_style") or ""),
                        "dataset": str(row.get("dataset") or dataset_key),
                    }
                )

    return rows


def save_repair_dataset(rows: list[dict[str, Any]], output_path: Path) -> Dataset:
    dataset = Dataset.from_list(rows, features=REPAIR_FEATURES)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f".{output_path.name}.tmp")
    if temp_path.exists():
        shutil.rmtree(temp_path)
    dataset.save_to_disk(str(temp_path))
    if output_path.exists():
        shutil.rmtree(output_path)
    temp_path.rename(output_path)
    return dataset


def write_inspection_files(
    dataset: Dataset,
    output_dir: Path,
    inspect_size: int,
) -> None:
    if inspect_size <= 0:
        return

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for index in range(min(inspect_size, len(dataset))):
        row = dataset[index]
        prefix = (
            f"{index:03d}_{row['dataset']}_{row['corruption_type']}_"
            f"{Path(row['filename']).stem}"
        )
        clean_svg_path = output_dir / f"{prefix}.clean.svg"
        corrupted_svg_path = output_dir / f"{prefix}.corrupted.svg"
        clean_png_path = output_dir / f"{prefix}.clean.png"
        corrupted_png_path = output_dir / f"{prefix}.corrupted.png"

        clean_svg_path.write_text(row["svg"])
        corrupted_svg_path.write_text(row["corrupted_svg"])

        for svg_text, png_path in [
            (row["svg"], clean_png_path),
            (row["corrupted_svg"], corrupted_png_path),
        ]:
            try:
                render_svg_to_png(svg_text, png_path)
            except Exception as exc:
                png_path.with_suffix(".render_error.txt").write_text(str(exc))


def process_split(
    *,
    dataset_key: str,
    spec: DatasetSpec,
    split_name: str,
    input_root: Path,
    output_root: Path,
    corruption_types: list[str],
    corruption_level: str,
    per_row: int,
    seed: int,
    start_index: int,
    sample_size: int | None,
    min_render_mse: float,
    min_changed_pixel_ratio: float,
) -> Dataset:
    input_path = resolve_split_path(
        spec=spec,
        dataset_key=dataset_key,
        split_name=split_name,
        input_root=input_root,
    )
    output_path = output_root / spec.key / split_name
    dataset = load_from_disk(str(input_path))
    if not isinstance(dataset, Dataset):
        raise TypeError(f"Expected a Dataset at {input_path}")

    rows = build_repair_rows(
        dataset,
        dataset_key=spec.key,
        split_name=split_name,
        corruption_types=corruption_types,
        corruption_level=corruption_level,
        per_row=per_row,
        seed=seed,
        start_index=start_index,
        sample_size=sample_size,
        min_render_mse=min_render_mse,
        min_changed_pixel_ratio=min_changed_pixel_ratio,
    )
    repair_dataset = save_repair_dataset(rows, output_path)
    print(f"saved: {output_path} rows={len(repair_dataset)}")
    return repair_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build image + corrupted SVG repair datasets."
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=sorted(DATASETS),
        default=None,
        help="Dataset key to process. Pass multiple times, or omit for all.",
    )
    parser.add_argument(
        "--split",
        action="append",
        default=None,
        help=(
            "Split name to process. Pass multiple times, or omit for all "
            "configured splits."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--per-row", type=int, default=1)
    parser.add_argument(
        "--corruption-level",
        choices=["easy", "medium", "hard"],
        default="easy",
    )
    parser.add_argument(
        "--corruption-type",
        action="append",
        choices=sorted(CORRUPTIONS),
        default=None,
        help="Corruption type to use. Pass multiple times, or omit for all.",
    )
    parser.add_argument("--input-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=REPAIR_OUTPUT_ROOT)
    parser.add_argument(
        "--min-render-mse",
        type=float,
        default=0.002,
        help=(
            "Keep only corruptions with normalized render MSE at least this value. "
            "Use 0 to disable this part of the visual-difference filter."
        ),
    )
    parser.add_argument(
        "--min-changed-pixel-ratio",
        type=float,
        default=0.01,
        help=(
            "Keep only corruptions where at least this fraction of rendered pixels "
            "changed. Use 0 to disable this part of the visual-difference filter."
        ),
    )
    parser.add_argument(
        "--inspect-size",
        type=int,
        default=0,
        help="Write this many clean/corrupted SVG and PNG pairs for inspection.",
    )
    parser.add_argument("--inspect-root", type=Path, default=INSPECT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_datasets = args.dataset or sorted(DATASETS)
    corruption_types = args.corruption_type or sorted(CORRUPTIONS)
    per_row = max(args.per_row, 1)
    inspect_dataset: Dataset | None = None

    for dataset_key in selected_datasets:
        spec = DATASETS[dataset_key]
        selected_splits = args.split or list(spec.splits)
        for split_name in selected_splits:
            if split_name not in spec.splits:
                print(f"skipped: {spec.key} does not define split {split_name}")
                continue

            repair_dataset = process_split(
                dataset_key=dataset_key,
                spec=spec,
                split_name=split_name,
                input_root=args.input_root,
                output_root=args.output_root,
                corruption_types=corruption_types,
                corruption_level=args.corruption_level,
                per_row=per_row,
                seed=args.seed,
                start_index=args.start_index,
                sample_size=args.sample_size,
                min_render_mse=args.min_render_mse,
                min_changed_pixel_ratio=args.min_changed_pixel_ratio,
            )
            inspect_dataset = inspect_dataset or repair_dataset

    if inspect_dataset is not None:
        write_inspection_files(
            inspect_dataset,
            output_dir=args.inspect_root,
            inspect_size=args.inspect_size,
        )
        if args.inspect_size:
            print(f"inspection: {args.inspect_root}")


if __name__ == "__main__":
    main()
