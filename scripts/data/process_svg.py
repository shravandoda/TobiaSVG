import base64
import os
import re
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path
from typing import Any

from groq import Groq


IMAGE_LABEL_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
IMAGE_LABEL_PROMPT = (
    "Describe the contents of the image in 2-3 sentences. "
    "Be concise but include the details."
)
SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)
MIN_STRUCTURAL_RATIO = 0.4
MAX_COMPLEX_SHAPES = 50
BASIC_PRIMITIVES = {"rect", "circle", "ellipse"}
CONNECTORS = {"line", "polyline"}
COMPLEX_SHAPES = {"path", "polygon"}


class SvgQualityError(ValueError):
    pass


@lru_cache(maxsize=1)
def get_groq_client() -> Groq:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is required to generate labels.")

    return Groq(api_key=api_key)


def encode_image(image_path: Path) -> str:
    with image_path.open("rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def clean_svg(svg_text: str) -> str:
    """Return a normalized SVG string that can be rendered to PNG."""
    svg_text = maybe_decode_escaped_svg(svg_text)
    svg_text = repair_missing_svg_closings(svg_text)

    root = ET.fromstring(svg_text)
    inner_svg = find_inner_svg(root)

    if inner_svg is None:
        return ET.tostring(root, encoding="unicode")

    for key, value in root.attrib.items():
        if key not in inner_svg.attrib:
            inner_svg.set(key, value)

    return ET.tostring(inner_svg, encoding="unicode")


def maybe_decode_escaped_svg(svg_text: str) -> str:
    if "<svg" in svg_text:
        return svg_text

    try:
        decoded = svg_text.encode("utf-8").decode("unicode_escape")
    except UnicodeDecodeError:
        return svg_text

    return decoded if "<svg" in decoded else svg_text


def repair_missing_svg_closings(svg_text: str) -> str:
    opening_svg_count = len(re.findall(r"<svg\b(?=[^>]*>)(?![^>]*?/>)", svg_text))
    closing_svg_count = len(re.findall(r"</svg\s*>", svg_text))
    missing = opening_svg_count - closing_svg_count

    if missing > 0:
        svg_text = svg_text.rstrip() + ("\n</svg>" * missing)

    return svg_text


def find_inner_svg(root: ET.Element) -> ET.Element | None:
    for child in root:
        if child.tag == f"{{{SVG_NS}}}svg" or child.tag == "svg":
            return child

    return None


def local_tag_name(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1].lower()


def get_svg_quality_stats(svg_text: str) -> dict[str, float | int]:
    root = ET.fromstring(svg_text)
    basic_count = 0
    connector_count = 0
    complex_count = 0

    for element in root.iter():
        tag_name = local_tag_name(element)

        if tag_name in BASIC_PRIMITIVES:
            basic_count += 1
        elif tag_name in CONNECTORS:
            connector_count += 1
        elif tag_name in COMPLEX_SHAPES:
            complex_count += 1

    structural_count = basic_count + connector_count
    geometric_count = structural_count + complex_count
    structural_ratio = structural_count / geometric_count if geometric_count else 0.0

    return {
        "basic_count": basic_count,
        "connector_count": connector_count,
        "complex_count": complex_count,
        "geometric_count": geometric_count,
        "structural_ratio": structural_ratio,
    }


def validate_svg_quality(svg_text: str) -> None:
    stats = get_svg_quality_stats(svg_text)

    if stats["geometric_count"] == 0:
        raise SvgQualityError("SVG quality filter failed: no geometric elements found.")

    if stats["structural_ratio"] < MIN_STRUCTURAL_RATIO:
        raise SvgQualityError(
            "SVG quality filter failed: structural primitive ratio "
            f"{stats['structural_ratio']:.2f} < {MIN_STRUCTURAL_RATIO:.2f}."
        )

    if stats["complex_count"] > MAX_COMPLEX_SHAPES:
        raise SvgQualityError(
            "SVG quality filter failed: complex shape count "
            f"{stats['complex_count']} > {MAX_COMPLEX_SHAPES}."
        )


def render_svg_to_png(svg_text: str, image_path: Path) -> None:
    """Render cleaned SVG text to a PNG file at image_path."""
    from cairosvg import svg2png

    image_path.parent.mkdir(parents=True, exist_ok=True)
    svg2png(bytestring=svg_text.encode("utf-8"), write_to=str(image_path))


def build_label_prompt(row: dict[str, Any]) -> str:
    """Build the prompt used to label one processed row."""
    return "Describe the image in 2-3 sentences. Be concise but include all the major details."


def extract_label_from_response(response: Any) -> str:
    """Extract the label text from the Groq chat completion response."""
    return response.choices[0].message.content or ""


def request_text_label(
    image_path: Path,
    *,
    prompt: str,
    model: str = IMAGE_LABEL_MODEL,
) -> str:
    base64_image = encode_image(image_path)
    client = get_groq_client()

    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                        },
                    },
                ],
            }
        ],
        model=model,
    )

    return extract_label_from_response(response)
