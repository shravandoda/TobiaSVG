import re
import xml.etree.ElementTree as ET

from cairosvg import svg2png
from datasets import load_dataset

SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)


def repair_missing_svg_closings(svg_text: str) -> str:
    """
    Repairs SVG text when it has more opening <svg> tags than closing </svg> tags.
    Example:
        <svg ...>
          <svg>
            ...
          </svg>

    becomes:
        <svg ...>
          <svg>
            ...
          </svg>
        </svg>
    """

    opening_svg_count = len(re.findall(r"<svg(?:\s|>|/)", svg_text))
    closing_svg_count = len(re.findall(r"</svg\s*>", svg_text))

    missing = opening_svg_count - closing_svg_count

    if missing > 0:
        svg_text = svg_text.rstrip() + ("\n</svg>" * missing)

    return svg_text


def clean_svg(svg_text: str) -> tuple[str, str | None, str | None]:
    """
    Cleans an SVG that may look like this:

        <svg width="700" height="800" xmlns="http://www.w3.org/2000/svg">
            <svg>
                ...
            </svg>
        </svg>

    Returns:
        cleaned_svg_text, width, height
    """

    root = ET.fromstring(svg_text)

    width = root.attrib.get("width")
    height = root.attrib.get("height")

    # Find nested SVG directly inside the root SVG
    inner_svg = None
    for child in root:
        if child.tag == f"{{{SVG_NS}}}svg" or child.tag == "svg":
            inner_svg = child
            break

    # If there is no nested SVG, return original SVG
    if inner_svg is None:
        cleaned_svg = ET.tostring(root, encoding="unicode")
        return cleaned_svg, width, height

    # Preserve root attributes like width, height, xmlns, viewBox, etc.
    for key, value in root.attrib.items():
        if key not in inner_svg.attrib:
            inner_svg.set(key, value)

    cleaned_svg = ET.tostring(inner_svg, encoding="unicode")

    return cleaned_svg, width, height


iterable_dataset = load_dataset("starvector/svg-diagrams", split="train", streaming=True)

# Inspecting the StarVector diagram dataset
iterable_dataset = iterable_dataset.take(5000)
print(iterable_dataset.column_names)

count = 0

for example in iterable_dataset:
    count += 1
    if count == 4200:
        break

## Use Cairo SVG to render this to PNG
filename: str = example["Filename"]
svg: str = example["Svg"]
svg = svg.encode("utf-8").decode("unicode_escape")

svg = repair_missing_svg_closings(svg)
svg, width, height = clean_svg(svg)

svg2png(svg, write_to=f"{filename}.png")
