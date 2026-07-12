"""Compute text, SVG, rendering, and repair-quality evaluation metrics."""

from pathlib import Path
from xml.etree import ElementTree as ET

from PIL import Image, ImageChops


def is_valid_svg(svg_text: str) -> bool:
    """Return whether text is parseable XML with an SVG root element."""
    try:
        root = ET.fromstring(svg_text)
    except ET.ParseError:
        return False

    return root.tag.rsplit("}", 1)[-1].lower() == "svg"


def normalized_pixel_mse(target_path: Path, prediction_path: Path) -> float:
    """Return normalized RGBA pixel MSE after matching the target dimensions."""
    with Image.open(target_path) as target_image:
        target = target_image.convert("RGBA")

    with Image.open(prediction_path) as prediction_image:
        prediction = prediction_image.convert("RGBA")

    if prediction.size != target.size:
        prediction = prediction.resize(target.size, Image.Resampling.LANCZOS)

    histogram = ImageChops.difference(target, prediction).histogram()
    squared_error = sum(
        count * ((value_index % 256) ** 2)
        for value_index, count in enumerate(histogram)
    )
    channel_count = target.width * target.height * 4
    return squared_error / (channel_count * 255**2)
