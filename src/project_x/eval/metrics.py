"""Compute text, SVG, rendering, and repair-quality evaluation metrics."""

from xml.etree import ElementTree as ET


def is_valid_svg(svg_text: str) -> bool:
    """Return whether text is parseable XML with an SVG root element."""
    try:
        root = ET.fromstring(svg_text)
    except ET.ParseError:
        return False

    return root.tag.rsplit("}", 1)[-1].lower() == "svg"
