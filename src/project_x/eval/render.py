"""Extract and render SVG output produced during evaluation."""

from pathlib import Path

from project_x.utils.svg import svg2pil


class SvgExtractionError(ValueError):
    """Raised when generated text does not contain a complete SVG document."""


def extract_svg(generated_text: str) -> str:
    """Return the first complete SVG document found in generated text."""
    start = generated_text.find("<svg")
    end = generated_text.find("</svg>", start)

    if start == -1 or end == -1:
        raise SvgExtractionError("Generated text does not contain a complete SVG.")

    return generated_text[start : end + len("</svg>")].strip()


def render_svg(svg_text: str, output_path: Path) -> None:
    """Render SVG markup to a PNG file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = svg2pil(svg_text)
    image.save(output_path, format="PNG")
