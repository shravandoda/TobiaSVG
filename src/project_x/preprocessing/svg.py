import base64
import logging
import os
import random
import re
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from groq import Groq, RateLimitError

GROQ_IMAGE_LABEL_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
GEMINI_IMAGE_LABEL_MODEL = "gemini-3.1-flash-lite"
CONCISE_IMAGE_LABEL_PROMPT = (
    "Describe the image in 3-4 concise sentences. "
    "Preserve important visible text exactly when readable, especially labels, "
    "node names, field names, and values. When the image contains metadata boxes "
    "or table-like text, include the key visible entries exactly. Include visual "
    "structure and connections, but skip details you are unsure about."
)
DETAILED_IMAGE_LABEL_PROMPT = (
    "Describe the diagram in 5-7 detailed but concise sentences. "
    "Preserve all readable text exactly, including node labels, table names, "
    "field names, values, and captions. Describe the main visual structure, "
    "important shapes, groups, arrows, connectors, hierarchy, and layout. "
    "Mention colors, line styles, and repeated patterns only when they help "
    "identify parts of the diagram. If the image looks like a technical diagram, "
    "explain the apparent relationships or flow between the visible elements. "
    "Skip any detail you are unsure about rather than guessing."
)
IMAGE_LABEL_PROMPTS = {
    "concise": CONCISE_IMAGE_LABEL_PROMPT,
    "detailed": DETAILED_IMAGE_LABEL_PROMPT,
}
IMAGE_LABEL_MAX_COMPLETION_TOKENS = 512
IMAGE_LABEL_RATE_LIMIT_RETRIES = 12
IMAGE_LABEL_TIMEOUT_SECONDS = 60
SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)
MIN_STRUCTURAL_RATIO = 0.4
MAX_COMPLEX_SHAPES = 50
BASIC_PRIMITIVES = {"rect", "circle", "ellipse"}
CONNECTORS = {"line", "polyline"}
COMPLEX_SHAPES = {"path", "polygon"}
THREAD_STATE = threading.local()
CAIROSVG_RENDER_LOCK = threading.Lock()
logger = logging.getLogger(__name__)


class SvgQualityError(ValueError):
    pass


class TextLabelError(ValueError):
    pass


def get_groq_client() -> Groq:
    client = getattr(THREAD_STATE, "groq_client", None)
    if client is not None:
        return client

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is required to generate labels.")

    client = Groq(api_key=api_key)
    THREAD_STATE.groq_client = client
    return client


def get_gemini_client() -> genai.Client:
    client = getattr(THREAD_STATE, "gemini_client", None)
    if client is not None:
        return client

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is required to generate Gemini labels.")

    client = genai.Client(api_key=api_key)
    THREAD_STATE.gemini_client = client
    return client


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
    try:
        with CAIROSVG_RENDER_LOCK:
            svg2png(bytestring=svg_text.encode("utf-8"), write_to=str(image_path))
    except Exception as exc:
        raise SvgQualityError(f"SVG render failed: {exc}") from exc


def clean_validate_and_render_svg(
    svg_text: str,
    image_path: Path,
    *,
    validate_quality: bool = True,
) -> str:
    """Normalize an SVG, run quality checks, render it, and return cleaned SVG."""
    try:
        svg = clean_svg(svg_text)
        if validate_quality:
            validate_svg_quality(svg)
        render_svg_to_png(svg, image_path)
    except ET.ParseError as exc:
        raise SvgQualityError(f"SVG parse failed: {exc}") from exc

    return svg


def build_label_prompt(row: dict[str, Any], *, caption_style: str) -> str:
    """Build the prompt used to label one processed row."""
    try:
        return IMAGE_LABEL_PROMPTS[caption_style]
    except KeyError as exc:
        available_styles = ", ".join(sorted(IMAGE_LABEL_PROMPTS))
        raise ValueError(
            f"Unknown caption style `{caption_style}`. "
            f"Available styles: {available_styles}"
        ) from exc


def extract_label_from_response(response: Any) -> str:
    """Extract the label text from the Groq chat completion response."""
    return response.choices[0].message.content or ""


def get_retry_after_seconds(exc: Exception, attempt: int) -> float:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", {})
    retry_after = headers.get("retry-after") if headers else None

    if retry_after is not None:
        try:
            return float(retry_after)
        except ValueError:
            pass

    message = str(exc)
    retry_ms_match = re.search(r"try again in ([0-9.]+)ms", message, re.IGNORECASE)
    if retry_ms_match:
        return float(retry_ms_match.group(1)) / 1000

    retry_s_match = re.search(r"try again in ([0-9.]+)s", message, re.IGNORECASE)
    if retry_s_match:
        return float(retry_s_match.group(1))

    return min(0.5 * (2**attempt), 8.0)


def sleep_before_retry(provider: str, exc: Exception, attempt: int) -> None:
    retry_after = get_retry_after_seconds(exc, attempt)
    sleep_seconds = max(retry_after, min(0.5 * (2**attempt), 8.0))
    sleep_seconds += random.uniform(0, 0.25)
    logger.warning(
        "%s rate limited; sleeping %.2fs before retry %s/%s",
        provider,
        sleep_seconds,
        attempt + 1,
        IMAGE_LABEL_RATE_LIMIT_RETRIES,
    )
    time.sleep(sleep_seconds)


def is_retryable_gemini_error(exc: genai_errors.APIError) -> bool:
    status = getattr(exc, "status", None)
    return status in {429, 500, 502, 503, 504}


def request_groq_text_label(
    image_path: Path,
    *,
    prompt: str,
    model: str,
) -> str:
    base64_image = encode_image(image_path)
    client = get_groq_client()
    messages: Any = [
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
    ]

    for attempt in range(IMAGE_LABEL_RATE_LIMIT_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                messages=messages,
                model=model,
                max_completion_tokens=IMAGE_LABEL_MAX_COMPLETION_TOKENS,
                timeout=IMAGE_LABEL_TIMEOUT_SECONDS,
            )
            return extract_label_from_response(response)
        except RateLimitError as exc:
            if attempt >= IMAGE_LABEL_RATE_LIMIT_RETRIES:
                raise TextLabelError(f"Groq label request failed: {exc}") from exc

            sleep_before_retry("Groq", exc, attempt)

    raise RuntimeError("Groq label request failed without returning or raising.")


def request_gemini_text_label(
    image_path: Path,
    *,
    prompt: str,
    model: str,
) -> str:
    client = get_gemini_client()
    contents: Any = [
        genai_types.Part.from_text(text=prompt),
        genai_types.Part.from_bytes(
            data=image_path.read_bytes(),
            mime_type="image/png",
        ),
    ]
    config = genai_types.GenerateContentConfig(
        max_output_tokens=IMAGE_LABEL_MAX_COMPLETION_TOKENS,
        temperature=0.2,
    )

    for attempt in range(IMAGE_LABEL_RATE_LIMIT_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
            return response.text or ""
        except genai_errors.APIError as exc:
            if (
                attempt >= IMAGE_LABEL_RATE_LIMIT_RETRIES
                or not is_retryable_gemini_error(exc)
            ):
                raise TextLabelError(f"Gemini label request failed: {exc}") from exc

            sleep_before_retry("Gemini", exc, attempt)

    raise RuntimeError("Gemini label request failed without returning or raising.")


def get_default_label_model(label_provider: str) -> str:
    if label_provider == "groq":
        return GROQ_IMAGE_LABEL_MODEL
    if label_provider == "gemini":
        return GEMINI_IMAGE_LABEL_MODEL

    raise ValueError(f"Unknown label provider: {label_provider}")


def request_text_label(
    image_path: Path,
    *,
    prompt: str,
    label_provider: str,
    model: str,
) -> str:
    if label_provider == "groq":
        return request_groq_text_label(image_path, prompt=prompt, model=model)
    if label_provider == "gemini":
        return request_gemini_text_label(image_path, prompt=prompt, model=model)

    raise ValueError(f"Unknown label provider: {label_provider}")
