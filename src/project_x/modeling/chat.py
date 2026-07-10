"""Build and serialize model-specific prompt messages."""

from PIL.Image import Image

from project_x.constants import SYSTEM_MESSAGE


def get_text2svg_prompt(text: str):
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": SYSTEM_MESSAGE,
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": text,
                }
            ],
        },
    ]


def get_image2svg_prompt(image: Image):
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": SYSTEM_MESSAGE,
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Generate valid SVG markup for this image.",
                },
                {
                    "type": "image",
                    "image": image,
                },
            ],
        },
    ]


def get_repair_prompt(image: Image, corrupted_svg: str):
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": SYSTEM_MESSAGE,
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Repair this corrupted SVG so it matches the image. "
                        "Return only the corrected SVG markup.\n\n"
                        f"{corrupted_svg}"
                    ),
                },
                {
                    "type": "image",
                    "image": image,
                },
            ],
        },
    ]


def serialize_prompt(processor, messages: list[dict]) -> str:
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
