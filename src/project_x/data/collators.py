"""Batch task examples into model-ready tensors."""

import torch

from src.project_x.modeling.chat import (
    get_image2svg_prompt,
    get_repair_prompt,
    get_text2svg_prompt,
    serialize_prompt,
)
from src.project_x.modeling.loading import get_processor
from src.project_x.utils.svg import svg2pil


TOKEN_ALIGNED_EXTRA_KEYS = ("mm_token_type_ids",)
IMAGE_KEYS = ("pixel_values", "image_grid_thw")


def _target_ids(processor, svg: str) -> list[int]:
    tokenizer = processor.tokenizer
    target_text = svg + tokenizer.eos_token + "\n"
    return tokenizer(target_text, add_special_tokens=False)["input_ids"]


def _pad_1d_tensors(tensors: list[torch.Tensor], pad_value: int) -> torch.Tensor:
    max_length = max(tensor.numel() for tensor in tensors)
    padded = tensors[0].new_full((len(tensors), max_length), pad_value)

    for index, tensor in enumerate(tensors):
        padded[index, : tensor.numel()] = tensor

    return padded


def _prepare_prompt_batch(prompt_texts: list[str], images: list | None = None):
    processor = get_processor()
    kwargs = {
        "text": prompt_texts,
        "return_tensors": "pt",
        "padding": True,
    }
    if images is not None:
        kwargs["images"] = images

    return processor(**kwargs)


def _append_targets(prompt_batch, target_svgs: list[str]):
    processor = get_processor()
    tokenizer = processor.tokenizer
    input_ids = []
    attention_masks = []
    labels = []
    token_aligned_extras = {key: [] for key in TOKEN_ALIGNED_EXTRA_KEYS}

    for index, svg in enumerate(target_svgs):
        prompt_length = int(prompt_batch["attention_mask"][index].sum().item())
        prompt_ids = prompt_batch["input_ids"][index, :prompt_length]
        target_ids = torch.tensor(
            _target_ids(processor, svg),
            dtype=prompt_ids.dtype,
            device=prompt_ids.device,
        )

        example_input_ids = torch.cat([prompt_ids, target_ids])
        example_labels = torch.cat(
            [
                torch.full_like(prompt_ids, -100),
                target_ids,
            ]
        )
        example_attention_mask = torch.ones_like(example_input_ids)

        input_ids.append(example_input_ids)
        labels.append(example_labels)
        attention_masks.append(example_attention_mask)

        for key in TOKEN_ALIGNED_EXTRA_KEYS:
            if key in prompt_batch:
                prompt_values = prompt_batch[key][index, :prompt_length]
                target_values = torch.zeros_like(target_ids)
                token_aligned_extras[key].append(
                    torch.cat([prompt_values, target_values])
                )

    batch = {
        "input_ids": _pad_1d_tensors(input_ids, tokenizer.pad_token_id),
        "attention_mask": _pad_1d_tensors(attention_masks, 0),
        "labels": _pad_1d_tensors(labels, -100),
    }

    for key, values in token_aligned_extras.items():
        if values:
            batch[key] = _pad_1d_tensors(values, 0)

    for key in IMAGE_KEYS:
        if key in prompt_batch:
            batch[key] = prompt_batch[key]

    return batch


def text2svg_collator(rows: list[dict]):
    processor = get_processor()
    prompt_texts = [
        serialize_prompt(processor, get_text2svg_prompt(row["text"])) for row in rows
    ]
    target_svgs = [row["svg"] for row in rows]
    prompt_batch = _prepare_prompt_batch(prompt_texts)

    return _append_targets(prompt_batch, target_svgs)


def image2svg_collator(rows: list[dict]):
    processor = get_processor()
    images = [svg2pil(row["svg"]) for row in rows]
    prompt_texts = [
        serialize_prompt(processor, get_image2svg_prompt(image)) for image in images
    ]
    target_svgs = [row["svg"] for row in rows]
    prompt_batch = _prepare_prompt_batch(prompt_texts, images)

    return _append_targets(prompt_batch, target_svgs)


def repair_collator(rows: list[dict]):
    processor = get_processor()
    images = [svg2pil(row["svg"]) for row in rows]
    prompt_texts = [
        serialize_prompt(
            processor,
            get_repair_prompt(image, row["corrupted_svg"]),
        )
        for image, row in zip(images, rows, strict=True)
    ]
    target_svgs = [row["svg"] for row in rows]
    prompt_batch = _prepare_prompt_batch(prompt_texts, images)

    return _append_targets(prompt_batch, target_svgs)
