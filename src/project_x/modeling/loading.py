"""Load models, processors, tokenizers, and related model configuration."""

from functools import cache

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, Qwen3VLProcessor

from src.project_x.constants import MODEL_ID


@cache
def get_processor() -> Qwen3VLProcessor:
    return AutoProcessor.from_pretrained(MODEL_ID)


@cache
def get_model():
    return AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16
    )
