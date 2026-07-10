import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN")

WANDB_API_KEY = os.environ.get("WANDB_API_KEY")
WANDB_PROJECT_NAME = "project_x"
WANDB_ENTITY = "shravandoda-georgia-institute-of-technology"


@dataclass(frozen=True)
class DatasetSpecSplit:
    sample_size: int
    target_size: int


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    path: str
    config_name: str | None
    svg_column: str
    id_column: str | None
    splits: dict[str, DatasetSpecSplit]
    text_column: str | None = None
    caption_style: str = "concise"
    validate_svg_quality: bool = True


DATASETS = {
    "starvector_diagrams": DatasetSpec(
        key="starvector_diagrams",
        path="starvector/svg-diagrams",
        config_name=None,
        svg_column="Svg",
        id_column="Filename",
        splits={
            "train": DatasetSpecSplit(
                sample_size=182_000,
                target_size=40_000,
            ),
            "test": DatasetSpecSplit(sample_size=474, target_size=474),
        },
    ),
    "vfig_shapes": DatasetSpec(
        key="vfig_shapes",
        path="QijiaHe/VFIG-Data",
        config_name="VFIG-Data-Shapes-and-Arrows",
        svg_column="svg",
        id_column="filename",
        splits={
            "train": DatasetSpecSplit(sample_size=6_545, target_size=6_545),
        },
    ),
    "vfig_diagrams": DatasetSpec(
        key="vfig_diagrams",
        path="QijiaHe/VFIG-Data",
        config_name="VFIG-Data-Complex-Diagrams",
        svg_column="svg",
        id_column="filename",
        splits={"train": DatasetSpecSplit(sample_size=60_000, target_size=60_000)},
        caption_style="detailed",
    ),
    "starvector_emoji": DatasetSpec(
        key="starvector_emoji",
        path="starvector/svg-emoji",
        config_name=None,
        svg_column="Svg",
        id_column="Filename",
        splits={
            "train": DatasetSpecSplit(sample_size=8_708, target_size=8_708),
            "test": DatasetSpecSplit(sample_size=668, target_size=668),
            "val": DatasetSpecSplit(sample_size=667, target_size=667),
        },
        caption_style="concise",
        validate_svg_quality=False,
    ),
    "animal_illustrations": DatasetSpec(
        key="animal_illustrations",
        path="yoavf/svg-animal-illustrations",
        config_name=None,
        svg_column="svg",
        id_column=None,
        splits={"train": DatasetSpecSplit(sample_size=1_416, target_size=1_416)},
        text_column="prompt",
        caption_style="source_prompt",
        validate_svg_quality=False,
    ),
    "svgx_core": DatasetSpec(
        key="svgx_core",
        path="xingxm/SVGX-Core-250k",
        config_name=None,
        svg_column="svg_code",
        id_column="uuid",
        splits={"train": DatasetSpecSplit(sample_size=40_000, target_size=40_000)},
        text_column="qwen_caption",
        caption_style="qwen_caption",
        validate_svg_quality=False,
    ),
}

DATA_PROCESSING_SEED = 42
SPLITS = {"train": 0.80, "test": 0.10, "val": 0.10}
MODEL_ID = "Qwen/Qwen3.5-4B"
MAX_SEQUENCE_LENGTH = 8192
SYSTEM_MESSAGE = (
    "You are an expert digital designer. Generate valid SVG markup that"
    " satisfies the user's request. Return only the SVG code, with no Markdown fences,"
    " no explanations, and no extra text."
)
