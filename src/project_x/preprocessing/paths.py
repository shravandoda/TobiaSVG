from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from project_x.constants import DatasetSpec

IMAGE_ROOT = Path("data/images")
OUTPUT_ROOT = Path("data/processed/datasets")
CHECKPOINT_ROOT = Path("data/processed/checkpoints")


def load_env_file(path: Path = Path(".env")) -> None:
    if not path.exists():
        return

    load_dotenv(path)


def get_required_row_value(
    row: dict[str, Any],
    *,
    column_name: str | None,
    dataset_key: str,
    value_name: str,
) -> Any:
    if column_name is None:
        raise ValueError(f"{dataset_key} does not define a {value_name} column.")

    if column_name not in row:
        raise ValueError(
            f"{dataset_key} row is missing {value_name} column `{column_name}`."
        )

    return row[column_name]


def get_source_filename(
    row: dict[str, Any],
    spec: DatasetSpec,
    *,
    split_name: str,
    row_index: int,
) -> str:
    if spec.id_column is None:
        return f"{spec.key}_{split_name}_{row_index:06d}.svg"

    return str(
        get_required_row_value(
            row,
            column_name=spec.id_column,
            dataset_key=spec.key,
            value_name="filename",
        )
    )


def get_source_text(row: dict[str, Any], spec: DatasetSpec) -> str | None:
    if spec.text_column is None:
        return None

    return str(
        get_required_row_value(
            row,
            column_name=spec.text_column,
            dataset_key=spec.key,
            value_name="text",
        )
    )


def get_source_svg(row: dict[str, Any], spec: DatasetSpec) -> str:
    return str(
        get_required_row_value(
            row,
            column_name=spec.svg_column,
            dataset_key=spec.key,
            value_name="SVG",
        )
    )


def build_image_path(
    *,
    spec: DatasetSpec,
    split_name: str,
    filename: str,
    image_root: Path = IMAGE_ROOT,
) -> Path:
    return image_root / spec.key / split_name / f"{filename}.png"


def build_split_output_path(
    *,
    spec: DatasetSpec,
    split_name: str,
    output_root: Path = OUTPUT_ROOT,
) -> Path:
    return output_root / spec.key / split_name


def build_split_checkpoint_path(
    *,
    spec: DatasetSpec,
    split_name: str,
    checkpoint_root: Path = CHECKPOINT_ROOT,
) -> Path:
    return checkpoint_root / spec.key / split_name
