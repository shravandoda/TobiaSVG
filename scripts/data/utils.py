from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from scripts.data.sample_dataset import DatasetSpec

IMAGE_ROOT = Path("data/images")
OUTPUT_ROOT = Path("data/processed/datasets")


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


def get_source_filename(row: dict[str, Any], spec: DatasetSpec) -> str:
    return str(
        get_required_row_value(
            row,
            column_name=spec.id_column,
            dataset_key=spec.key,
            value_name="filename",
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
