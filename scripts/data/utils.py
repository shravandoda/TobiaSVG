from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from scripts.data.sample_dataset import DatasetSpec


def load_env_file(path: Path = Path(".env")) -> None:
    if not path.exists():
        return

    load_dotenv(path)


def get_original_filename(row: dict[str, Any], spec: DatasetSpec) -> str:
    if not spec.id_column:
        raise ValueError(f"{spec.key} does not define an original filename column.")

    if spec.id_column not in row:
        raise ValueError(
            f"{spec.key} row is missing original filename column `{spec.id_column}`."
        )

    return str(row[spec.id_column])


def make_example_id(
    row: dict[str, Any],
    spec: DatasetSpec,
    *,
    split_name: str,
    index: int,
) -> str:
    return get_original_filename(row, spec)
