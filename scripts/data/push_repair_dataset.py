"""Push a packaged repair dataset to the Hugging Face Hub."""

import argparse
from pathlib import Path

from datasets import load_from_disk
from dotenv import load_dotenv

from src.project_x.constants import HF_TOKEN

load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Push the packaged TobiaSVG repair dataset to the Hub."
    )
    parser.add_argument("--repo-id", required=True)
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("data/processed/repair_datasets/tobiasvg_repair"),
    )
    parser.add_argument(
        "--private",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_from_disk(str(args.dataset_path))
    dataset.push_to_hub(args.repo_id, private=args.private, token=HF_TOKEN)


if __name__ == "__main__":
    main()
