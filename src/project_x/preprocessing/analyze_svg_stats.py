import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from datasets import Dataset, load_from_disk

from project_x.constants import DATASETS
from project_x.preprocessing.paths import build_split_output_path
from project_x.preprocessing.svg import get_svg_quality_stats

REPORT_ROOT = Path("data/processed/reports")


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def empty_totals() -> dict[str, int]:
    return {
        "rows": 0,
        "parse_errors": 0,
        "basic_count": 0,
        "connector_count": 0,
        "complex_count": 0,
        "geometric_count": 0,
    }


def update_totals(totals: dict[str, int], row: dict) -> None:
    totals["rows"] += 1
    try:
        stats = get_svg_quality_stats(row["svg"])
    except ET.ParseError:
        totals["parse_errors"] += 1
        return

    totals["basic_count"] += int(stats["basic_count"])
    totals["connector_count"] += int(stats["connector_count"])
    totals["complex_count"] += int(stats["complex_count"])
    totals["geometric_count"] += int(stats["geometric_count"])


def summarize_totals(dataset_key: str, split_name: str, totals: dict[str, int]) -> dict:
    geometric_count = totals["geometric_count"]
    basic_count = totals["basic_count"]
    connector_count = totals["connector_count"]
    complex_count = totals["complex_count"]
    simple_count = basic_count + connector_count

    return {
        "dataset": dataset_key,
        "split": split_name,
        "rows": totals["rows"],
        "parse_errors": totals["parse_errors"],
        "basic_count": basic_count,
        "connector_count": connector_count,
        "simple_count": simple_count,
        "complex_count": complex_count,
        "geometric_count": geometric_count,
        "basic_pct": safe_divide(basic_count, geometric_count),
        "connector_pct": safe_divide(connector_count, geometric_count),
        "simple_pct": safe_divide(simple_count, geometric_count),
        "complex_pct": safe_divide(complex_count, geometric_count),
        "avg_geometric_per_row": safe_divide(
            geometric_count,
            totals["rows"] - totals["parse_errors"],
        ),
    }


def analyze_split(dataset_key: str, split_name: str, dataset: Dataset) -> dict:
    totals = empty_totals()
    for row in dataset:
        update_totals(totals, row)

    return summarize_totals(dataset_key, split_name, totals)


def merge_summaries(dataset_key: str, summaries: list[dict]) -> dict:
    totals = empty_totals()
    for summary in summaries:
        totals["rows"] += summary["rows"]
        totals["parse_errors"] += summary["parse_errors"]
        totals["basic_count"] += summary["basic_count"]
        totals["connector_count"] += summary["connector_count"]
        totals["complex_count"] += summary["complex_count"]
        totals["geometric_count"] += summary["geometric_count"]

    return summarize_totals(dataset_key, "all", totals)


def format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def format_markdown_table(summaries: list[dict]) -> str:
    headers = [
        "dataset",
        "split",
        "rows",
        "parse_errors",
        "basic%",
        "connector%",
        "simple%",
        "complex%",
        "avg_geom/row",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for summary in summaries:
        lines.append(
            "| "
            + " | ".join(
                [
                    summary["dataset"],
                    summary["split"],
                    str(summary["rows"]),
                    str(summary["parse_errors"]),
                    format_pct(summary["basic_pct"]),
                    format_pct(summary["connector_pct"]),
                    format_pct(summary["simple_pct"]),
                    format_pct(summary["complex_pct"]),
                    f"{summary['avg_geometric_per_row']:.1f}",
                ]
            )
            + " |"
        )

    return "\n".join(lines)


def write_reports(summaries: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "svg_stats.json"
    markdown_path = output_dir / "svg_stats.md"

    json_path.write_text(json.dumps(summaries, indent=2) + "\n")
    markdown_path.write_text(
        "# SVG Primitive Statistics\n\n" + format_markdown_table(summaries) + "\n"
    )
    print(f"saved: {json_path}")
    print(f"saved: {markdown_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze simple/complex SVG primitive composition."
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=sorted(DATASETS),
        default=None,
        help="Dataset key to analyze. Pass multiple times, or omit for all.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPORT_ROOT,
        help="Directory where reports are written.",
    )
    parser.add_argument(
        "--include-splits",
        action="store_true",
        help="Include per-split rows in addition to the dataset-level all row.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_datasets = args.dataset or sorted(DATASETS)
    summaries: list[dict] = []

    for dataset_key in selected_datasets:
        spec = DATASETS[dataset_key]
        split_summaries: list[dict] = []
        for split_name in spec.splits:
            split_path = build_split_output_path(spec=spec, split_name=split_name)
            if not split_path.exists():
                print(f"missing: {split_path}")
                continue

            dataset = load_from_disk(str(split_path))
            if not isinstance(dataset, Dataset):
                raise TypeError(f"Expected a Dataset at {split_path}")

            summary = analyze_split(dataset_key, split_name, dataset)
            split_summaries.append(summary)

            if args.include_splits:
                summaries.append(summary)

        if split_summaries:
            summaries.append(merge_summaries(dataset_key, split_summaries))

    write_reports(summaries, args.output_dir)


if __name__ == "__main__":
    main()
