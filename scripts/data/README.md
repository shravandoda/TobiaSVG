# Data Scripts

This project keeps research/reference material and generated data separate:

- `artifacts/`: papers, notes, and research artifacts.
- `data/`: generated local datasets, checkpoints, rendered images, and reports.

Do not merge these directories. `artifacts/` answers "what did we read or inspect?"
while `data/` answers "what did we generate locally?"

`scripts/data` is intentionally thin. Reusable implementation lives in
`src/project_x/preprocessing`.

Processed caption datasets live under:

```text
data/processed/datasets/{dataset_key}/{split}
```

Repair datasets live under:

```text
data/processed/repair_datasets/{dataset_key}
```

Common commands:

```bash
uv run python -m scripts.data.sample_dataset --dataset starvector_diagrams --sample-size 10
uv run python -m scripts.data.process_dataset --dataset starvector_diagrams --sample-size 10
uv run python -m scripts.data.build_repair_dataset --sample-size 100 --workers 8
uv run python -m scripts.data.build_repair_dataset --package-only
uv run python -m scripts.data.analyze_svg_stats
```
