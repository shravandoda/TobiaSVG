# Data Scripts

This project keeps research/reference material and generated data separate:

- `artifacts/`: papers, notes, and research artifacts.
- `data/`: generated local datasets, checkpoints, rendered images, and reports.

Do not merge these directories. `artifacts/` answers "what did we read or inspect?"
while `data/` answers "what did we generate locally?"

Processed datasets live under:

```text
data/processed/datasets/{dataset_key}/{split}
```

Push processed datasets with:

```bash
uv run python -m scripts.data.push_dataset --repo-id USER_OR_ORG/REPO --dry-run
```

Remove `--dry-run` when the split summary looks correct.
