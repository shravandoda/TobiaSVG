# SVG Fine-Tuning Curriculum

## Dataset Inventory

Current configured dataset keys:

| Dataset key | Target rows | Caption source/style | Notes |
| --- | ---: | --- | --- |
| `starvector` | 40,474 | Gemini concise | SVG diagrams; includes original `train` and `test` splits. |
| `vfig_shapes` | 6,545 | Gemini concise | Synthetic shape/arrow diagrams. |
| `vfig_complex` | 60,000 | Gemini concise + detailed | Complex synthetic diagrams; roughly half concise and half detailed. |
| `starvector_emoji` | 10,043 | Gemini concise | Emoji SVGs; primitive-quality filter disabled. |
| `svg_animals` | 1,416 | Source prompt | Animal illustration text-to-SVG prompts; no label generation. |
| `svgx_core` | 40,000 | `qwen_caption` | General SVG data; no label generation. |

This is six dataset keys. If we want exactly five groups, the likely grouping is to merge `starvector_emoji`, `svg_animals`, and `svgx_core` under a broader "illustration/icon" family, or to exclude one dataset from the first training run.

## Split Policy

Use an 80/10/10 split for each dataset family:

- 80% train
- 10% validation
- 10% test

For datasets that already have named splits, treat the saved processed rows as a pool first, then create our own consistent 80/10/10 split. This avoids uneven evaluation caused by different source split definitions.

Recommended split seed:

```text
42
```

The split should preserve these metadata columns:

```text
filename
svg
text
caption_style
source_dataset
```

`source_dataset` should be added when combining datasets so we can evaluate per dataset later.

## Training Tasks

Each row can produce two task variants:

1. Text-to-SVG
   - input: `text`
   - target: `svg`

2. Image-to-SVG
   - input: rendered image from `svg`
   - target: `svg`

Use the same train/validation/test split for both task variants. Do not allow one row's text-to-SVG example in train and image-to-SVG example in validation/test.

## Curriculum

Use staged training. Each stage is dominated by one new dataset and includes a small replay mix from earlier datasets so the model does not forget earlier skills.

### Curriculum 1: Basic Diagram Shapes

Primary dataset:

- `vfig_shapes`

Mix:

```text
100% vfig_shapes
```

Goal: teach basic SVG syntax, simple shapes, arrows, labels, and small diagram layouts.

### Curriculum 2: Animal Illustrations

Primary dataset:

- `svg_animals`

Replay:

- `vfig_shapes`

Mix:

```text
80% svg_animals
20% vfig_shapes
```

Goal: add simple illustration-style SVG generation while retaining basic diagram structure.

### Curriculum 3: Emoji/Icon SVGs

Primary dataset:

- `starvector_emoji`

Replay:

- `svg_animals`
- `vfig_shapes`

Mix:

```text
80% starvector_emoji
10% svg_animals
10% vfig_shapes
```

Goal: add compact icon/emoji SVGs, richer colors, and stylized shapes.

### Curriculum 4: General SVG Core

Primary dataset:

- `svgx_core`

Replay:

- `starvector_emoji`
- `svg_animals`
- `vfig_shapes`

Mix:

```text
75% svgx_core
10% starvector_emoji
10% svg_animals
5% vfig_shapes
```

Goal: broaden general SVG coverage using the large SVGX slice while retaining illustration/icon behavior.

### Curriculum 5: StarVector Diagrams

Primary dataset:

- `starvector`

Replay:

- `svgx_core`
- `starvector_emoji`
- `svg_animals`
- `vfig_shapes`

Suggested mix:

```text
75% starvector
10% svgx_core
5% starvector_emoji
5% svg_animals
5% vfig_shapes
```

Goal: specialize back toward diagrams with real-world diagram SVGs while keeping general SVG ability.

### Curriculum 6: Complex Diagrams

Primary dataset:

- `vfig_complex`

Replay:

- `starvector`
- `svgx_core`
- `vfig_shapes`
- `starvector_emoji`
- `svg_animals`

Suggested mix:

```text
75% vfig_complex
10% starvector
5% svgx_core
5% vfig_shapes
3% starvector_emoji
2% svg_animals
```

Goal: teach complex diagram layouts, dense labels, detailed prompts, and robust diagram generation.

### Final Stabilization Mix

After Curriculum 6, run one final mixed pass over all datasets using the final desired behavior distribution.

Suggested mix:

```text
40% vfig_complex
25% starvector
15% svgx_core
8% starvector_emoji
7% vfig_shapes
5% svg_animals
```

Goal: reduce stage-specific drift and make the final model behave consistently across all target domains.

## Sampling Rules

Create train/validation/test splits once per dataset, then sample only within the appropriate split for every curriculum stage.

During training, random sampling with replacement is acceptable and usually preferred. A replay example can appear multiple times across stages; that is the point of replay. Do not move examples across train/validation/test boundaries.

For checkpointing and reproducibility, each curriculum stage should record:

- dataset mixture weights
- random seed
- number of training examples or steps
- source dataset for every sampled example

## Task Mix

Each curriculum can produce both task variants:

- text-to-SVG
- image-to-SVG

Start with:

```text
50% text-to-SVG
50% image-to-SVG
```

If image vectorization quality is weaker than text generation, shift toward:

```text
40% text-to-SVG
60% image-to-SVG
```

## Evaluation

Evaluate separately by dataset family and task:

- text-to-SVG on each dataset
- image-to-SVG on each dataset
- concise vs detailed prompts for `vfig_complex`

Track at minimum:

- SVG parse/render success rate
- output token length
- visual similarity against target render
- qualitative samples per dataset

## Current Status

Completed processed datasets:

- `starvector`
- `vfig_shapes`
- `vfig_complex`

Pending processed datasets:

- `starvector_emoji`
- `svg_animals`
- `svgx_core`
