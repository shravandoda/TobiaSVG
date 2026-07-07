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
dataset
```

`dataset` should contain the canonical dataset key so we can evaluate per dataset later.

## Training Tasks

Each row can produce three task variants:

1. Text-to-SVG
   - input: `text`
   - target: `svg`

2. Image-to-SVG
   - input: rendered image from `svg`
   - target: `svg`

3. Image + corrupted SVG repair
   - input: rendered image from `svg` plus a corrupted version of `svg`
   - target: original clean `svg`

Use the same train/validation/test split for all task variants. Do not allow one row's text-to-SVG example in train and image-to-SVG or repair example in validation/test.

The first pass should be text-to-SVG only. Repair belongs in the image-to-SVG phase because it needs visual grounding: the model sees the rendered image, compares it against the corrupted SVG, and predicts the clean SVG.

For repair examples, keep the corruption aligned with the current image curriculum stage. The model should repair examples from the same dataset family it is currently learning, plus replay repair examples from earlier datasets.

Start repair as a small auxiliary task during image-to-SVG training, then increase it once the model has basic visual grounding:

```text
Image Curriculum 1-2: 10% repair
Image Curriculum 3-4: 15% repair
Image Curriculum 5-6: 20% repair
Final image stabilization: 20% repair
```

The remaining image-phase examples should be normal image-to-SVG examples.

## Repair Corruptions

Use the repair objective as SVG denoising:

```text
input: image + corrupted_svg
target: clean_svg
```

Corruption types:

- Missing objects: remove one or more visible SVG elements.
- Wrong z-order: reorder sibling elements so overlap/rendering order is wrong.
- Primitive degradation: replace or simplify richer primitives, for example path-to-basic-shape approximations.
- Text corruption: alter, remove, or mask text content and text attributes.
- Style corruption: perturb fill, stroke, opacity, stroke-width, font, or color values.
- Geometry perturbation: jitter positions, sizes, path coordinates, endpoints, or transforms.
- Group flattening: remove or simplify `<g>` nesting and transforms.
- Truncated SVG: delete trailing characters/tags so the model learns how to close and end SVGs correctly.

Keep most corrupted SVGs close enough to the clean SVG that the task remains repair rather than full regeneration. A useful default is to preserve most of the structure and corrupt one or two localized aspects per example. Increase corruption severity later in the curriculum.

## Training Phases

Use the same dataset curriculum order twice:

1. Text-to-SVG phase
   - input: text
   - target: SVG
   - no images and no repair examples
   - goal: teach SVG syntax, structure, paths, layout, and dataset-specific SVG style

2. Image-to-SVG phase
   - input: rendered image
   - target: SVG
   - include repair as an auxiliary task
   - goal: teach visual grounding and image-conditioned SVG generation

3. Final stabilization
   - mixed image-to-SVG and repair examples
   - optional small text-to-SVG replay if text prompting quality regresses

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

Use different task mixes for the two phases.

Text-to-SVG phase:

```text
100% text-to-SVG
```

Image-to-SVG phase, early stages:

```text
90% image-to-SVG
10% repair
```

Image-to-SVG phase, middle stages:

```text
85% image-to-SVG
15% repair
```

Image-to-SVG phase, later diagram-heavy stages:

```text
80% image-to-SVG
20% repair
```

Final stabilization:

```text
75-80% image-to-SVG
20% repair
0-5% text-to-SVG replay, only if needed
```

## Evaluation

Evaluate separately by dataset family and task:

- text-to-SVG on each dataset
- image-to-SVG on each dataset
- image + corrupted SVG repair on each dataset
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
