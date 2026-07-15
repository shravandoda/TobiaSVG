---
annotations_creators:
- machine-generated
- found
language:
- en
license: other
pretty_name: TobiaSVG
size_categories:
- 100K<n<1M
source_datasets:
- extended|QijiaHe/VFIG-Data
- extended|yoavf/svg-animal-illustrations
- extended|xingxm/SVGX-Core-250k
task_categories:
- text-generation
tags:
- svg
- vector-graphics
- text-to-svg
- image-to-svg
---

# TobiaSVG

TobiaSVG contains 106,524 text-SVG pairs for text-to-SVG generation and
image-to-SVG vectorization. Raster images are not stored; SVGs can be rendered
when examples are loaded.

## Sources And Splits

| Subset | Source | Rows | License |
| --- | --- | ---: | --- |
| `vfig_diagrams` | [VFIG-Data](https://huggingface.co/datasets/QijiaHe/VFIG-Data) | 59,276 | ODC-BY 1.0 |
| `vfig_shapes` | [VFIG-Data](https://huggingface.co/datasets/QijiaHe/VFIG-Data) | 5,832 | ODC-BY 1.0 |
| `svgx_core` | [SVGX-Core-250k](https://huggingface.co/datasets/xingxm/SVGX-Core-250k) | 40,000 | CC BY-NC 4.0 |
| `animal_illustrations` | [SVG Animal Illustrations](https://huggingface.co/datasets/yoavf/svg-animal-illustrations) | 1,416 | CC0 1.0 |

Splits contain 85,170 training, 10,599 test, and 10,755 validation rows.
Identical clean SVGs are kept in the same split.

## Fields

- `filename`: original filename.
- `svg`: clean SVG source.
- `text`: text description or prompt.
- `dataset`: source subset listed above.

## Processing

SVGs were normalized, parsed, and rendered; invalid rows were removed. VFIG
rows additionally required at least 40% structural primitives and at most 50
complex shapes. Existing animal prompts and SVGX Qwen captions were retained.
VFIG captions were generated from rendered images with
`gemini-3.1-flash-lite`. Rows were sampled with seed 42 and split by clean SVG
identity to prevent target leakage.

## License

This is a mixed-license dataset. The `dataset` field maps each row to the source
table above. SVGX rows are CC BY-NC 4.0, so the complete dataset is for
noncommercial use. Animal illustrations are CC0 1.0. VFIG is ODC-BY 1.0, which
licenses the database but may not cover rights in every individual item; review
the [ODC-BY terms](https://opendatacommons.org/licenses/by/1-0/) and source card.
