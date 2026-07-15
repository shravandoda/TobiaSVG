---
annotations_creators:
- machine-generated
language:
- en
license: other
pretty_name: TobiaSVG Repair
size_categories:
- 10K<n<100K
source_datasets:
- extended|shravandoda/TobiaSVG
task_categories:
- text-generation
tags:
- svg
- vector-graphics
- svg-repair
- multimodal
---

# TobiaSVG Repair

TobiaSVG Repair contains 44,802 synthetic SVG repair pairs. Each row pairs a
corrupted SVG with its clean target. Raster images are rendered when examples
are loaded and are not stored.

## Sources And Splits

| Subset | Source | Rows | License |
| --- | --- | ---: | --- |
| `vfig_diagrams` | [VFIG-Data](https://huggingface.co/datasets/QijiaHe/VFIG-Data) | 33,624 | ODC-BY 1.0 |
| `vfig_shapes` | [VFIG-Data](https://huggingface.co/datasets/QijiaHe/VFIG-Data) | 8,884 | ODC-BY 1.0 |
| `animal_illustrations` | [SVG Animal Illustrations](https://huggingface.co/datasets/yoavf/svg-animal-illustrations) | 2,294 | CC0 1.0 |

Splits contain 35,833 training, 4,423 test, and 4,546 validation rows. Repair
variants remain in the same split as their clean target.

## Fields

- `filename`: original filename.
- `svg`: clean target SVG.
- `corrupted_svg`: corrupted input SVG.
- `dataset`: source subset listed above.

## Processing

Medium and hard corruptions alter objects, z-order, primitives, text, styles,
geometry, or groups. Non-truncated pairs were retained only when render MSE was
at least 0.002 and changed-pixel ratio was at least 0.01. Ten percent of pairs
also received truncation to represent incomplete SVG generation.

## License

This is a mixed-license dataset, and corrupted SVGs retain the obligations of
their clean source. Animal illustrations are CC0 1.0. VFIG is ODC-BY 1.0,
which licenses the database but may not cover rights in every individual item;
review the [ODC-BY terms](https://opendatacommons.org/licenses/by/1-0/) and
source card.
