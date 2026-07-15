import json

from PIL import Image

from project_x.eval.external import load_examples


def test_load_examples_uses_only_image_intersection(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    source_dir = tmp_path / "source"
    manifest_path.write_text(
        json.dumps(
            {
                "tasks": {
                    "text": [{"evaluation_index": 1, "filename": "text.svg"}],
                    "image": [
                        {"evaluation_index": 3, "filename": "first.svg"},
                        {"evaluation_index": 8, "filename": "second.svg"},
                    ],
                }
            }
        )
    )

    for index in (3, 8):
        example_dir = source_dir / f"{index:04d}"
        example_dir.mkdir(parents=True)
        Image.new("RGB", (2, 2), "white").save(example_dir / "target.png")
        (example_dir / "target.svg").write_text("<svg></svg>")

    examples = load_examples(manifest_path, source_dir)

    assert [example.evaluation_index for example in examples] == [3, 8]
    assert [example.filename for example in examples] == ["first.svg", "second.svg"]
    assert [example.output_index for example in examples] == [0, 1]
