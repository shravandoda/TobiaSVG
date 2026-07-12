import pytest
import torch
from datasets import Dataset, DatasetDict

from project_x.eval import evaluate
from project_x.eval.metrics import is_valid_svg
from project_x.eval.render import SvgExtractionError, extract_svg, render_svg


SIMPLE_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="12">'
    '<rect width="16" height="12" fill="red"/>'
    "</svg>"
)


class FakeBatch(dict):
    def to(self, device):
        assert device == torch.device("cpu")
        return self


class FakeProcessor:
    def __call__(self, **kwargs):
        assert kwargs["text"] == ["serialized prompt"]
        return FakeBatch(input_ids=torch.tensor([[1, 2, 3]]))

    def batch_decode(self, generated_ids, *, skip_special_tokens):
        assert generated_ids.tolist() == [[4, 5]]
        assert skip_special_tokens is True
        return [SIMPLE_SVG]


class FakeModel:
    device = torch.device("cpu")

    def generate(self, *, input_ids, max_new_tokens, do_sample):
        assert input_ids.tolist() == [[1, 2, 3]]
        assert max_new_tokens == 10
        assert do_sample is False
        return torch.tensor([[1, 2, 3, 4, 5]])


def test_extract_svg_ignores_surrounding_text():
    generated = f"Here is the result:\n```xml\n{SIMPLE_SVG}\n```"

    assert extract_svg(generated) == SIMPLE_SVG


def test_extract_svg_requires_complete_document():
    with pytest.raises(SvgExtractionError):
        extract_svg("<svg><rect/></svg")


def test_is_valid_svg_checks_xml_and_root_element():
    assert is_valid_svg(SIMPLE_SVG)
    assert not is_valid_svg("<html></html>")
    assert not is_valid_svg("<svg>")


def test_render_svg_writes_png(tmp_path):
    output_path = tmp_path / "render.png"

    render_svg(SIMPLE_SVG, output_path)

    assert output_path.read_bytes().startswith(b"\x89PNG")


def test_load_evaluation_rows_returns_raw_dataset_rows(monkeypatch):
    validation = Dataset.from_dict(
        {
            "filename": ["first.svg", "second.svg"],
            "svg": [SIMPLE_SVG, SIMPLE_SVG],
            "text": ["first", "second"],
        }
    )
    dataset = DatasetDict({"val": validation})
    monkeypatch.setattr(evaluate, "get_tobias_dataset", lambda: dataset)

    rows = evaluate.load_evaluation_rows("text", "val", num_examples=1)

    assert len(rows) == 1
    assert rows.column_names == ["filename", "svg", "text"]


def test_generate_svg_returns_decoded_completion_string(monkeypatch):
    monkeypatch.setattr(evaluate, "get_text2svg_prompt", lambda text: [text])
    monkeypatch.setattr(
        evaluate,
        "serialize_prompt",
        lambda processor, prompt: "serialized prompt",
    )

    generated = evaluate.generate_svg(
        FakeModel(),
        FakeProcessor(),
        {"text": "draw a square"},
        task="text",
        max_new_tokens=10,
    )

    assert generated == SIMPLE_SVG
