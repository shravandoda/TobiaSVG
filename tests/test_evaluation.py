import pytest
import torch
from datasets import Dataset, DatasetDict
from PIL import Image

from project_x.eval import evaluate
from project_x.eval.metrics import is_valid_svg, normalized_pixel_mse
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
    tokenizer = object()

    def __call__(self, **kwargs):
        assert kwargs["text"] == ["serialized prompt"]
        return FakeBatch(input_ids=torch.tensor([[1, 2, 3]]))

    def batch_decode(self, generated_ids, *, skip_special_tokens):
        assert generated_ids.tolist() == [[4, 5]]
        assert skip_special_tokens is True
        return [SIMPLE_SVG]


class FakeModel:
    device = torch.device("cpu")

    def generate(
        self,
        *,
        input_ids,
        max_new_tokens,
        do_sample,
        logits_processor,
        stop_strings,
        tokenizer,
    ):
        assert input_ids.tolist() == [[1, 2, 3]]
        assert max_new_tokens == 10
        assert do_sample is False
        assert len(logits_processor) == 1
        assert logits_processor[0].penalty == 1.1
        assert logits_processor[0].prompt_ignore_length == 3
        assert stop_strings == ["</svg>"]
        assert tokenizer is FakeProcessor.tokenizer
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


def test_load_evaluation_rows_filters_test_rows_by_task_length(monkeypatch):
    test = Dataset.from_dict(
        {
            "filename": ["first.svg", "second.svg"],
            "svg": [SIMPLE_SVG, SIMPLE_SVG],
            "text": ["first", "second"],
            "measured_length": [100, 13_000],
        }
    )
    dataset = DatasetDict({"test": test})
    monkeypatch.setattr(evaluate, "get_tobias_dataset", lambda: dataset)
    monkeypatch.setitem(
        evaluate.SEQUENCE_LENGTH_FUNCTIONS,
        "text",
        lambda row: row["measured_length"],
    )

    rows = evaluate.load_evaluation_rows("text", num_examples=2)

    assert len(rows) == 1
    assert rows[0]["filename"] == "first.svg"


def test_normalized_pixel_mse(tmp_path):
    target_path = tmp_path / "target.png"
    identical_path = tmp_path / "identical.png"
    different_path = tmp_path / "different.png"
    Image.new("RGBA", (2, 2), "red").save(target_path)
    Image.new("RGBA", (2, 2), "red").save(identical_path)
    Image.new("RGBA", (2, 2), "blue").save(different_path)

    assert normalized_pixel_mse(target_path, identical_path) == 0.0
    assert normalized_pixel_mse(target_path, different_path) == 0.5


def test_summarize_results_aggregates_render_metrics():
    summary = evaluate.summarize_results(
        [
            {
                "prediction_has_svg": True,
                "prediction_is_valid_svg": True,
                "prediction_rendered": True,
                "pixel_mse": 0.1,
            },
            {
                "prediction_has_svg": False,
                "prediction_is_valid_svg": False,
                "prediction_rendered": False,
            },
        ]
    )

    assert summary == {
        "num_examples": 2,
        "svg_extraction_rate": 0.5,
        "valid_svg_rate": 0.5,
        "render_success_rate": 0.5,
        "pixel_mse_examples": 1,
        "mean_pixel_mse": 0.1,
    }


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
        repetition_penalty=1.1,
    )

    assert generated == SIMPLE_SVG
