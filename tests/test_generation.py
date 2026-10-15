"""Verify bounded generation, history conditioning and completion reporting."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import fish_drama3.generation as generation


def frames(count: int) -> torch.Tensor:
    value = torch.zeros((count, 11), dtype=torch.long)
    value[:, 0] = 248099
    return value


@pytest.mark.parametrize("first_reason,budget,expected,calls", [
    ("eos", 5, "eos", 2), ("length", 5, "length", 1),
    ("eos", 2, "length", 1),
])
def test_history_budget_and_completion(monkeypatch, first_reason, budget, expected, calls):
    prompts, limits = [], []
    monkeypatch.setattr(generation, "split_text", lambda *a: ["<|speaker:1|> Hello.", " Next."])

    def prompt(text, tokenizer, config, texts, codes, history):
        prompts.append((text, history))
        return torch.zeros((3, 11), dtype=torch.long)

    def generate(model, prompt, max_new_tokens, **kwargs):
        limits.append(max_new_tokens)
        return frames(2), first_reason if len(limits) == 1 else "eos"

    monkeypatch.setattr(generation, "build_drama3_prompt", prompt)
    monkeypatch.setattr(generation, "generate_drama3", generate)
    args = SimpleNamespace(text="text", chunk_length=200, max_frames=budget,
                           temperature=.7, top_p=.7, top_k=30)
    model = SimpleNamespace(tokenizer=None, config=None)
    result, reason, segments = generation.generate_text(model, args, [], [])
    assert reason == expected and len(prompts) == calls and len(result) == calls * 2
    assert limits == ([budget, budget - 2] if calls == 2 else [budget])
    if calls == 2:
        assert prompts[1][0].startswith("<|speaker:1|>")
        assert prompts[1][1][0][1].shape == (10, 2)
        assert prompts[1][1][0][1].dtype == torch.int64
        assert segments[1]["text"] == " Next."


@pytest.mark.parametrize("prompt_rows,accepted", [(9728, True), (9729, False)])
def test_context_limit_checked_before_generation(monkeypatch, prompt_rows, accepted):
    prompt = torch.zeros((prompt_rows, 11), dtype=torch.long)
    monkeypatch.setattr(generation, "build_drama3_prompt", Mock(return_value=prompt))
    generate = Mock(return_value=(frames(1), "eos"))
    monkeypatch.setattr(generation, "generate_drama3", generate)
    args = SimpleNamespace(text="Hello.", chunk_length=0, max_frames=512,
                           temperature=.7, top_p=.7, top_k=30, max_context_rows=10240)
    model = SimpleNamespace(tokenizer=None, config=None)
    if accepted:
        result, reason, _ = generation.generate_text(model, args, [], [])
        assert len(result) == 1 and reason == "eos"
        assert generate.call_args.kwargs["max_new_tokens"] == 512
    else:
        with pytest.raises(ValueError, match="10240 context rows"):
            generation.generate_text(model, args, [], [])
        generate.assert_not_called()


def test_context_limit_caps_history_conditioned_segment(monkeypatch):
    monkeypatch.setattr(generation, "split_text", lambda *args: ["Hello.", "Again."])
    prompts = [torch.zeros((3, 11), dtype=torch.long), torch.zeros((10000, 11), dtype=torch.long)]
    monkeypatch.setattr(generation, "build_drama3_prompt", Mock(side_effect=prompts))
    generate = Mock(return_value=(frames(1), "eos"))
    monkeypatch.setattr(generation, "generate_drama3", generate)
    args = SimpleNamespace(text="Hello. Again.", chunk_length=200, max_frames=512,
                           temperature=.7, top_p=.7, top_k=30, max_context_rows=10240)
    model = SimpleNamespace(tokenizer=None, config=None)
    _, reason, segments = generation.generate_text(model, args, [], [])
    assert reason == "eos" and len(segments) == 2
    assert generate.call_args_list[0].kwargs["max_new_tokens"] == 512
    assert generate.call_args_list[1].kwargs["max_new_tokens"] == 240


def test_long_total_budget_is_bounded_per_segment_and_reports_progress(monkeypatch):
    chunks = ["Hello.", " Again."]
    monkeypatch.setattr(generation, "split_text", lambda *args: chunks)
    monkeypatch.setattr(generation, "build_drama3_prompt",
                        Mock(return_value=torch.zeros((100, 11), dtype=torch.long)))
    generate = Mock(return_value=(frames(2), "eos"))
    monkeypatch.setattr(generation, "generate_drama3", generate)
    args = SimpleNamespace(text="".join(chunks), chunk_length=200, max_frames=20000,
                           temperature=.7, top_p=.7, top_k=30, max_context_rows=10240)
    updates = []
    output, reason, _ = generation.generate_text(
        SimpleNamespace(tokenizer=None, config=None), args, [], [], progress=updates.append)
    assert len(output) == 4 and reason == "eos"
    assert [call.kwargs["max_new_tokens"] for call in generate.call_args_list] == [8192, 8192]
    assert updates[2]["frames"] == updates[2]["completed_frames"] == 2
    assert updates[2]["completed_characters"] == len(chunks[0])
    assert updates[3]["frames"] == 4
    assert updates[3]["completed_characters"] == len(args.text)


def test_no_context_capacity_stops_before_model_call(monkeypatch):
    monkeypatch.setattr(generation, "build_drama3_prompt",
                        Mock(return_value=torch.zeros((10240, 11), dtype=torch.long)))
    generate = Mock()
    monkeypatch.setattr(generation, "generate_drama3", generate)
    args = SimpleNamespace(text="Hello.", chunk_length=200, max_frames=20000,
                           temperature=.7, top_p=.7, top_k=30, max_context_rows=10240)
    with pytest.raises(ValueError, match="No speech generation has started"):
        generation.generate_text(SimpleNamespace(tokenizer=None, config=None), args, [], [])
    generate.assert_not_called()
