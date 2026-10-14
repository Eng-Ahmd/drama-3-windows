"""Verify bounded generation, history conditioning and completion reporting."""
from types import SimpleNamespace

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
