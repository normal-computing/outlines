import re

import pytest
import torch

import outlines.models as models
from outlines.text.generate.continuation import continuation
from outlines.text.generate.regex import integer, regex


def test_transformers_integration_continuation():
    rng = torch.Generator()
    rng.manual_seed(10000)  # Choosen so <EOS> is generated

    model_name = "hf-internal-testing/tiny-random-GPTJForCausalLM"
    model = models.transformers(model_name, device="cpu")
    sequence = continuation(model)("Write a short sentence", rng=rng)
    assert isinstance(sequence, str)
    assert model.tokenizer.eos_token not in sequence

    sequence = continuation(model, max_tokens=10)("Write a short sentence", rng=rng)
    assert isinstance(sequence, str)

    prompts = ["Write a short sentence", "And another one"]
    sequence = continuation(model, max_tokens=10)(prompts, rng=rng)
    assert isinstance(sequence, list)
    assert len(sequence) == 2
    assert isinstance(sequence[0], str)


@pytest.mark.xfail
def test_transformers_integration_continuation_array_samples():
    rng = torch.Generator()
    rng.manual_seed(0)

    model_name = "hf-internal-testing/tiny-random-GPTJForCausalLM"
    model = models.transformers(model_name, device="cpu")
    prompts = ["Write a short sentence", "And another one"]
    _ = continuation(model, max_tokens=10)(prompts, rng=rng, samples=3)


def test_transformers_integration_integer():
    rng = torch.Generator()
    rng.manual_seed(0)

    model_name = "hf-internal-testing/tiny-random-GPTJForCausalLM"
    model = models.transformers(model_name, device="cpu")
    prompt = "Write a short sentence"
    sequence = integer(model, max_tokens=10)(prompt, rng=rng)

    generated = sequence[len(prompt) :]
    assert generated[0] != 0
    int(generated)


def test_transformers_various_regexes():
    rng = torch.Generator()
    rng.manual_seed(0)

    model_name = "hf-internal-testing/tiny-random-GPTJForCausalLM"
    model = models.transformers(model_name, device="cpu")
    prompt = "Write an email address"
    regex_str = r"([a-z]{10})@([a-z]{5})\.([a-z]{3})"
    sequence = regex(model, regex_str)(prompt, rng=rng)
    assert re.fullmatch(regex_str, sequence[len(prompt) :]) is not None


def test_transformers_integration_integer_array():
    rng = torch.Generator()
    rng.manual_seed(0)

    model_name = "hf-internal-testing/tiny-random-GPTJForCausalLM"
    model = models.transformers(model_name, device="cpu")
    prompts = ["Give me a number", "And another one"]
    sequence = integer(model, max_tokens=10)(prompts, rng=rng)
    assert isinstance(sequence, list)
    assert len(sequence) == 2
    int(sequence[0][len(prompts[0]) :])
    int(sequence[1][len(prompts[1]) :])


def test_transformers_integration_with_pad_token():
    model_name = "hf-internal-testing/tiny-random-XLMRobertaXLForCausalLM"
    model = models.transformers(model_name, device="cpu")
    assert model.tokenizer.pad_token_id == 1
    assert model.tokenizer.pad_token == "<pad>"
