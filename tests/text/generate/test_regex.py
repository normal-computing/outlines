import math

import pytest
import torch

from outlines.text.generate.regex import integer, regex


class Tokenizer:
    eos_token = "<EOS>"
    pad_token = None
    eos_token_id = 0
    pad_token_id = -1
    vocabulary = {"<EOS>": 0, "00": 1, "1": 2, "0.": 3, "431": 4, "a": 5, "A": 6}
    tokens = list(vocabulary.keys())

    def decode(self, token_ids):
        decoded = []
        for i in range(token_ids.shape[0]):
            decoded.append("".join([self.tokens[idx] for idx in token_ids[i]]))

        return decoded


class Model:
    tokenizer = Tokenizer()
    device = "cpu"


@pytest.mark.parametrize(
    "regex_string, valid_first_token, proposal",
    [
        (
            r"[A-Z]+",
            6,
            [-math.inf, -math.inf, -math.inf, -math.inf, -math.inf, -math.inf, 1.0],
        ),
        (
            r"[a-z]+",
            5,
            [-math.inf, -math.inf, -math.inf, -math.inf, -math.inf, 1.0, -math.inf],
        ),
        (
            r"(a|A)",
            6,
            [-math.inf, -math.inf, -math.inf, -math.inf, -math.inf, 1.0, 1.0],
        ),
        (r"\d+", 1, [-math.inf, 1.0, 1.0, -math.inf, 1.0, -math.inf, -math.inf]),
        (r"\d+\.", 3, [-math.inf, 1.0, 1.0, 1.0, 1.0, -math.inf, -math.inf]),
    ],
)
def test_regex_proposal(regex_string, valid_first_token, proposal):
    model = Model()
    generator = regex(model, regex_string)

    logits = torch.ones(len(model.tokenizer.vocabulary))
    result = generator.create_proposal(torch.tensor([[]]), logits)
    assert torch.equal(result.squeeze(), torch.tensor(proposal))
    assert result.squeeze()[0] == -math.inf

    # The EOS token can be generated once the FSM is in an accept state
    result = generator.create_proposal(torch.tensor([[valid_first_token]]), logits)
    assert result.squeeze()[0] == 1


@pytest.mark.parametrize(
    "input_ids, proposal",
    [
        ([[]], [[-math.inf, -math.inf, 1.0, -math.inf, 1.0, -math.inf, -math.inf]]),
        ([[2]], [[-math.inf, 1.0, 1.0, -math.inf, 1.0, -math.inf, -math.inf]]),
        ([[4]], [[1.0, 1.0, 1.0, -math.inf, 1.0, -math.inf, -math.inf]]),
        (
            [[4], [2]],
            [
                [1.0, 1.0, 1.0, -math.inf, 1.0, -math.inf, -math.inf],
                [-math.inf, 1.0, 1.0, -math.inf, 1.0, -math.inf, -math.inf],
            ],
        ),
        (
            [[4, 0], [2, 2]],
            [
                [1.0, -math.inf, -math.inf, -math.inf, -math.inf, -math.inf, -math.inf],
                [1.0, 1.0, 1.0, -math.inf, 1.0, -math.inf, -math.inf],
            ],
        ),
    ],
)
def test_integer_proposal(input_ids, proposal):
    model = Model()
    generator = integer(model)

    logits = torch.ones(len(model.tokenizer.vocabulary))
    result = generator.create_proposal(torch.tensor(input_ids), logits)
    assert torch.equal(
        result,
        torch.tensor(proposal),
    )
