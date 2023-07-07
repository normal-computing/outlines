import math

import torch

from outlines.text.generate.regex import integer


class Tokenizer:
    eos_token = "<EOS>"
    pad_token = None
    eos_token_id = 0
    pad_token_id = -1
    vocabulary = {"<EOS>": 0, "00": 1, "1": 2, "0.": 3, "431": 4, "a": 5}
    tokens = list(vocabulary.keys())

    def decode(self, token_ids):
        decoded = []
        for i in range(token_ids.shape[0]):
            decoded.append("".join([self.tokens[idx] for idx in token_ids[i]]))

        return decoded


class Model:
    tokenizer = Tokenizer()
    device = "cpu"


def test_integer_proposal():
    model = Model()
    generator = integer(model)

    logits = torch.ones(len(model.tokenizer.vocabulary))
    result = generator.create_proposal(torch.tensor([[]]), logits)
    assert torch.equal(
        result, torch.tensor([[-math.inf, -math.inf, 1.0, -math.inf, 1.0, -math.inf]])
    )

    logits = torch.ones(len(model.tokenizer.vocabulary))
    result = generator.create_proposal(torch.tensor([[2]]), logits)
    assert torch.equal(
        result, torch.tensor([[-math.inf, 1.0, 1.0, -math.inf, 1.0, -math.inf]])
    )

    logits = torch.ones(len(model.tokenizer.vocabulary))
    result = generator.create_proposal(torch.tensor([[4]]), logits)
    assert torch.equal(
        result, torch.tensor([[1.0, 1.0, 1.0, -math.inf, 1.0, -math.inf]])
    )

    logits = torch.ones(len(model.tokenizer.vocabulary))
    result = generator.create_proposal(torch.tensor([[4], [2]]), logits)
    assert torch.equal(
        result,
        torch.tensor(
            [
                [1.0, 1.0, 1.0, -math.inf, 1.0, -math.inf],
                [-math.inf, 1.0, 1.0, -math.inf, 1.0, -math.inf],
            ]
        ),
    )

    logits = torch.ones((4, len(model.tokenizer.vocabulary)))
    result = generator.create_proposal(torch.tensor([[]]), logits)
    assert torch.equal(
        result,
        torch.tile(
            torch.tensor([[-math.inf, -math.inf, 1.0, -math.inf, 1.0, -math.inf]]),
            (4, 1),
        ),
    )

    logits = torch.ones(len(model.tokenizer.vocabulary))
    result = generator.create_proposal(torch.tensor([[4, 0], [2, 2]]), logits)
    assert torch.equal(
        result,
        torch.tensor(
            [
                [1.0, -math.inf, -math.inf, -math.inf, -math.inf, -math.inf],
                [1.0, 1.0, 1.0, -math.inf, 1.0, -math.inf],
            ]
        ),
    )
