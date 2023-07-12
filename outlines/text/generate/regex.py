import math
from typing import List, Optional, Tuple

import interegular
import torch

from outlines.text.generate.continuation import Continuation
from outlines.text.parsing import find_partial_matches, map_partial_states_to_vocab


class Regex(Continuation):
    """Represents a regex-based generation model.

    `Regex` instances are constrained generation models that only generate
    sequences that match an input regex. We assume that the sequence can be
    terminated (but not necessarily) when the finite state machine corresponding
    to the regex is in an accepting state.

    >>> import outlines.text as text
    >>> sequence = text.generate.regex(model, "(0|[1-9][0-9]+)")("Return an integer between 0 and 10")

    """

    def __init__(self, model, regex_string: str, max_tokens: Optional[int]):
        super().__init__(model, max_tokens)

        vocabulary = model.tokenizer.vocabulary
        sorted_vocabulary = [
            k for k, v in sorted(vocabulary.items(), key=lambda kv: kv[1])
        ]

        regex_pattern = interegular.parse_pattern(regex_string)
        self.regex_fsm = regex_pattern.simplify().to_fsm()

        def partial_match_filter(string, end_idx, state_seq):
            if end_idx is not None and end_idx < len(string) - 1:
                return False
            return True

        pstate_to_vocab = map_partial_states_to_vocab(
            list(sorted_vocabulary),
            {"REGEX": self.regex_fsm},
            True,
            partial_match_filter,
            final_state_string=model.tokenizer.eos_token,
        )
        self.pstate_to_vocab = {k: list(v) for k, v in pstate_to_vocab.items()}
        self.eos_idx = sorted_vocabulary.index(model.tokenizer.eos_token)
        self.pad_idx = (
            sorted_vocabulary.index(model.tokenizer.pad_token)
            if getattr(model.tokenizer, "pad_token", None) is not None
            else self.eos_idx
        )
        self.pstates: List[Optional[Tuple[str, int]]] = []

    def create_proposal(
        self, generated_token_ids: torch.LongTensor, logits: torch.DoubleTensor
    ) -> torch.DoubleTensor:
        """Modify the next-token logits so that only integers can be generated.

        Parameters
        ----------
        generated_token_ids
            The token ids generated so far.
        logits
            The next-token logits.

        """
        if generated_token_ids.shape[-1] > 0:
            self.pstates = []
            for token_seq in generated_token_ids:
                if token_seq[-1] != self.eos_idx:
                    sequence = self.model.tokenizer.decode(token_seq[..., None])
                    pmatches = find_partial_matches(
                        self.regex_fsm,
                        "".join(sequence),
                        start_state=self.regex_fsm.initial,
                    )
                    pstate = max(
                        pmatches, key=lambda x: x[0] if x[0] is not None else -1
                    )
                    pstate = ("REGEX", pstate[1][-1])
                else:
                    pstate = None

                self.pstates.append(pstate)
        else:
            self.pstates = [
                ("REGEX", self.regex_fsm.initial)
                for _ in range(generated_token_ids.shape[0])
            ]

        masks = []
        for pstate in self.pstates:
            mask = torch.full((len(self.model.tokenizer.vocabulary),), -math.inf)

            if pstate is not None:
                next_support = self.pstate_to_vocab[pstate]
            else:
                next_support = [self.pad_idx]

            mask[next_support] = 0
            masks.append(mask.unsqueeze(0))

        mask = torch.concatenate(masks, dim=0)

        return logits + mask


def regex(model, regex_string: str, max_tokens: Optional[int] = None):
    return Regex(model, regex_string, max_tokens)


def integer(model, max_tokens: Optional[int] = None):
    return Regex(model, r"(0|[1-9][0-9]+)", max_tokens)
