"""An example illustrating parser-based masking."""
import math
import time
import urllib.request

import torch
from lark import Lark
from lark.indenter import DedentError
from lark.lexer import UnexpectedCharacters, UnexpectedToken
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
    set_seed,
)

from outlines.text.parsing import PartialPythonIndenter, copy_parser_state, parse_to_end

revision = None
checkpoint = "Salesforce/codegen-350M-mono"
device = "cuda"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)

model = AutoModelForCausalLM.from_pretrained(
    checkpoint, trust_remote_code=True, revision=revision
).to(device)

sql_grammar_url = "https://github.com/zbrookle/sql_to_ibis/raw/0e9226da42065940ce21439d490f9fcacadc7f92/sql_to_ibis/grammar/sql.lark"
sql_grammar = "".join(
    [line.decode("utf-8") for line in urllib.request.urlopen(sql_grammar_url)]
)
with open("sql_grammar.lark", "w") as f:
    f.write(sql_grammar)

sqlparser = Lark.open(
    "sql_grammar.lark",
    parser="lalr",
)

pyparser = Lark.open_from_package(
    "lark",
    "python.lark",
    ["grammars"],
    parser="lalr",
    postlex=PartialPythonIndenter(),
    start="file_input",
)


class ParserLogitsProcessor(LogitsProcessor):
    """Bias invalid token scores according to a running parse state."""

    def __init__(self, parser):
        ip = parser.parse_interactive("")
        self.parser_state = ip.parser_state
        self.states_stack = [self.parser_state]
        self.token_seq = None
        self.token_idx = 0

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        if self.token_seq is None:
            self.token_seq = tokenizer.decode(input_ids[0])
            self.token_idx = len(input_ids[0]) - 1
        else:
            self.token_idx += 1
            self.token_seq += tokenizer.decode(input_ids[0][self.token_idx])

        # Process the last sampled token
        lex_state = self.parser_state.lexer.state
        lex_state.text = self.token_seq

        self.parser_state, partial_tokens = parse_to_end(self.parser_state)

        print(f'parsed:"{self.token_seq}"')
        print(f"partial_tokens: {partial_tokens}")

        mask = torch.full_like(scores, -math.inf)

        # Determine which tokens in the vocabulary are valid next tokens
        # given the parser state.
        #
        # TODO: This is a very naive and slow approach.  It could be done in
        # parallel, but there are a few other approaches to try first, and
        # those should dramatically reduce the amount of work done here.
        t0 = time.perf_counter()
        for test_token, token_id in tokenizer.vocab.items():
            ps = copy_parser_state(self.parser_state)
            ls = ps.lexer.state
            ls.text = self.token_seq + test_token

            try:
                # TODO: The resulting states could possibly be reused?
                parse_to_end(ps)
                mask[0][token_id] = 0
            except (UnexpectedToken, UnexpectedCharacters, DedentError):
                pass

        print(f"next token masking duration: {time.perf_counter() - t0}")

        return scores + mask


set_seed(20399)

parser = sqlparser
input_text = "select "
inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)

outputs = model.generate(
    inputs,
    max_length=100,
    temperature=0.1,
    logits_processor=LogitsProcessorList([ParserLogitsProcessor(parser)]),
    renormalize_logits=True,
)

print(tokenizer.decode(outputs[0]))
