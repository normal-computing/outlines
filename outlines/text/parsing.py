from collections import ChainMap, defaultdict
from copy import copy
from itertools import chain
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    DefaultDict,
    Dict,
    Iterable,
    Optional,
    Set,
    Tuple,
)

import interegular
import regex
from interegular.fsm import FSM, anything_else
from interegular.patterns import Unsupported
from lark import Lark, Token
from lark.exceptions import (
    LexError,
    UnexpectedCharacters,
    UnexpectedEOF,
    UnexpectedToken,
)
from lark.indenter import PythonIndenter
from lark.lexer import BasicLexer, ContextualLexer, LexerState, Scanner
from lark.parsers.lalr_analysis import Shift
from lark.parsers.lalr_interactive_parser import InteractiveParser
from lark.parsers.lalr_parser import ParseConf, ParserState
from lark.utils import get_regexp_width

if TYPE_CHECKING:
    from lark.lexer import LexerThread


PartialParseState = Tuple[str, int]


class PartialTokenEOF(UnexpectedEOF):
    pass


class PartialScanner(Scanner):
    def __init__(self, scanner: Scanner):
        self.terminals = scanner.terminals
        self.g_regex_flags = scanner.g_regex_flags
        self.re_ = regex
        self.use_bytes = scanner.use_bytes
        self.match_whole = scanner.match_whole
        self.allowed_types = scanner.allowed_types
        self._mres = scanner._mres

    def match(self, text, pos) -> Optional[Tuple[str, Optional[str], bool]]:
        for mre in self._mres:
            m = mre.match(text, pos=pos, partial=True)
            if m:  # and ((not m.partial) or m.endpos == len(text)):
                return m.group(0), m.lastgroup, m.partial
        return None


class PartialBasicLexer(BasicLexer):
    def __init__(self, basic_lexer: BasicLexer):
        self.re = regex
        self.newline_types = basic_lexer.newline_types
        self.ignore_types = basic_lexer.ignore_types
        self.terminals = basic_lexer.terminals
        self.user_callbacks = basic_lexer.user_callbacks
        self.g_regex_flags = basic_lexer.g_regex_flags
        self.use_bytes = basic_lexer.use_bytes
        self.terminals_by_name = basic_lexer.terminals_by_name
        self.callback = getattr(basic_lexer, "callback", None)

        if basic_lexer._scanner is not None:
            self._scanner: Optional[PartialScanner] = PartialScanner(
                basic_lexer._scanner
            )
        else:
            self._scanner = None

        # This is used to determine the token type for partial matches
        self.terminal_to_regex = {}
        for name, terminal in self.terminals_by_name.items():
            self.terminal_to_regex[name] = self.re.compile(
                terminal.pattern.to_regexp(), self.g_regex_flags
            )

    def _build_scanner(self):
        super()._build_scanner()
        self._scanner = PartialScanner(self._scanner)

    def partial_matches(self, value, type_):
        partial_matches = set()

        # TODO: It's unfortunate that we have to do this costly search (again).
        # It would be better if we could *not* short-circuit the first time we
        # scan in the call to `self.match`.
        for term_name, term_regex in self.terminal_to_regex.items():
            if term_name == type_:
                # A standard lexed token result could actual indicate a partial
                # match
                regex_min, regex_max = get_regexp_width(term_regex.pattern)
                if regex_min <= len(value) < regex_max:
                    partial_matches.add(term_name)
            else:
                m = term_regex.match(value, partial=True)
                if m:
                    partial_matches.add(term_name)

        return partial_matches

    def next_token(self, lex_state: LexerState, parser_state: Any = None) -> Token:
        line_ctr = lex_state.line_ctr
        while line_ctr.char_pos < len(lex_state.text):
            res = self.match(lex_state.text, line_ctr.char_pos)

            if not res:
                allowed = self.scanner.allowed_types - self.ignore_types
                if not allowed:
                    allowed = {"<END-OF-FILE>"}
                raise UnexpectedCharacters(
                    lex_state.text,
                    line_ctr.char_pos,
                    line_ctr.line,
                    line_ctr.column,
                    allowed=allowed,
                    token_history=lex_state.last_token and [lex_state.last_token],
                    state=parser_state,
                    terminals_by_name=self.terminals_by_name,
                )

            value, type_, partial = res

            # Don't advance the lexing state if we're at the end; there could
            # be ambiguous token types that aren't finished.
            if line_ctr.char_pos + len(value) >= len(lex_state.text):
                partial_matches = self.partial_matches(value, type_)
                if partial_matches or partial:
                    raise PartialTokenEOF(partial_matches)

            assert isinstance(self.callback, Dict)

            if type_ not in self.ignore_types:
                t = Token(
                    type_, value, line_ctr.char_pos, line_ctr.line, line_ctr.column
                )
                line_ctr.feed(value, type_ in self.newline_types)
                t.end_line = line_ctr.line
                t.end_column = line_ctr.column
                t.end_pos = line_ctr.char_pos
                if t.type in self.callback:
                    t = self.callback[t.type](t)
                    if not isinstance(t, Token):
                        raise LexError(
                            "Callbacks must return a token (returned %r)" % t
                        )
                lex_state.last_token = t
                return t

            if type_ in self.callback:
                t2 = Token(
                    type_, value, line_ctr.char_pos, line_ctr.line, line_ctr.column
                )
                self.callback[type_](t2)

            line_ctr.feed(value, type_ in self.newline_types)

        raise EOFError(self)


class PartialPythonIndenter(PythonIndenter):
    """An `Indenter` that doesn't reset its state every time `process` is called."""

    def process(self, stream):
        return self._process(stream)

    def _process(self, stream):
        for token in stream:
            # These were previously *after* the `yield`, but that makes the
            # state tracking unnecessarily convoluted.
            if token.type in self.OPEN_PAREN_types:
                self.paren_level += 1
            elif token.type in self.CLOSE_PAREN_types:
                self.paren_level -= 1
                if self.paren_level < 0:
                    raise UnexpectedToken(token, [])

            if token.type == self.NL_type:
                yield from self.handle_NL(token)
            else:
                yield token

        # while len(self.indent_level) > 1:
        #     self.indent_level.pop()
        #     yield Token(self.DEDENT_type, "")

    def __copy__(self):
        res = type(self)()
        res.paren_level = self.paren_level
        res.indent_level = copy(self.indent_level)
        return res


def copy_lexer_thread(lexer_thread: "LexerThread") -> "LexerThread":
    res = copy(lexer_thread)
    res.lexer = copy(res.lexer)

    if getattr(res.lexer, "postlexer", None):
        if isinstance(res.lexer.postlexer, PythonIndenter) and not isinstance(
            res.lexer.postlexer, PartialPythonIndenter
        ):
            # Patch these methods so that the post lexer keeps its state
            # XXX: This won't really work in generality.
            postlexer = PartialPythonIndenter()
            postlexer.paren_level = res.lexer.postlexer.paren_level
            postlexer.indent_level = res.lexer.postlexer.indent_level
            res.lexer.postlexer = postlexer
        else:
            res.lexer.postlexer = copy(res.lexer.postlexer)

    # Patch/replace the lexer objects so that they support partial matches
    context_lexer = res.lexer

    if not isinstance(context_lexer, ContextualLexer):
        # XXX: The layouts change with the grammars
        context_lexer = context_lexer.lexer
        assert isinstance(context_lexer, ContextualLexer)

    if not isinstance(context_lexer.root_lexer, PartialBasicLexer):
        context_lexer.root_lexer = PartialBasicLexer(context_lexer.root_lexer)

        basic_lexers = context_lexer.lexers
        for idx, lexer in basic_lexers.items():
            basic_lexers[idx] = PartialBasicLexer(lexer)

    return res


def copy_parser_state(parser_state: ParserState) -> ParserState:
    res = copy(parser_state)
    res.lexer = copy_lexer_thread(res.lexer)

    return res


def copy_ip(ip: "InteractiveParser") -> "InteractiveParser":
    res = copy(ip)
    res.lexer_thread = copy_lexer_thread(res.lexer_thread)
    return res


def parse_to_end(parser_state: ParserState) -> Tuple[ParserState, Set[str]]:
    """Continue parsing from the current parse state and return partial next tokens.

    .. warning::
        The parse state `parser_state` is updated in-place and must be patched
        to work with this function.  Either patch it manually or use
        `copy_parser_state` before calling this.

    """

    expected_next_tokens: Set[str] = set()
    try:
        for token in parser_state.lexer.lex(parser_state):
            parser_state.feed_token(token)
    except PartialTokenEOF as e:
        expected_next_tokens = e.expected

    return parser_state, expected_next_tokens


def find_partial_matches(
    fsm: FSM, input_string: str
) -> Set[Tuple[Optional[int], Tuple[int, ...]]]:
    """Find the states in the finite state machine `fsm` that accept `input_string`.

    Returns
    -------
    A set of tuples corresponding to each valid starting state in the FSM.
    The first element of each tuple contains either ``None`` or an integer
    indicating the position in `input_string` at which the FSM terminated.  The
    second element is a tuple of the states visited during execution of the
    FSM.

    """
    if len(input_string) == 0 or input_string[0] not in fsm.alphabet:
        return set()

    trans_key = fsm.alphabet[input_string[0]]

    # TODO: We could probably memoize this easily (i.e. no need to recompute
    # paths shared by different starting states)
    def _partial_match(trans: int) -> Optional[Tuple[Optional[int], Tuple[int, ...]]]:
        fsm_map = ChainMap({fsm.initial: trans}, fsm.map)
        state = fsm.initial
        accepted_states: Tuple[int, ...] = ()

        for i, symbol in enumerate(input_string):
            if anything_else in fsm.alphabet and symbol not in fsm.alphabet:
                symbol = anything_else

            trans_key = fsm.alphabet[symbol]

            if not (state in fsm_map and trans_key in fsm_map[state]):
                if state in fsm.finals:
                    i -= 1
                    break
                return None

            state = fsm_map[state][trans_key]

            accepted_states += (state,)

        terminated = state in fsm.finals
        if not terminated and state == fsm.initial:
            return None

        return None if not terminated else i, accepted_states

    res = set()
    for s_now, trans in fsm.map.items():
        if trans_key in trans:
            path = _partial_match(trans)
            if path is not None:
                res.add(path)

    return res


def terminals_to_fsms(lp: Lark) -> Dict[str, FSM]:
    """Construct a ``dict`` mapping terminal symbol names to their finite state machines."""

    symbol_names_and_fsms = {}
    for terminal in lp.terminals:
        pattern = interegular.parse_pattern(terminal.pattern.to_regexp())
        # TODO: Use `pyparser.terminals[0].pattern.flags`?
        try:
            fsm = pattern.to_fsm()
        except Unsupported:
            fsm = None

        symbol_names_and_fsms[terminal.name] = fsm

    return symbol_names_and_fsms


def map_partial_states_to_vocab(
    vocabulary: Iterable[str],
    terminals_to_fsms_map: Dict[str, FSM],
    map_to_antecedents: bool = False,
    partial_match_filter: Callable[
        [str, Optional[int], Tuple[int, ...]], bool
    ] = lambda *args: True,
) -> DefaultDict[PartialParseState, Set[int]]:
    """Construct a map from partial parse states to the vocabulary elements that start in those states.

    Parameters
    ----------
    vocabulary
        The vocabulary composed of strings.
    terminals_to_fsms_map
        Terminal symbol names mapped to FSMs, as provided by `terminals_to_fsms`.
    map_to_antecedents
        When ``True``, return a map with keys that are the antecedent partial
        parse states.  In other words, this is a map that can be used to
        determine valid next tokens given a parse state.
    partial_match_filter
        A callable that determines which partial matches to keep.  The first
        argument is the string being match, the rest are the unpacked partial
        match return values of `find_partial_matches`.
    """

    pstate_to_vocab = defaultdict(set)
    for symbol_name, fsm in terminals_to_fsms_map.items():
        for i, vocab_string in enumerate(vocabulary):
            for end_idx, state_seq in find_partial_matches(fsm, vocab_string):
                if partial_match_filter(vocab_string, end_idx, state_seq):
                    pstate_to_vocab[(symbol_name, state_seq[0])].add(i)

    if not map_to_antecedents:
        return pstate_to_vocab

    # Partially parsed states to next/transition states (for the same terminal symbol)
    ts_pstate_to_substates = dict(
        chain.from_iterable(
            [
                ((symbol_name, s), {(symbol_name, v) for v in ts.values()})
                for s, ts in fsm.map.items()
            ]
            for symbol_name, fsm in terminals_to_fsms_map.items()
        )
    )

    # Reverse the map
    # TODO: We could construct this more directly.
    rev_ts_pstate_to_substates = defaultdict(set)
    for pstate, to_pstates in ts_pstate_to_substates.items():
        for to_pstate in to_pstates:
            rev_ts_pstate_to_substates[to_pstate].add(pstate)

    # A version of `pstate_to_vocab` that is keyed on states that *transition to*
    # the original keys of `pstate_to_vocab`.
    _pstate_to_vocab: DefaultDict[PartialParseState, Set[int]] = defaultdict(set)
    for pstate, vocab in pstate_to_vocab.items():
        for next_pstate in rev_ts_pstate_to_substates[pstate]:
            _pstate_to_vocab[next_pstate] |= vocab

    return _pstate_to_vocab


def terminals_to_lalr_states(lp: Lark) -> DefaultDict[str, Set[int]]:
    terminals_to_states = defaultdict(set)
    parse_table = lp.parser.parser.parser.parse_table
    for state, tokens_to_ops in parse_table.states.items():
        for token, op in tokens_to_ops.items():
            if op[0] == Shift:
                # `op[1]` is the state we shift to when `token` is observed
                terminals_to_states[token].add(op[1])

    return terminals_to_states


def create_pmatch_parser_states(
    lp: Lark,
    terminals_to_states: Dict[str, Set[int]],
    term_type: str,
    ptoken: str,
    pmatch: Tuple[int, Tuple[int, ...]],
) -> Tuple[ParserState, ...]:
    parse_table = lp.parser.parser.parser.parse_table

    # TODO: We need to effectively disable the callbacks that build the
    # trees, because we aren't actually parsing a valid state that can, say,
    # be reduced
    def noop(*args, **kwargs):
        pass

    callbacks = {rule: noop for rule, cb in lp._callbacks.items()}
    parse_conf = ParseConf(parse_table, callbacks, lp.options.start[0])
    lexer_thread = lp.parser._make_lexer_thread(ptoken)
    lexer_state = lexer_thread.state
    lexer_state.line_ctr.char_pos = pmatch[0] + 1
    lexer_state.last_token = Token(term_type, "")
    res = tuple(
        ParserState(parse_conf, lexer_thread, [state], None)
        for state in terminals_to_states[term_type]
    )
    return res
