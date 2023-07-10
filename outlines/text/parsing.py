from collections import ChainMap, defaultdict
from copy import copy
from itertools import chain
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    DefaultDict,
    Dict,
    Generator,
    Iterable,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import interegular
import regex
from interegular.fsm import FSM, Alphabet, OblivionError, anything_else
from interegular.patterns import Unsupported
from lark import Lark, Token
from lark.exceptions import (
    LexError,
    UnexpectedCharacters,
    UnexpectedEOF,
    UnexpectedToken,
)
from lark.indenter import PythonIndenter
from lark.lexer import BasicLexer, LexerState, Scanner
from lark.parsers.lalr_analysis import Shift
from lark.parsers.lalr_interactive_parser import InteractiveParser
from lark.parsers.lalr_parser import ParseConf, ParserState

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
        postfix = "$" if self.match_whole else ""

        fsms = []
        for t in self.terminals:
            regex_str = t.pattern.to_regexp() + postfix
            pattern = interegular.parse_pattern(regex_str).simplify()
            _, max_len = pattern.lengths
            fsm = pattern.to_fsm()
            fsms.append(fsm)

        self.fsm, self.fsms_to_transitions = fsm_union(fsms)

    def match(self, text, pos):
        """Get the match end position, terminal type, and final FSM state."""

        res = find_partial_matches(self.fsm, text[pos:], start_state=self.fsm.initial)

        if len(res) == 0:
            return None

        ((lex_end, state_seq),) = res

        (fsm_id, has_transition) = next(
            get_sub_fsms_from_seq(state_seq, self.fsm, self.fsms_to_transitions)
        )
        type_ = self.terminals[fsm_id]

        return lex_end, type_, state_seq[-1] if not has_transition else None


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

    def _build_scanner(self):
        super()._build_scanner()
        self._scanner = PartialScanner(self._scanner)

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

            (
                lex_end,
                type_,
                last_fsm_state,
            ) = res

            value = lex_state.text[line_ctr.char_pos : lex_end + 1]

            # Don't advance the lexing state if we're at the end
            if line_ctr.char_pos + len(value) >= len(lex_state.text):
                if last_fsm_state is not None:
                    raise PartialTokenEOF((type_.name, last_fsm_state))

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

    if (
        res.lexer.postlexer
        and isinstance(res.lexer.postlexer, PythonIndenter)
        and not isinstance(res.lexer.postlexer, PartialPythonIndenter)
    ):
        # Patch these methods so that the post lexer keeps its state
        # XXX: This won't really work in generality.
        postlexer = PartialPythonIndenter()
        postlexer.paren_level = res.lexer.postlexer.paren_level
        postlexer.indent_level = res.lexer.postlexer.indent_level
        res.lexer.postlexer = postlexer

    # Patch/replace the lexer objects so that they support partial matches
    lexer = res.lexer.lexer
    if not isinstance(lexer.root_lexer, PartialBasicLexer):
        lexer.root_lexer = PartialBasicLexer(lexer.root_lexer)

        basic_lexers = res.lexer.lexer.lexers
        for idx, lexer in basic_lexers.items():
            basic_lexers[idx] = PartialBasicLexer(lexer)

    res.lexer.postlexer = copy(res.lexer.postlexer)

    return res


def copy_parser_state(parser_state: ParserState) -> ParserState:
    res = copy(parser_state)
    res.lexer = copy_lexer_thread(res.lexer)

    return res


def copy_ip(ip: "InteractiveParser") -> "InteractiveParser":
    res = copy(ip)
    res.lexer_thread = copy_lexer_thread(res.lexer_thread)
    return res


def parse_to_end(
    parser_state: ParserState,
) -> Tuple[ParserState, Tuple[Optional[str], Optional[int]]]:
    """Continue parsing from the current parse state and return the terminal name and FSM state."""

    parser_state = copy_parser_state(parser_state)

    terminal_name, fsm_state = None, None
    try:
        for token in parser_state.lexer.lex(parser_state):
            parser_state.feed_token(token)
    except PartialTokenEOF as e:
        terminal_name, fsm_state = e.expected

    return parser_state, (terminal_name, fsm_state)


def find_partial_matches(
    fsm: FSM, input_string: str, start_state: Optional[int] = None
) -> Set[Tuple[Optional[int], Tuple[int, ...]]]:
    """Find the states in the finite state machine `fsm` that accept `input_string`.

    This will consider all possible states in the finite state machine (FSM)
    that accept the beginning of `input_string` as starting points, unless a
    specific `start_state` is provided.

    Parameters
    ----------
    fsm
        The finite state machine.
    input_string
        The string for which we generate partial matches.
    start_state
        A single fixed starting state to consider.  For example, if this value
        is set to `fsm.initial`, it attempt to read `input_string` from the
        beginning of the FSM/regular expression.

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
    def _partial_match(
        trans: Dict[int, int]
    ) -> Optional[Tuple[Optional[int], Tuple[int, ...]]]:
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
    transition_maps = (
        fsm.map.values() if start_state is None else [fsm.map[start_state]]
    )
    for trans in transition_maps:
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
    final_state_string: Optional[str] = None,
) -> DefaultDict[PartialParseState, Set[int]]:
    """Construct a map from partial parse states to subsets of `vocabulary`.

    The subsets of `vocabulary` consist of elements that are accepted by--or
    transition to--the corresponding partial parse states.

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
    final_state_string
        A string from `vocabulary` that is to be added to all the final states
        in the FSM.
    """

    final_state_string_idx = None
    # Partial parse states to the subsets of the vocabulary that accept them
    pstate_to_vocab = defaultdict(set)
    for symbol_name, fsm in terminals_to_fsms_map.items():
        for i, vocab_string in enumerate(vocabulary):
            if vocab_string == final_state_string:
                final_state_string_idx = i

            for end_idx, state_seq in find_partial_matches(fsm, vocab_string):
                if partial_match_filter(vocab_string, end_idx, state_seq):
                    pstate_to_vocab[(symbol_name, state_seq[0])].add(i)

    if not map_to_antecedents:
        return pstate_to_vocab

    # Partial parse states to their valid next/transition states
    ts_pstate_to_substates = dict(
        chain.from_iterable(
            [
                ((symbol_name, s), {(symbol_name, v) for v in ts.values()})
                for s, ts in fsm.map.items()
            ]
            for symbol_name, fsm in terminals_to_fsms_map.items()
        )
    )

    # Reverse the state transitions map
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

    if final_state_string_idx is not None:
        # Allow transitions to EOS from all terminals FSM states
        for symbol_name, fsm in terminals_to_fsms_map.items():
            for state in fsm.finals:
                _pstate_to_vocab[(symbol_name, state)].add(final_state_string_idx)

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


def fsm_union(fsms):
    """Construct an FSM representing the union of the FSMs in `fsms`.

    This is an updated version of `interegular.fsm.FSM.union` made to return an
    extra map of component FSMs to the sets of state transitions that
    correspond to them in the new FSM.

    """

    alphabet, new_to_old = Alphabet.union(*[fsm.alphabet for fsm in fsms])

    indexed_fsms = tuple(enumerate(fsms))

    initial = {i: fsm.initial for (i, fsm) in indexed_fsms}

    # dedicated function accepts a "superset" and returns the next "superset"
    # obtained by following this transition in the new FSM
    def follow(current_state, new_transition: int):
        next = {}
        for i, f in indexed_fsms:
            old_transition = new_to_old[i][new_transition]
            if (
                i in current_state
                and current_state[i] in f.map
                and old_transition in f.map[current_state[i]]
            ):
                next[i] = f.map[current_state[i]][old_transition]
        if not next:
            raise OblivionError
        return next

    # This is a dict that maps component FSMs to a running state
    states = [initial]
    finals = set()
    map = {}

    # Map component fsms to their new state-to-state transitions
    fsms_to_transitions = defaultdict(set)

    # iterate over a growing list
    i = 0
    while i < len(states):
        state = states[i]

        # Add to the finals of the aggregate FSM whenever we hit a final in a
        # component FSM
        if any(state.get(j, -1) in fsm.finals for (j, fsm) in indexed_fsms):
            finals.add(i)

        # compute map for this state
        map[i] = {}
        for transition in alphabet.by_transition:
            try:
                next = follow(state, transition)
            except OblivionError:
                # Reached an oblivion state. Don't list it.
                continue
            else:
                try:
                    # TODO: Seems like this could--and should--be avoided
                    j = states.index(next)
                except ValueError:
                    j = len(states)
                    states.append(next)

                map[i][transition] = j

                for fsm_id, fsm_state in next.items():
                    fsms_to_transitions[fsm_id].add((i, j))

        i += 1

    return (
        FSM(
            alphabet=alphabet,
            states=range(len(states)),
            initial=0,
            finals=finals,
            map=map,
            __no_validation__=True,
        ),
        fsms_to_transitions,
    )


def get_sub_fsms_from_seq(
    state_seq: Sequence[int],
    fsm: FSM,
    fsms_to_transitions: Dict[int, Set[Tuple[int, int]]],
) -> Generator[Tuple[int, bool], None, None]:
    """Get the indices of the sub-FSMs in `fsm` along the state sequence `state_seq`.

    Parameters
    ----------
    state_seq
        A state sequence.
    fsm
        A FSM that is the union of sub-FSMs.
    fsms_to_transitions
        A map from FSM indices to sets of their state transitions.

    Returns
    -------
    A generator returning tuples containing each sub-FSM index (in the order
    they were union-ed to construct `fsm`) and booleans indicating whether or
    not there is another valid transition from the last state in the sequence
    for the associated sub-FSM (i.e. if the FSM can continue
    accepting/matching).
    """
    pmatch_transitions = set(zip((fsm.initial,) + tuple(state_seq[:-1]), state_seq))
    last_fsm_state = state_seq[-1]
    yield from (
        (
            # The sub-FMS index
            fsm_idx,
            # Is there another possible transition in this sub-FSM?
            any(last_fsm_state == from_s for (from_s, to_s) in transitions),
        )
        for fsm_idx, transitions in fsms_to_transitions.items()
        if pmatch_transitions.issubset(transitions)
    )
