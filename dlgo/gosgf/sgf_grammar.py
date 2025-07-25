import re
import string


_propident_re = re.compile(r"\A[A-Z]{1,8}\Z".encode('ascii'))
_propvalue_re = re.compile(r"\A [^\\\]]* (?: \\. [^\\\]]* )* \Z".encode('ascii'), re.VERBOSE | re.DOTALL)
_find_start_re = re.compile(r"\(\s*;".encode('ascii'))
_tokenise_re = re.compile(r"""
\s*
(?:
    \[ (?P<V> [^\\\]]* (?: \\. [^\\\]]* )* ) \]   # PropValue
    |
    (?P<I> [A-Z]{1,8} )                           # PropIdent
    |
    (?P<D> [;()] )                                # delimiter
)
""".encode('ascii'), re.VERBOSE | re.DOTALL)


def is_valid_property_identifier(s):
    """Check whether 's' is a well-formed PropIdent.

    s -- 8-bit string

    This accepts the same values as the tokeniser.

    Details:
     - it doesn't permit lower-case letters (these are allow in some ancient SGF variants)
     - it accepts at most 8 letters (there is no limit in the spec; no standard property has more than 2)

    """
    return bool(_propident_re.search(s))

def is_valid_property_value(s):
    """Check whether 's' is a well-formed PropValue.

    s -- 8-bit string

    This accepts the same values as the tokeniser: any string that doesn't contain an unescaped ] or end with and unescaped \ .

    """
    return bool(_propvalue_re.search(s))

def tokenise(s, start_position=0):
    """Tokenise a string containing SGF data.

    s               -- 8-bit string
    start_position  -- index into 's'

    Skips leading junk.

    Returns a list of pairs of strings (token type, contents), and also the index in 's' of the start of the unprocessed 'tail'.

    token types and contents:
      I -- PropIdent: upper-case letters
      V -- PropValue: raw value, without the enclosing brackets
      D -- delimiter: ';', '(', or ')'

    Stops when it has seen as many closing parens as open ones, at the end of the string, or when it first finds something it can't tokenise.

    The first two tokens are always '(' and ';' (otherwise it won't find the start of the content).

    """
    result = []
    m = _find_start_re.search(s, start_position)
    if not m:
        return [], 0
    i = m.start()
    depth = 0
    while True:
        m = _tokenise_re.match(s, i)
        if not m:
            break
        group = m.lastgroup
        token = m.group(m.lastindex)
        result.append((group, token))
        i = m.end()
        if group == 'D':
            if token == b'(':
                depth += 1
            elif token == b')':
                depth -= 1
                if depth == 0:
                    break
    return result, i


class Coarse_game_tree:
    """An SGF GameTree.

    This is a direct representation of the SGF parse tree. It's 'coarse' in the sense that the objects in the tree structure represent node sequences, not individual nodes.

    Public attributes
      sequence -- nonempty list of property maps
      children -- list of Coarse_game_trees

    This sequence represents the nodes before the variations.

    """
    def __init__(self):
        self.sequence = [] # must be at least one node
        self.children = [] # may be empty


def _parse_sgf_game(s, start_position):
    """Common implementation for parse_sgf_game and parse_sgf_games."""
    tokens, end_position = tokenise(s, start_position)
    if not tokens:
        return None, None
    stack = []
    game_tree = None
    sequence = None
    properties = None
    index = 0
    try:
        while True:
            token_type, token = tokens[index]
            index += 1
            if token_type == 'V':
                raise ValueError("unexpected value")
            if token_type == 'D':
                if token == b';':
                    if sequence is None:
                        raise ValueError("unexpected node")
                    properties = {}
                    sequence.append(properties)
                else:
                    if sequence is not None:
                        if not sequence:
                            raise ValueError("empty sequence")
                        game_tree.sequence = sequence
                        sequence = None
                    if token == b'(':
                        stack.append(game_tree)
                        game_tree = Coarse_game_tree()
                        sequence = []
                    else:
                        # token == ')'
                        variation = game_tree
                        game_tree = stack.pop()
                        if game_tree is None:
                            break
                        game_tree.children.append(variation)
                    properties = None
            else:
                # token_type = 'I'
                prop_ident = token
                prop_values = []
                while True:
                    token_type, token = tokens[index]
                    if token_type != 'V':
                        break
                    index += 1
                    prop_values.append(token)
                if not prop_values:
                    raise ValueError("property with no values")
                try:
                    if prop_ident in properties:
                        properties[prop_ident] += prop_values
                    else:
                        properties[prop_ident] = prop_values
                except TypeError:
                    raise ValueError("property value outside a node")
    except IndexError:
        raise ValueError("unexpected end of SGF data")
    assert index == len(tokens)
    return variation, end_position


def parse_sgf_game(s):
    """Read a single SGF game from a string, returning the parse tree.

    s -- 8-bit string

    Returns a Coarse_game_tree.

    Applies the rules for FF[4].

    Raises ValueError if can't parse the string.

    If a property appears more than once in a node (which is not permitted by the spec), treats it the same as a single property with multiple values.

    Identifies the start of the SGF content by looking for '(;' (with possible whitespace between); ignores everything preceding that. Ignores everything following the first game.

    """
    game_tree, _ = _parse_sgf_game(s, 0)
    if game_tree is None:
        raise ValueError("no SGF data found")
    return game_tree


def parse_sgf_collection(s):
    """Read an SGF game collection, returning the parse trees.

    s -- 8-bit string

    Returns a nonempty list of Coarse_game_trees.

    Raises ValueError if no games were found in the string.

    Raises ValueError if there is an error parsing a game. See parse_sgf_game() for details.

    Ignoores non-SGF data before the first game, between games, and after the final game. Indetifies the start of each game in the same way as parse_sgf_game().

    """
    position = 0
    result = []
    while True:
        try:
            game_tree, position = _parse_sgf_game(s, position)
        except ValueError as e:
            raise ValueError("error parsing game %d: %s" % (len(result), e))
        if game_tree is None:
            break
        result.append(game_tree)
    if not result:
        raise ValueError("no SGF data found")
    return result


def block_format(pieces, width=79):
    """Concatenate bytestrings, adding newlines.

    pieces -- iterable of string
    width  -- int (default 79)

    Returns "".join(pieces), with added newlines between pieces as necessary to avoid lines longer than 'width'.

    Leaves newlines inside 'pieces' untouched, and ignores them in its width calculation. If a singe piece is longer than 'width', it will become a single long line in the output.

    """
    lines = []
    line = b""
    for s in pieces:
        if len(line) + len(s) > width:
            lines.append(line)
            line = b""
        line += s
    if line:
        lines.append(line)
    return b"\n".join(lines)


def serialise_game_tree(game_tree, wrap=79):
    """Serialise an SGF game as a string.

    game_tree -- Coarse_game_tree
    wrap      -- int (default 79), or None

    Returns an 8-bit string, ending with a newline.

    If 'wrap' is not None, make some effort to keep output lines no longer than 'wrap'.

    """
    l = []
    to_serialise = [game_tree]
    while to_serialise:
        game_tree = to_serialise.pop()
        if game_tree is None:
            l.append(b")")
            continue
        l.append(b"(")
        for properties in game_tree.sequence:
            l.append(b";")
            # Force FF to the front, largely to work around a Quarry bug which makes it ignore the first few bytes of the file.
            for prop_ident, prop_values in sorted(list(properties.items()), key=lambda pair: (-(pair[0] == b"FF"), pair[0])):
                # Make a single string for each property, to get prettier block_format output.
                m = [prop_ident]
                for value in prop_values:
                    m.append(b"[" + value + b"]")
                l.append(b"".join(m))
        to_serialise.append(None)
        to_serialise.extend(reversed(game_tree.children))
    l.append(b"\n")
    if wrap is None:
        return b"".join(l)
    else:
        return block_format(l, wrap)

def make_tree(game_tree, root, node_builder, node_adder):
    """Construct a node tree from a Coarse_game_tree.

    game_tree    -- Coarse_game_tree
    root         -- node
    node_builder -- function taking parameters (parent node, property map) returning a node
    node_adder   -- function taking a pair (parent node, child node)

    Builds a tree of nodes corresponding to this GameTree, calling node_builder() to make new nodes and node_adder() to add child nodes to their parent.

    Make no further assumptions about the node type.

    """
    to_build = [(root, game_tree, 0)]
    while to_build:
        node, game_tree, index = to_build.pop()
        if index < len(game_tree.sequence) - 1:
            child = node_builder(node, game_tree.sequence[index + 1])
            node_adder(node, child)
            to_build.append((child, game_tree, index + 1))
        else:
            node._children = []
            for child_tree in game_tree.children:
                child = node_builder(node, child_tree.sequence[0])
                node_adder(node, child)
                to_build.append((child, child_tree, 0))


def make_coarse_game_tree(root, get_children, get_properties):
    """Construct a Coarse_game_tree from a node tree.

    root           -- node
    get_children   -- function taking a node, returning a sequence of nodes
    get_properties -- function taking a node, returning a property map

    Returns a Coarse_game_tree.

    Walks the node tree based at 'root' using get_childern(), and uses get_properties() to extract the raw properties.

    Make no further assumptions about the node type.

    Doesn't check that the property maps have well-formed keys and values.

    """
    result = Coarse_game_tree()
    to_serialise = [(result, root)]
    while to_serialise:
        game_tree, node = to_serialise.pop()
        while True:
            game_tree.sequence.append(get_properties(node))
            children = get_children(node)
            if len(children) != 1:
                break
            node = children[0]
        for child in children:
            child_tree = Coarse_game_tree()
            game_tree.children.append(child_tree)
            to_serialise.append((child_tree, child))
    return result


def main_sequence_iter(game_tree):
    """Provide the 'leftmost' complete sequence of a Coarse_game_tree.

    game_tree -- Coarse_game_tree

    Returns an iterable of property maps.

    If the game has no variations, this provides the complete game. Otherwise, it chooses the first variation each time it has a choice.

    """
    while True:
        for properties in game_tree.sequence:
            yield properties
        if not game_tree.children:
            break
        game_tree = game_tree.children[0]


_split_compose_re = re.compile(r"( (?: [^\\:] | \\. )* ) :".encode('ascii'), re.VERBOSE | re.DOTALL)

def parse_compose(s):
    """Split the parts of an SGF Compose value.

    If the value is a well-formed Compose, returns a pair of strings.

    If it isn't (ie, there is no delimiter), returns the complete string and None.

    Interprets backslash escapes in order to find the delimiter, but leaves backslash escapes unchanged in the returned strings.

    """
    m = _split_compose_re.match(s)
    if not m:
        return s, None
    return m.group(1), s[m.end():]


def compose(s1, s2):
    """Construct a value of Compose value type.

    s1, s2 -- serialised form of a property value

    (This is only needed if the type of the first value permits colons.)

    """
    return s1.replace(b":", b"\\:") + b":" + s2


_newline_re = re.compile(r"\n\r|\r\n|\n|\r".encode('ascii'))
_binary_maketrans = bytes.maketrans
_whitespace_table = _binary_maketrans(b"\t\f\v", b"   ")
_chunk_re = re.compile(r" [^\n\\]+ | [\n\\] ".encode('ascii'), re.VERBOSE)


def simpletext_value(s):
    """Convert a raw SimpleText property value to the string it represents.

    Returns an 8-bit string, in the encoding of the original SGF string.

    This interprets escape characters, and does whitespace mapping:

    - backslash followed by linebreak (LF, CR, LFCR, or CRLF) disappears
    - any other line break is replaced by a space
    - any other whitespace character is replaced by a space
    - other backslashes disappear (but double-backslash -> single-backslash)

    """
    s = _newline_re.sub(b"\n", s)
    s = s.translate(_whitespace_table)
    is_escaped = False
    result = []
    for chunk in _chunk_re.findall(s):
        if is_escaped:
            if chunk != b"\n":
                result.append(chunk)
            is_escaped = False
        elif chunk == b"\\":
            is_escaped = True
        elif chunk == b"\n":
            result.append(b" ")
        else:
            result.append(chunk)
    return b"".join(result)


def text_value(s):
    """Construct a raw Text property value to the string it represents.

    Return an 8-bit string, in the encoding of the original SGF string.

    This interprets escape characters, and does whitespace mapping:

    - linebreak (LF, CR, LFCR, or CRLF) is converted to \n
    - any other whitespace character is replaced by a space
    - backslash followed by linebreak disappears
    - other backslashes disappear (but double-backslash -> single-backslash)

    """
    s = _newline_re.sub(b"\n", s)
    s = s.translate(_whitespace_table)
    is_escaped = False
    result = []
    for chunk in _chunk_re.findall(s):
        if is_escaped:
            if chunk != b"\n":
                result.append(chunk)
            is_escaped = False
        elif chunk == b"\\":
            is_escape = True
        else:
            result.append(chunk)
    return b"".join(result)


def escape_text(s):
    """Convert a string to a raw Text property value that represents it.

    s -- 8-bit string, in the desired output encoding.

    Returns an 8-bit string which passes is_valid_property_value().

    Normally text_value(escape_text(s)) == s, but there are the following exceptions:
     - all linebreaks are normalised to \n
     - whitespace other than line breaks is converted to a single space

    """
    return s.replace(b"\\", b"\\\\").replace(b"]", b"\\]")
