"""Lexical scans of C and Python sources shared by the repository checks."""
import io
import re
import tokenize


def _segments(text):
    """Yield (kind, start, end) spans: comment, literal or code."""
    i, n, code = 0, len(text), 0
    while i < n:
        if text.startswith('//', i) or text.startswith('/*', i):
            if text[i + 1] == '/':
                end = text.find('\n', i)
            else:
                end = text.find('*/', i + 2)
                end = end + 2 if end >= 0 else -1
            end = n if end < 0 else end
        elif text[i] in '"\'':
            end = i + 1
            while end < n and text[end] not in (text[i], '\n'):
                end += 2 if text[end] == '\\' else 1
            end = min(end + 1, n)
        else:
            i += 1
            continue
        if code < i:
            yield 'code', code, i
        yield ('comment' if text[i] == '/' else 'literal'), i, end
        i = code = end
    if code < n:
        yield 'code', code, n


def c_comments(text):
    """Return (first_line, last_line, body) for every C comment."""
    comments = []
    for kind, start, end in _segments(text):
        if kind == 'comment':
            first = text.count('\n', 0, start) + 1
            body = text[start:end]
            comments.append((first, first + body.count('\n'), body))
    return comments


def c_code(text):
    """Blank comments, literals and preprocessor lines, keeping line numbers."""
    parts = []
    for kind, start, end in _segments(text):
        part = text[start:end]
        parts.append(part if kind == 'code' else re.sub(r'[^\n]', ' ', part))
    lines = ''.join(parts).split('\n')
    continued = False
    for k, value in enumerate(lines):
        directive = continued or value.lstrip().startswith('#')
        continued = directive and value.rstrip().endswith('\\')
        if directive:
            lines[k] = ''
    return '\n'.join(lines)


def c_functions(text):
    """Return (name, first_line, last_line) for every function definition."""
    code = c_code(text)
    functions = []
    depth, boundary, start = 0, 0, None
    for i, c in enumerate(code):
        if c == '{':
            if depth == 0:
                match = re.search(r'(\w+)\s*\([^;{}=]*\)\s*$', code[boundary:i])
                line = code.count('\n', 0, i) + 1
                start = (match[1], line) if match else None
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                if start:
                    last = code.count('\n', 0, i) + 1
                    functions.append((start[0], start[1], last))
                start = None
                boundary = i + 1
        elif c == ';' and depth == 0:
            boundary = i + 1
    return functions


def python_comments(text):
    """Return (line, body) for every Python comment except a first-line shebang."""
    comments = []
    for token in tokenize.generate_tokens(io.StringIO(text).readline):
        if token.type != tokenize.COMMENT:
            continue
        if token.start[0] == 1 and token.string.startswith('#!'):
            continue
        comments.append((token.start[0], token.string))
    return comments
