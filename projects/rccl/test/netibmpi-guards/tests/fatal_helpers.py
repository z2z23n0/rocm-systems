# Analysis behind the NetIbMPI fatal-helper guard. See ../README.md.
#
# A gtest ASSERT_* expands to `return;`, so it only leaves the function it is
# written in. A void helper that asserts internally therefore returns to its
# caller, which carries on as if nothing happened -- with whatever out-params
# the helper never got round to filling in. That is how a failed connection
# setup reached the net_ib_cast plugin with a null comm and segfaulted.
#
# The rule this module enforces is about helper *definitions*:
#
#     a void helper with an out-parameter must not fail fatally
#
# It deliberately says nothing about call sites. Wrapping such a call in
# ASSERT_NO_FATAL_FAILURE looks like the fix and is not: these helpers fail
# locally -- one bad NIC, one node's plugin init -- so the wrapper ends the test
# on that rank alone while its peer waits at the next collective, and
# MPI_Barrier has no timeout. Trading an unset out-parameter for a hang is not
# a trade worth making, and a guard that recommends it is worse than no guard.
#
# The safe shape is the one SetupConnection uses: return a status, and have the
# ranks agree on it (MPI_Allreduce) before any of them gives up. Helpers built
# that way can assert fatally at the call site because every rank reaches the
# same verdict. The four helpers listed in KNOWN_FATAL_HELPERS predate this and
# are recorded as debt -- that list may shrink, never grow.
#
# Standard library only, so the guard runs anywhere Python 3 does.

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# Macros that expand to a `return` in the enclosing function. GTEST_SKIP is one
# of them: a helper that skips partway through leaves its out-parameter unset
# just as surely as one that fails, and the caller cannot tell the difference.
FATAL_RE = re.compile(
    r"\bASSERT_[A-Z0-9_]+\s*\(|\bFAIL\s*\(|\bGTEST_FAIL\s*\(|\bGTEST_SKIP\s*\("
)
# Test bodies. These are void too, but they ARE the caller of last resort -- an
# ASSERT_ in one of them already ends the test, which is the point.
TEST_MACRO_RE = re.compile(r"\b(TEST|TEST_F|TEST_P)\s*\(")
VOID_FN_RE = re.compile(r"\bvoid\s+(\w+)\s*\(")

SOURCE_SUFFIXES = (".cpp", ".hpp", ".cc", ".h")

# Helpers that already had this shape when the guard was written. Each needs its
# failure agreed across ranks before it can stop being fatal; see ../README.md.
# Shrink this list, never extend it.
KNOWN_FATAL_HELPERS = frozenset({
    "AssertInitAndGetDevices",
    "PostSendWithRetry",
    "PostSingleRecv",
    "RunRecvFlushBurst",
})

# Pointed-to types that are buffers or opaque handles the helper reads, not
# slots it writes back through. `void* comm`, `char* sendBuf` are inputs;
# `void** request`, `int* ndev`, `ncclResult_t* lastFlush` are outputs.
OPAQUE_POINTEE = {"void", "char", "unsigned char", "uint8_t", "int8_t"}


def blank_comments_and_strings(text: str) -> str:
    """Replace comment and string *contents* with spaces, preserving length.

    Every offset in the result still refers to the same character of the
    original, so line numbers and brace matching stay accurate while
    `// ASSERT_EQ(...)` in a comment no longer counts as an assertion.
    """
    out = list(text)
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            while i < n and text[i] != "\n":
                out[i] = " "
                i += 1
        elif c == "/" and i + 1 < n and text[i + 1] == "*":
            out[i] = out[i + 1] = " "
            i += 2
            while i < n and not (text[i] == "*" and i + 1 < n and text[i + 1] == "/"):
                if text[i] != "\n":
                    out[i] = " "
                i += 1
            if i < n:
                out[i] = " "
                if i + 1 < n:
                    out[i + 1] = " "
                i += 2
        elif c in "\"'":
            quote = c
            i += 1
            while i < n and text[i] != quote:
                if text[i] == "\\":
                    out[i] = " "
                    i += 1
                if i < n and text[i] != "\n":
                    out[i] = " "
                i += 1
            i += 1
        else:
            i += 1
    return "".join(out)


def _match_forward(text: str, start: int, open_ch: str, close_ch: str) -> int:
    """Index just past the delimiter matching the one at `start`, or -1."""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == open_ch:
            depth += 1
        elif text[i] == close_ch:
            depth -= 1
            if depth == 0:
                return i + 1
    return -1


def split_top_level(text: str, sep: str = ",") -> list[str]:
    """Split on `sep`, ignoring separators nested in (), [] or {}.

    Angle brackets are deliberately not tracked: `->` is far more common in
    these parameter lists than a template argument, and counting its `>` drove
    the depth negative, which silently merged everything into one entry.
    """
    parts, depth, current = [], 0, []
    for ch in text:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == sep and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    parts.append("".join(current))
    return parts


@dataclass
class Function:
    name: str
    path: Path
    decl_start: int
    body_start: int
    body_end: int
    is_test_body: bool

    def body(self, clean: str) -> str:
        return clean[self.body_start : self.body_end]


def find_functions(clean: str, path: Path) -> list[Function]:
    """Void functions and gtest test bodies, with their brace-matched extents."""
    found: list[Function] = []

    for m in TEST_MACRO_RE.finditer(clean):
        paren_end = _match_forward(clean, m.end() - 1, "(", ")")
        if paren_end < 0:
            continue
        brace = clean.find("{", paren_end)
        if brace < 0 or clean[paren_end:brace].strip():
            continue
        end = _match_forward(clean, brace, "{", "}")
        if end < 0:
            continue
        found.append(Function(m.group(1), path, m.start(), brace, end, True))

    for m in VOID_FN_RE.finditer(clean):
        paren_end = _match_forward(clean, m.end() - 1, "(", ")")
        if paren_end < 0:
            continue
        between = clean[paren_end:]
        brace_rel = between.find("{")
        semi_rel = between.find(";")
        if brace_rel < 0 or (0 <= semi_rel < brace_rel):
            continue  # a declaration, not a definition
        if re.sub(r"\b(const|noexcept|override|final)\b", "", between[:brace_rel]).strip():
            continue
        brace = paren_end + brace_rel
        end = _match_forward(clean, brace, "{", "}")
        if end < 0:
            continue
        found.append(Function(m.group(1), path, m.start(), brace, end, False))

    return found


def line_of(text: str, index: int) -> int:
    return text.count("\n", 0, index) + 1


def collect(root: Path) -> tuple[dict[str, Function], dict[Path, tuple[str, str]]]:
    """Parse every source under `root`. Returns void helpers and file texts."""
    files: dict[Path, tuple[str, str]] = {}
    helpers: dict[str, Function] = {}
    for path in sorted(root.rglob("*")):
        if path.suffix not in SOURCE_SUFFIXES or not path.is_file():
            continue
        raw = path.read_text(errors="ignore")
        clean = blank_comments_and_strings(raw)
        files[path] = (raw, clean)
        for fn in find_functions(clean, path):
            if not fn.is_test_body:
                helpers[fn.name] = fn
    return helpers, files


def fatal_helpers(root: Path) -> tuple[set[str], dict[str, Function], dict[Path, tuple[str, str]]]:
    """Transitive closure of void helpers that can end the caller's test.

    A helper is fatal if it asserts fatally itself, or calls another fatal void
    helper. Both levels occur in this suite, so the closure is computed rather
    than assumed: RunRecvFlushBurst has no assertion of its own.
    """
    helpers, files = collect(root)
    bodies = {name: fn.body(files[fn.path][1]) for name, fn in helpers.items()}

    fatal = {name for name, body in bodies.items() if FATAL_RE.search(body)}
    changed = True
    while changed:
        changed = False
        for name, body in bodies.items():
            if name in fatal:
                continue
            if any(re.search(r"\b%s\s*\(" % re.escape(f), body) for f in fatal):
                fatal.add(name)
                changed = True
    return fatal, helpers, files


def out_param_indices(fn: Function, clean: str) -> list[int]:
    """Positions of parameters the helper writes back through."""
    sig_open = clean.index("(", fn.decl_start)
    sig_close = _match_forward(clean, sig_open, "(", ")") - 1
    params = split_top_level(clean[sig_open + 1 : sig_close])

    indices = []
    for i, param in enumerate(params):
        p = param.strip()
        if not p or "const" in p:
            continue
        decl = p.split("=")[0].strip()  # drop any default argument
        if "**" in decl or "&" in decl:
            indices.append(i)
        elif "*" in decl and decl[: decl.index("*")].strip() not in OPAQUE_POINTEE:
            indices.append(i)
    return indices


def out_param_names(fn: Function, clean: str) -> list[str]:
    sig_open = clean.index("(", fn.decl_start)
    sig_close = _match_forward(clean, sig_open, "(", ")") - 1
    params = split_top_level(clean[sig_open + 1 : sig_close])
    picked = out_param_indices(fn, clean)
    return [params[i].strip() for i in picked if i < len(params)]


@dataclass
class Violation:
    path: Path
    line: int
    helper: str
    out_params: str

    def __str__(self) -> str:
        return (
            f"{self.path.name}:{self.line}: '{self.helper}' is a void helper that "
            f"can fail fatally while handing back {self.out_params}. A caller "
            f"cannot use that safely: the assertion returns from the helper, not "
            f"from the test, so the test carries on with a value that was never "
            f"written. Return a status instead, and agree it across ranks the way "
            f"SetupConnection does."
        )


def find_violations(root: Path) -> list[Violation]:
    """Fatal void helpers with out-parameters, excluding the known debt."""
    fatal, helpers, files = fatal_helpers(root)
    violations: list[Violation] = []

    for name in sorted(fatal):
        if name in KNOWN_FATAL_HELPERS:
            continue
        fn = helpers[name]
        raw, clean = files[fn.path]
        outs = out_param_names(fn, clean)
        if not outs:
            continue  # nothing the caller could go on to misuse
        violations.append(
            Violation(fn.path, line_of(raw, fn.decl_start), name, " and ".join(outs))
        )
    return violations


def stale_known_helpers(root: Path) -> list[str]:
    """Names in KNOWN_FATAL_HELPERS that no longer need to be there."""
    fatal, helpers, files = fatal_helpers(root)
    stale = []
    for name in sorted(KNOWN_FATAL_HELPERS):
        fn = helpers.get(name)
        if fn is None or name not in fatal or not out_param_names(fn, files[fn.path][1]):
            stale.append(name)
    return stale
