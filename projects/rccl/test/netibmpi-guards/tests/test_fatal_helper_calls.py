# Guards the NetIbMPI suite against the defect fixed in PR #10484: a void helper
# that asserts internally returns to its caller, which then uses out-parameters
# the helper never filled in. See ../README.md.

from pathlib import Path

import pytest

import fatal_helpers as fh

# .../test/netibmpi-guards/tests/this_file.py -> parents[2] is .../test
NETIBMPI = Path(__file__).resolve().parents[2] / "transport" / "NetIbMPI"


@pytest.mark.netibmpi_guard
def test_source_tree_is_present():
    # A silently-empty scan would make every other check below pass vacuously.
    assert NETIBMPI.is_dir(), f"{NETIBMPI} not found"
    assert list(NETIBMPI.glob("*.cpp")), f"no sources under {NETIBMPI}"


@pytest.mark.netibmpi_guard
def test_fatal_helpers_are_still_detected():
    # If a refactor renamed or removed every fatal helper this suite would pass
    # for the wrong reason, so assert the analysis still finds some.
    fatal, _, _ = fh.fatal_helpers(NETIBMPI)
    assert fatal, "no fatal void helpers detected -- the parser is probably broken"


@pytest.mark.netibmpi_guard
def test_no_new_fatal_helper_with_an_out_parameter():
    violations = fh.find_violations(NETIBMPI)
    assert not violations, (
        "A void helper that fails fatally cannot hand anything back safely: the "
        "assertion returns from the helper, not from the test.\n\n"
        + "\n".join(str(v) for v in violations)
    )


@pytest.mark.netibmpi_guard
def test_known_debt_list_is_current():
    # The list may shrink, never grow. A name that no longer qualifies means
    # someone fixed it -- drop it, so the list keeps meaning what it says.
    stale = fh.stale_known_helpers(NETIBMPI)
    assert not stale, (
        "These no longer need to be in KNOWN_FATAL_HELPERS; remove them: "
        + ", ".join(stale)
    )


# --- self-tests: the guard has to fail on the shape it exists to catch --------


def _write(tmp_path: Path, body: str) -> Path:
    (tmp_path / "Base.hpp").write_text(body)
    return tmp_path


def test_detects_the_pr10484_shape(tmp_path):
    root = _write(tmp_path, "class F {\n"
                            "    void SetupThing(void** comm) {\n"
                            "        ASSERT_EQ(MakeThing(comm), 0);\n    }\n};\n")
    names = {v.helper for v in fh.find_violations(root)}
    assert "SetupThing" in names


def test_accepts_a_helper_that_returns_status(tmp_path):
    # The safe shape: no fatal assertion, so the caller must look at the result.
    root = _write(tmp_path, "class F {\n"
                            "    ncclResult_t SetupThing(void** comm) {\n"
                            "        return MakeThing(comm);\n    }\n};\n")
    assert fh.find_violations(root) == []


def test_follows_helpers_transitively(tmp_path):
    # Wrapper has no assertion of its own; it is fatal only because SetupThing
    # is, and it hands back an out-parameter of its own.
    root = _write(tmp_path, "class F {\n"
                            "    void SetupThing(void** comm) {\n"
                            "        ASSERT_EQ(MakeThing(comm), 0);\n    }\n"
                            "    void Wrapper(void** comm) {\n"
                            "        SetupThing(comm);\n    }\n};\n")
    names = {v.helper for v in fh.find_violations(root)}
    assert "Wrapper" in names


def test_ignores_helpers_without_out_params(tmp_path):
    # Consuming its arguments and dying leaves the caller no worse off.
    root = _write(tmp_path, "class F {\n"
                            "    void Consumes(void* comm) {\n"
                            "        ASSERT_EQ(Use(comm), 0);\n    }\n};\n")
    assert fh.find_violations(root) == []


def test_buffer_pointers_are_not_out_params(tmp_path):
    # `void*`/`char*` are buffers the helper reads, not slots it writes back.
    root = _write(tmp_path, "class F {\n"
                            "    void Sends(void* comm, char* buf, size_t n) {\n"
                            "        ASSERT_EQ(Post(comm, buf, n), 0);\n    }\n};\n")
    assert fh.find_violations(root) == []


def test_splits_parameters_containing_arrow(tmp_path):
    # `->` in a default argument drove the splitter's depth negative, so the
    # comma after it stopped separating parameters and the whole list merged
    # into one entry -- which the `const` in the first parameter then caused to
    # be skipped wholesale, losing the out-parameter. The arrow has to sit
    # *before* a comma for that to happen, so a trailing default argument would
    # not exercise it.
    root = _write(tmp_path, "class F {\n"
                            "    void SetupThing(const char* tag = g->tag, void** comm = nullptr) {\n"
                            "        ASSERT_EQ(MakeThing(comm), 0);\n    }\n};\n")
    names = {v.helper for v in fh.find_violations(root)}
    assert "SetupThing" in names


def test_gtest_skip_is_fatal_too(tmp_path):
    # GTEST_SKIP() returns from the helper exactly as ASSERT_* does, so a helper
    # that skips before filling its out-parameter leaves the caller with the
    # unset value just the same.
    root = _write(tmp_path, "class F {\n"
                            "    void SetupThing(void** comm) {\n"
                            "        if (!supported) GTEST_SKIP() << \"no device\";\n"
                            "        *comm = Make();\n    }\n};\n")
    names = {v.helper for v in fh.find_violations(root)}
    assert "SetupThing" in names


def test_ignores_assertions_in_comments(tmp_path):
    root = _write(tmp_path, "class F {\n"
                            "    void SetupThing(void** comm) {\n"
                            "        // ASSERT_EQ(MakeThing(comm), 0);\n"
                            "        Quietly(comm);\n    }\n};\n")
    assert fh.find_violations(root) == []


def test_test_bodies_are_not_helpers(tmp_path):
    # A TEST_F body is void and full of ASSERT_*; it is the caller of last
    # resort, not a helper, and must never be reported.
    root = _write(tmp_path, "TEST_F(S, T) {\n    void* comm = nullptr;\n"
                            "    ASSERT_EQ(MakeThing(&comm), 0);\n}\n")
    assert fh.find_violations(root) == []
