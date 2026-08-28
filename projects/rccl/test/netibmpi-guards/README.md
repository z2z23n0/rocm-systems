# NetIbMPI assertion guards

CPU-only, no GPU and no build required. Run them with:

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
./venv/bin/python -m pytest tests -v
```

The host-test pipeline runs them for you (`test/host/run_host_tests.sh guards`).

## What this guards, and why

A gtest `ASSERT_*` expands to `return;`. It leaves **the function it is written
in** — not the test. So a `void` helper that asserts internally returns to its
caller, and the caller carries on as though the helper had succeeded.

That is not hypothetical. It is the defect fixed in
[PR #10484](https://github.com/ROCm/rocm-systems/pull/10484):
`SetupCastConnection` asserted internally when a connection could not be
established, returned to the test body with the caller's three comm pointers
still at their initialised `nullptr`, and the body's next call handed one of
them to the plugin's `regMr`, which read the device list off it and segfaulted.
A failed connection setup surfaced as a SIGSEGV instead of a readable failure.

## The rule

> A `void` helper with an **out-parameter** must not fail fatally.

Both halves matter. A helper that only consumes its arguments leaves the caller
no worse off by returning early. A helper that was supposed to hand back a comm,
a request or a device count leaves the caller holding a value that was never
written — and the caller then uses it.

"Fatal" is transitive. A helper counts if it uses `ASSERT_*`/`FAIL()`, *or* if
it calls another fatal void helper. `GdrFlushTests.cpp`'s `RunRecvFlushBurst` is
fatal only through the second rule, so the guard computes the closure rather
than pattern-matching one level.

A parameter counts as an out-parameter when it is a `T**`, a non-`const`
reference, or a pointer to a typed value (`int*`, `ncclResult_t*`). Pointers to
`void`/`char` are buffers and handles the helper reads.

## What this guard deliberately does not ask for

**It never tells you to wrap a call in `ASSERT_NO_FATAL_FAILURE`.** That looks
like the fix and is not.

These helpers fail *locally* — one node's plugin init, one NIC's transport
hiccup — so the failure lands on one rank while the others carry on. Wrapping
the call turns "this rank continues with an unset out-parameter" into "this rank
leaves the test and its peers wait at the next collective". `MPI_Barrier` has no
timeout, so that is a hang. Calling a helper from every rank does not make it
fail on every rank, and that distinction is the whole point.

The safe shape is the one `SetupConnection` uses: return a status, and have the
ranks agree on it with `MPI_Allreduce` before any of them gives up. A helper
built that way can be asserted on at the call site, because every rank reaches
the same verdict — which is why `ASSERT_SETUP_CONNECTION` is safe and a bare
wrapper is not.

## Known debt

`KNOWN_FATAL_HELPERS` in `tests/fatal_helpers.py` lists four helpers that
already had this shape:

| helper | hands back |
|---|---|
| `AssertInitAndGetDevices` | `int* ndev` |
| `PostSendWithRetry` | `void** request` |
| `PostSingleRecv` | `void** request` |
| `RunRecvFlushBurst` | `ncclResult_t* rank0LastFlush` |

Fixing one means giving it a status return *and* agreeing that status across
ranks — the agreement cannot live inside the helper, because these are often
called from inside `if (rank == N)` where only one rank would reach the
collective. It has to be hoisted to a point every rank reaches, which changes
each test's control flow. That is real work and is not this change.

**The list may shrink, never grow.** A test fails if a name in it no longer
qualifies, so fixing a helper forces the list to be updated with it.

## Scope

`test/transport/NetIbMPI/` only. The rest of `test/` has the same pattern in
quantity and has not been audited; widening the scope is a separate change, not
a flag flip.
