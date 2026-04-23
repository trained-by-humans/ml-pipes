"""
Concurrency tests for BatchGate.

These tests target BatchGate directly (not through Pipeline) so that timing
and thread ordering can be controlled precisely.  They cover every ordering
of arrivals the implementation must handle:

  - Full batch (all threads arrive before size is reached)
  - Timeout-triggered partial batch
  - Cascade wakeup: timeout on T1 must wake T2 quickly (not wait for T2's own timeout)
  - Cross-batch cascade isolation: T1's cascade must not bleed into the next batch
  - Stolen entry: woken follower must not drain entries that belong to the next batch
  - Rapid sequential batches under maximum concurrency
  - Concurrent overlapping batches with no result mixing
  - Exception isolation: one batch's failure must not corrupt another
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from ml_pipes.batch import BatchGate, LeaderBatch, FollowerResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(gate: BatchGate, value: int, batch_delay: float = 0.0) -> int:
    """
    Simulate one pipeline call through the gate.
    Pipeline doubles every value.  batch_delay lets tests hold the leader
    inside the batch region to create specific interleavings.
    """
    outcome = gate.enter(value)
    if isinstance(outcome, LeaderBatch):
        if batch_delay:
            time.sleep(batch_delay)
        return gate.distribute([v * 2 for v in outcome.inputs])
    return outcome.result


def _run_tracked(gate: BatchGate, value: int, results: dict, errors: list,
                 batch_delay: float = 0.0, batches: dict | None = None) -> None:
    """Thread target that stores result or exception in shared containers.

    If *batches* is provided, the leader writes ``{v: frozenset(batch_inputs)}``
    for every member of its batch before distributing results.  This lets tests
    assert on *which* inputs were co-batched, not just that each result is correct.
    """
    try:
        outcome = gate.enter(value)
        if isinstance(outcome, LeaderBatch):
            if batch_delay:
                time.sleep(batch_delay)
            if batches is not None:
                batch_key = frozenset(outcome.inputs)
                for v in outcome.inputs:
                    batches[v] = batch_key
            results[value] = gate.distribute([v * 2 for v in outcome.inputs])
        else:
            if outcome.exception is not None:
                raise outcome.exception
            results[value] = outcome.result
    except Exception as exc:  # noqa: BLE001
        errors.append((value, type(exc).__name__, str(exc)))


def _start_and_join(threads: list[threading.Thread], timeout: float = 5.0) -> None:
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=timeout)


# ---------------------------------------------------------------------------
# Full batch — parameterised over batch size
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("size", [1, 2, 4, 8])
def test_full_batch_every_thread_gets_its_own_value_doubled(size):
    """size threads arrive simultaneously and form exactly one full batch."""
    gate = BatchGate(size=size, timeout=5.0)
    results = {}
    errors = []
    batches: dict = {}
    barrier = threading.Barrier(size)

    def work(v):
        barrier.wait()
        _run_tracked(gate, v, results, errors, batches=batches)

    _start_and_join([threading.Thread(target=work, args=(i,)) for i in range(size)])

    assert not errors, errors
    expected_batch = frozenset(range(size))
    for i in range(size):
        assert results[i] == i * 2, f"value {i}: expected {i * 2}, got {results[i]}"
        assert batches.get(i) == expected_batch, (
            f"value {i} must be in the single batch {expected_batch}; got {batches.get(i)!r}"
        )


# ---------------------------------------------------------------------------
# Timeout / partial batch — parameterised over fill count
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_threads", [1, 2, 3])
def test_partial_batch_timeout_correct_results(n_threads):
    """k < size threads form a partial batch after the timeout fires."""
    gate = BatchGate(size=8, timeout=0.1)
    results = {}
    errors = []
    batches: dict = {}
    barrier = threading.Barrier(n_threads)

    def work(v):
        barrier.wait()
        _run_tracked(gate, v, results, errors, batches=batches)

    _start_and_join([threading.Thread(target=work, args=(i,)) for i in range(n_threads)],
                    timeout=3.0)

    assert not errors, errors
    expected_batch = frozenset(range(n_threads))
    for i in range(n_threads):
        assert results[i] == i * 2
        assert batches.get(i) == expected_batch, (
            f"value {i} must be in the single partial batch {expected_batch}; "
            f"got {batches.get(i)!r}"
        )


# ---------------------------------------------------------------------------
# Cascade wakeup efficiency
# ---------------------------------------------------------------------------

def test_cascade_timeout_wakes_all_batch_members_together():
    """
    T1 arrives 30 ms before T2.  When T1's timeout fires it cascades notify_all()
    so T2 wakes immediately rather than waiting for its own timeout deadline.
    Both threads must complete well within two timeout periods.
    """
    timeout = 0.1
    gate = BatchGate(size=4, timeout=timeout)  # batch never fills (only 2 threads)
    results = {}
    errors = []

    batches: dict = {}
    t_start = time.monotonic()

    def work(v):
        _run_tracked(gate, v, results, errors, batches=batches)

    t1 = threading.Thread(target=work, args=(1,))
    t2 = threading.Thread(target=work, args=(2,))
    t1.start()
    time.sleep(0.03)   # T2 arrives 30 ms after T1
    t2.start()

    t1.join(timeout=2.0)
    t2.join(timeout=2.0)
    elapsed = time.monotonic() - t_start

    assert not errors, errors
    assert results[1] == 2
    assert results[2] == 4
    # Both should finish within ~2 × timeout + slack, not each waiting their own full timeout
    assert elapsed < timeout * 3, (
        f"Cascade did not wake T2 promptly: took {elapsed:.3f}s "
        f"(expected < {timeout * 3:.3f}s)"
    )
    # Cascade must group T1 and T2 into the same batch, not two solo batches
    assert batches[1] == batches[2] == frozenset({1, 2}), (
        f"T1 and T2 must be co-batched; got {batches.get(1)!r} vs {batches.get(2)!r}"
    )

# ---------------------------------------------------------------------------
# Timeout boundary — late arrival straddles the drain window
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("offset_ms", [-10, -5, 0, 5, 10])
def test_timeout_boundary_late_arrival_always_gets_correct_result(offset_ms):
    """
    T1 and T2 enter together; the gate (size=4) never fills so T1's timeout
    fires at t≈timeout.  T3 arrives at t = timeout + offset_ms, which places
    it either just before or just after the deadline.

    Two possible outcomes — both must succeed:

      Early (offset < 0): T3 joins the lobby before T1 cascades, rides the
        same cond.notify_all() and ends up in the same batch as T1/T2.

      Late (offset >= 0): T3 may race the leader's lock acquisition.
        If the leader hasn't drained yet T3 can still join the current batch;
        otherwise T3 starts a fresh batch, waits its own timeout, and becomes
        a solo leader.

    In every case T1→2, T2→4, T3→6.  No deadlock, no exception.
    """
    timeout = 0.05
    gate = BatchGate(size=4, timeout=timeout)
    results: dict = {}
    errors: list = []
    batches: dict = {}

    def work(v):
        _run_tracked(gate, v, results, errors, batches=batches)

    t1 = threading.Thread(target=work, args=(1,))
    t2 = threading.Thread(target=work, args=(2,))
    t1.start()
    t2.start()

    delay = timeout + offset_ms / 1000.0
    if delay > 0:
        time.sleep(delay)

    t3 = threading.Thread(target=work, args=(3,))
    t3.start()

    # Worst case: T3 misses the current batch and waits its own full timeout.
    join_timeout = timeout * 6 + 1.0
    t1.join(timeout=join_timeout)
    t2.join(timeout=join_timeout)
    t3.join(timeout=join_timeout)

    assert not errors, errors
    assert results.get(1) == 2, f"T1: expected 2, got {results.get(1)}"
    assert results.get(2) == 4, f"T2: expected 4, got {results.get(2)}"
    assert results.get(3) == 6, f"T3: expected 6, got {results.get(3)}"
    # T1 and T2 start together — they must always land in the same batch.
    assert batches.get(1) == batches.get(2), (
        f"T1 and T2 must be co-batched; got {batches.get(1)!r} vs {batches.get(2)!r}"
    )
    # T3 either joined T1/T2 (early) or formed a solo batch (late) — never something else.
    assert batches.get(3) in (frozenset({1, 2, 3}), frozenset({3})), (
        f"T3 batch must be {{1,2,3}} or {{3}}; got {batches.get(3)!r}"
    )


# ---------------------------------------------------------------------------
# Stolen entry — woken follower must not drain next-batch entries
# ---------------------------------------------------------------------------

def test_follower_wakeup_does_not_steal_next_batch_entry():
    """
    After batch 1 drains, T4 may enqueue before the batch-1 follower (T1)
    finishes its Phase 1 lock acquisition.  T1 must not drain T4's entry.

    Without the 'entry in self._pending' guard, T1 would drain [T4] and
    batch.index(T1's entry) would raise ValueError.
    """
    gate = BatchGate(size=2, timeout=5.0)
    results = {}
    errors = []
    batches: dict = {}

    # Hold the batch-1 leader inside the batch region so that when it
    # calls distribute() and fires T1's event, T1 re-acquires the lock
    # and calls its cascade — at the same time as T4 enqueues.
    can_distribute = threading.Event()

    def run_with_hold(v):
        try:
            outcome = gate.enter(v)
            if isinstance(outcome, LeaderBatch):
                can_distribute.wait(timeout=3.0)  # hold leader in batch region
                batch_key = frozenset(outcome.inputs)
                for x in outcome.inputs:
                    batches[x] = batch_key
                results[v] = gate.distribute([x * 2 for x in outcome.inputs])
            else:
                results[v] = outcome.result
        except Exception as exc:  # noqa: BLE001
            errors.append((v, type(exc).__name__, str(exc)))

    def run(v):
        _run_tracked(gate, v, results, errors, batches=batches)

    # Batch 1: T1 waits, T2 fills and becomes leader (held)
    t1 = threading.Thread(target=run_with_hold, args=(1,))
    t2 = threading.Thread(target=run_with_hold, args=(2,))
    t1.start()
    time.sleep(0.02)
    t2.start()

    # Wait until one of them is the leader and blocked
    time.sleep(0.05)

    # T3 and T4 arrive while T1/T2 are mid-wakeup
    t3 = threading.Thread(target=run, args=(3,))
    t4 = threading.Thread(target=run, args=(4,))
    t3.start()

    # Now release the leader so T1 starts its Phase 1 cascade
    can_distribute.set()

    t4.start()

    for t in [t1, t2, t3, t4]:
        t.join(timeout=5.0)

    assert not errors, f"Unexpected exceptions: {errors}"
    assert results.get(1) == 2
    assert results.get(2) == 4
    assert results.get(3) == 6
    assert results.get(4) == 8
    # Strict membership: T1/T2 must form batch 1, T3/T4 must form batch 2.
    # A stolen-entry bug would put T3 or T4 into {1,2}'s batch.
    assert batches.get(1) == batches.get(2) == frozenset({1, 2}), (
        f"Batch 1 must be exactly {{1,2}}; got {batches.get(1)!r}"
    )
    assert batches.get(3) == batches.get(4) == frozenset({3, 4}), (
        f"Batch 2 must be exactly {{3,4}}; got {batches.get(3)!r}"
    )


# ---------------------------------------------------------------------------
# Rapid sequential batches — stress test for all orderings
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("size,n_batches", [(2, 30), (4, 20), (8, 10)])
def test_rapid_sequential_batches_no_corruption(size, n_batches):
    """
    n_batches × size threads all launched simultaneously.  Maximum interleaving
    between batch generations.  Every thread must get exactly its own value * 2.

    Catches:
      - Stolen entry: a woken follower drains entries from the next batch,
        causing batch.index(entry) to raise ValueError.
      - Cascade bleed: a follower's cascade notify_all() prematurely wakes a
        thread from the next batch generation, forming a spurious 1-item batch.
        (Deterministically forcing this race requires pausing a thread between
        cond.wait() returning and notify_all() firing, which is not feasible
        without mocking the threading primitives.  Maximum concurrency across
        many generations is the practical substitute.)
      - Cross-batch result mixing, deadlocks.
    """
    gate = BatchGate(size=size, timeout=0.05)
    total = size * n_batches
    results = [None] * total
    batches: list = [None] * total
    errors = []
    barrier = threading.Barrier(total)

    def work(i):
        barrier.wait()
        try:
            outcome = gate.enter(i)
            if isinstance(outcome, LeaderBatch):
                batch_key = frozenset(outcome.inputs)
                for v in outcome.inputs:
                    batches[v] = batch_key
                results[i] = gate.distribute([v * 2 for v in outcome.inputs])
            else:
                results[i] = outcome.result
        except Exception as exc:  # noqa: BLE001
            errors.append((i, type(exc).__name__, str(exc)))

    threads = [threading.Thread(target=work, args=(i,)) for i in range(total)]
    _start_and_join(threads, timeout=15.0)

    assert not errors, f"Exceptions during stress run: {errors}"
    for i in range(total):
        assert results[i] == i * 2, f"Thread {i}: expected {i * 2}, got {results[i]}"

    # Batch grouping: every value in exactly one batch, no batch exceeds size.
    assert all(b is not None for b in batches), "Some threads have no recorded batch"
    batch_groups = set(batches[i] for i in range(total))
    for group in batch_groups:
        assert len(group) <= size, f"Batch {group} exceeds gate size {size}"
    # Disjoint partition: sum of sizes == total (no value in two batches)
    assert sum(len(g) for g in batch_groups) == total, (
        "Batch groups overlap — a value appears in more than one batch"
    )
    assert frozenset().union(*batch_groups) == frozenset(range(total)), (
        "Batch groups do not cover all values"
    )


# ---------------------------------------------------------------------------
# Concurrent overlapping batches — result isolation
# ---------------------------------------------------------------------------

def test_two_concurrent_batches_results_do_not_mix():
    """
    Two independent sets of threads hit the same gate simultaneously.
    Each thread must get back its own value * 2, not a value from the other batch.
    """
    gate = BatchGate(size=3, timeout=0.5)
    # Two groups of 3 threads each — they may form two concurrent batches.
    inputs = list(range(6))
    results = [None] * 6
    batches: list = [None] * 6
    errors = []
    barrier = threading.Barrier(6)

    def work(i):
        barrier.wait()
        try:
            outcome = gate.enter(i)
            if isinstance(outcome, LeaderBatch):
                batch_key = frozenset(outcome.inputs)
                for v in outcome.inputs:
                    batches[v] = batch_key
                results[i] = gate.distribute([v * 2 for v in outcome.inputs])
            else:
                results[i] = outcome.result
        except Exception as exc:  # noqa: BLE001
            errors.append((i, type(exc).__name__, str(exc)))

    _start_and_join([threading.Thread(target=work, args=(i,)) for i in inputs],
                    timeout=5.0)

    assert not errors, errors
    for i in inputs:
        assert results[i] == i * 2, f"Thread {i}: expected {i * 2}, got {results[i]}"

    # Structural check: two disjoint batches covering all 6 values, each ≤ size=3.
    assert all(b is not None for b in batches), "Some threads have no recorded batch"
    batch_groups = {batches[i] for i in inputs}
    assert len(batch_groups) == 2, (
        f"Expected exactly 2 batches, got {len(batch_groups)}: {batch_groups}"
    )
    b1, b2 = batch_groups
    assert b1 & b2 == frozenset(), f"Batches overlap: {b1 & b2}"
    assert b1 | b2 == frozenset(range(6)), "Batches don't cover all inputs"
    for b in (b1, b2):
        assert len(b) <= 3, f"Batch exceeds gate size: {b}"


# ---------------------------------------------------------------------------
# Exception isolation
# ---------------------------------------------------------------------------

def test_exception_in_batch_does_not_corrupt_next_batch():
    """
    Batch 1 raises; all its members receive the exception.
    Batch 2 runs cleanly immediately after and produces correct results.
    """
    gate = BatchGate(size=2, timeout=1.0)
    errors_b1 = []
    results_b2 = {}

    # --- batch 1: leader raises (mirrors how _step_into_batch handles exceptions) ---
    def run_failing(v):
        outcome = gate.enter(v)
        if isinstance(outcome, LeaderBatch):
            exc = RuntimeError("deliberate failure")
            gate.distribute_exception(exc)
            errors_b1.append((v, str(exc)))
        elif outcome.exception is not None:
            errors_b1.append((v, str(outcome.exception)))

    t1 = threading.Thread(target=run_failing, args=(1,))
    t2 = threading.Thread(target=run_failing, args=(2,))
    t1.start()
    time.sleep(0.02)
    t2.start()
    t1.join(timeout=2.0)
    t2.join(timeout=2.0)

    assert len(errors_b1) == 2, f"Expected both batch-1 threads to raise, got: {errors_b1}"

    # --- batch 2: clean run ---
    def run(v):
        _run_tracked(gate, v, results_b2, [])

    t3 = threading.Thread(target=run, args=(3,))
    t4 = threading.Thread(target=run, args=(4,))
    t3.start()
    time.sleep(0.02)
    t4.start()
    t3.join(timeout=2.0)
    t4.join(timeout=2.0)

    assert results_b2[3] == 6
    assert results_b2[4] == 8


def test_exception_in_one_concurrent_batch_does_not_affect_other():
    """
    Two batches run simultaneously.  One raises; only its members see the
    exception.  The other batch completes correctly.
    """
    gate = BatchGate(size=2, timeout=1.0)
    results = {}
    exc_values = []

    # Batch A raises; batch B uses a gate-level delay to overlap with A.
    batch_a_entered = threading.Barrier(2)
    batch_b_start = threading.Event()

    def run_a(v):
        batch_a_entered.wait()
        outcome = gate.enter(v)
        if isinstance(outcome, LeaderBatch):
            batch_b_start.set()
            time.sleep(0.05)       # hold A's leader so B overlaps
            gate.distribute_exception(RuntimeError("batch A failed"))
            exc_values.append(v)
        elif outcome.exception is not None:
            exc_values.append(v)

    def run_b(v):
        batch_b_start.wait()
        outcome = gate.enter(v)
        if isinstance(outcome, LeaderBatch):
            results[v] = gate.distribute([x * 2 for x in outcome.inputs])
        else:
            results[v] = outcome.result

    threads = [
        threading.Thread(target=run_a, args=(1,)),
        threading.Thread(target=run_a, args=(2,)),
        threading.Thread(target=run_b, args=(3,)),
        threading.Thread(target=run_b, args=(4,)),
    ]
    _start_and_join(threads, timeout=5.0)

    # Both batch-A members must have seen the exception.
    assert sorted(exc_values) == [1, 2], f"Batch A exceptions: {exc_values}"
    # Batch B must be unaffected.
    assert results.get(3) == 6, f"Batch B value 3: {results.get(3)}"
    assert results.get(4) == 8, f"Batch B value 4: {results.get(4)}"


def test_threadpool_reuse_does_not_leak_local_state():
    """
    When used in a ThreadPool, worker threads are reused.
    This test ensures that threading.local() state is correctly purged
    so subsequent tasks on the same thread do not fail.
    """
    gate = BatchGate(size=2, timeout=0.1)

    def run_pipeline(v):
        outcome = gate.enter(v)
        if isinstance(outcome, LeaderBatch):
            return gate.distribute([x * 2 for x in outcome.inputs])
        return outcome.result

    # Run 10 batches using only 2 threads.
    # This guarantees threads are reused for multiple leader/follower roles.
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run_pipeline, i) for i in range(20)]

        results = [f.result(timeout=5.0) for f in futures]

    for i in range(20):
        assert results[i] == i * 2


def test_reentrant_pipeline_is_safe():
    """
    A thread that completes a batch (as leader or follower) must be able
    to immediately enter the gate again without deadlocking or state corruption.
    """
    gate = BatchGate(size=2, timeout=1.0)
    results = {}

    def run_twice(v):
        # First pass
        out1 = gate.enter(v)
        res1 = gate.distribute([x * 2 for x in out1.inputs]) if isinstance(out1, LeaderBatch) else out1.result

        # Immediate Second pass (Re-entrancy)
        out2 = gate.enter(res1)
        res2 = gate.distribute([x * 2 for x in out2.inputs]) if isinstance(out2, LeaderBatch) else out2.result

        results[v] = res2

    t1 = threading.Thread(target=run_twice, args=(10,))
    t2 = threading.Thread(target=run_twice, args=(20,))

    t1.start()
    t2.start()
    t1.join(timeout=2.0)
    t2.join(timeout=2.0)

    # 10 * 2 * 2 = 40
    # 20 * 2 * 2 = 80
    assert results == {10: 40, 20: 80}