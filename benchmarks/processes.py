"""Run functions in separate processes with a time limit.

A solve that calls NumPy code through callbacks cannot be interrupted from
inside JAX, and a crash in foreign code would take the whole benchmark down. A
separate process can be ended after a time limit, and only loses one problem.
"""

import multiprocessing
import queue
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from multiprocessing.process import BaseProcess
from typing import Generic, TypeVar


Result = TypeVar("Result")


@dataclass
class ProcessOutcome(Generic[Result]):
    """The result of a function run in a separate process, or why there is none."""

    result: Result | None
    """The return value, if the function finished."""
    error: str | None
    """The error message if the function raised, crashed or ran out of time."""


@dataclass
class _RunningProcess:
    """A started process with the queue it reports on."""

    index: int
    """The position of its arguments in the input."""
    process: BaseProcess
    """The process itself."""
    results: "multiprocessing.Queue[ProcessOutcome]"
    """The queue the process puts its outcome on."""
    started: float
    """When the process started, from `time.monotonic`."""


def _call_and_report(
    function: Callable[..., object],
    arguments: tuple,
    results: "multiprocessing.Queue[ProcessOutcome]",
) -> None:
    """Call `function` and put its result or error message on `results`."""
    try:
        results.put(ProcessOutcome(function(*arguments), None))
    except Exception as error:
        results.put(ProcessOutcome(None, f"{type(error).__name__}: {error}"))


def _collect(running: _RunningProcess, timeout: float) -> ProcessOutcome | None:
    """Return the outcome of a process if it has one, or `None` if it is running."""
    try:
        outcome = running.results.get(timeout=0.05)
        running.process.join()
        return outcome
    except queue.Empty:
        pass
    if not running.process.is_alive():
        # The process may have reported just before exiting.
        try:
            return running.results.get(timeout=1.0)
        except queue.Empty:
            code = running.process.exitcode
            return ProcessOutcome(None, f"The process exited with code {code}.")
    if time.monotonic() - running.started > timeout:
        running.process.terminate()
        running.process.join()
        return ProcessOutcome(None, f"The process ran longer than {timeout} seconds.")
    return None


def run_in_processes(
    function: Callable[..., Result],
    argument_tuples: Iterable[tuple],
    timeout: float,
    workers: int,
    on_outcome: Callable[[int, ProcessOutcome[Result]], None] | None = None,
) -> list[ProcessOutcome[Result]]:
    """Call `function` once per argument tuple, each in its own process.

    Up to `workers` processes run at a time. A process that exceeds `timeout`
    seconds is terminated. The outcomes come back in the order of the arguments.
    `on_outcome`, if given, is called with the index and outcome of each process
    as soon as it finishes, so callers can save progress. `function` has to be
    importable by name, since processes start with `spawn`.
    """
    context = multiprocessing.get_context("spawn")
    pending = list(enumerate(argument_tuples))
    total = len(pending)
    outcomes: dict[int, ProcessOutcome[Result]] = {}
    running: list[_RunningProcess] = []
    while pending or running:
        while pending and len(running) < workers:
            index, arguments = pending.pop(0)
            results = context.Queue()
            process = context.Process(
                target=_call_and_report, args=(function, arguments, results)
            )
            process.start()
            running.append(_RunningProcess(index, process, results, time.monotonic()))
        still_running = []
        for entry in running:
            outcome = _collect(entry, timeout)
            if outcome is None:
                still_running.append(entry)
            else:
                outcomes[entry.index] = outcome
                if on_outcome is not None:
                    on_outcome(entry.index, outcome)
        running = still_running
    return [outcomes[index] for index in range(total)]
