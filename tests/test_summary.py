"""Tests for the summary of result files against a reference."""

import math
from pathlib import Path

import pytest

from benchmarks.records import ResultFile, SolveRecord, write_result_file
from benchmarks.summary import label_for, main, summarise


def record(name: str, outcome: str, evaluations: int | None) -> SolveRecord:
    """Return a record of a problem with `variables` set to one."""
    return SolveRecord(
        problem=name,
        variables=1,
        equality_constraints=0,
        inequality_constraints=0,
        outcome=outcome,
        message="",
        evaluations=evaluations,
    )


def result_file(configuration: str, records: list[SolveRecord]) -> ResultFile:
    """Return a result file of the `hock_schittkowski` set with `records`."""
    return ResultFile(
        configuration=configuration,
        problem_set="hock_schittkowski",
        commit="test",
        versions={},
        device="cpu",
        date="test",
        records=records,
    )


REFERENCE = result_file(
    "newton",
    [
        record("hs1", "solved", 10),
        record("hs2", "solved", 20),
        record("hs3", "solved", 40),
        record("hs4", "failed", 5),
    ],
)
CANDIDATE = result_file(
    "candidate",
    [
        record("hs1", "solved", 20),
        record("hs2", "failed", 30),
        record("hs3", "solved", 20),
        record("hs4", "solved", 7),
    ],
)


def test_summary_counts_solved_records():
    summary = summarise(REFERENCE, CANDIDATE, "candidate")
    assert summary.solved_count == 3
    assert summary.record_count == 4


def test_summary_ratios_use_the_problems_both_files_solve():
    summary = summarise(REFERENCE, CANDIDATE, "candidate")
    # The shared problems are hs1 (ratio 2) and hs3 (ratio 0.5).
    assert summary.median_ratio == pytest.approx(1.25)
    assert summary.geometric_mean_ratio == pytest.approx(1.0)


def test_summary_lists_problems_the_reference_solves_only():
    summary = summarise(REFERENCE, CANDIDATE, "candidate")
    assert summary.missed_problems == ["hs2"]


def test_reference_against_itself_has_unit_ratios_and_no_misses():
    summary = summarise(REFERENCE, REFERENCE, "newton")
    assert summary.median_ratio == 1.0
    assert summary.geometric_mean_ratio == 1.0
    assert summary.missed_problems == []


def test_ratios_are_nan_without_shared_solved_problems():
    other = result_file("other", [record("hs1", "failed", None)])
    summary = summarise(REFERENCE, other, "other")
    assert math.isnan(summary.median_ratio)
    assert math.isnan(summary.geometric_mean_ratio)


def test_label_drops_the_problem_set_and_keeps_the_precision_suffix():
    path = Path("results/hock_schittkowski-newton-f32.json")
    assert label_for(path, REFERENCE) == "newton-f32"


def test_command_line_prints_the_table_and_the_missed_problems(tmp_path, capsys):
    reference_path = tmp_path / "hock_schittkowski-newton.json"
    candidate_path = tmp_path / "hock_schittkowski-candidate.json"
    write_result_file(REFERENCE, reference_path)
    write_result_file(CANDIDATE, candidate_path)
    main([str(reference_path), str(candidate_path)])
    output = capsys.readouterr().out
    assert "newton" in output
    assert "candidate" in output
    assert "candidate: hs2" in output


def test_command_line_rejects_a_reference_outside_the_files(tmp_path):
    reference_path = tmp_path / "hock_schittkowski-newton.json"
    candidate_path = tmp_path / "hock_schittkowski-candidate.json"
    write_result_file(REFERENCE, reference_path)
    write_result_file(CANDIDATE, candidate_path)
    with pytest.raises(SystemExit):
        main(["--reference", str(tmp_path / "other.json"), str(candidate_path)])
