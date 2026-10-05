# test_approach2.py
#
# Local, network-free checks for the Approach 2 data contract and metrics.

from data.approach2_verifier_reranking.candidates import (
    answer_is_correct,
    default_candidate_specs,
    extract_answer_and_rationale,
)
from data.approach2_verifier_reranking.evaluation import evaluate_candidate_groups
from data.approach2_verifier_reranking.schema import CandidateTrace
from models.verifier import parse_verifier_score


def _trace(identifier: str, correct: bool, score: float) -> CandidateTrace:
    """Build a minimal labeled trace for metric tests."""
    return CandidateTrace(
        candidate_id=identifier,
        image_id="image-1",
        question="What color is the object?",
        prediction="green" if correct else "blue",
        raw_response="",
        is_correct=correct,
        verifier_score=score,
    )


def test_candidate_specs_are_controlled_and_label_free():
    """Default sampling should vary settings without using a gold box."""
    specs = default_candidate_specs(4)
    assert len(specs) == 4
    assert all(spec["crop_box"] is None for spec in specs)
    assert len({spec["seed"] for spec in specs}) == 4


def test_answer_labeling_supports_aliases():
    """Exact-match labeling should normalize case and simple punctuation."""
    assert answer_is_correct("Green!", ["blue", "green"])
    assert not answer_is_correct("red", ["blue", "green"])


def test_response_parser_keeps_rationale_and_final_answer():
    """Candidate prompts use an explicit FINAL line that can be parsed reliably."""
    answer, rationale = extract_answer_and_rationale("The stapler beside the notebook is green.\nFINAL: green")
    assert answer == "green"
    assert "stapler" in rationale


def test_verifier_parser_bounds_score():
    """A verifier score should remain a valid probability-like value."""
    score, reason = parse_verifier_score('{"score": 1.3, "reason": "visible evidence"}')
    assert score == 1.0
    assert reason == "visible evidence"


def test_metrics_compare_same_candidate_pool():
    """Verifier and oracle results should reflect labels within each saved group."""
    groups = [
        [_trace("a0", False, 0.1), _trace("a1", True, 0.9)],
        [_trace("b0", True, 0.8), _trace("b1", False, 0.2)],
    ]
    metrics = evaluate_candidate_groups(groups, random_seed=0)
    assert metrics.single_pass_accuracy == 0.5
    assert metrics.verifier_selected_accuracy == 1.0
    assert metrics.oracle_best_of_k_accuracy == 1.0
    assert metrics.selection_gain >= 0.0
