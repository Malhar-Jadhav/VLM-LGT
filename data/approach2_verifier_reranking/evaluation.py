# evaluation.py
#
# Computes selection metrics from one fixed candidate pool. Random and
# verifier metrics must use the same pool; otherwise the comparison does not
# isolate the verifier's contribution.

from dataclasses import dataclass
from typing import Iterable, List, Sequence

from data.approach2_verifier_reranking.schema import CandidateTrace


@dataclass
class SelectionMetrics:
    """Aggregate outcomes for a collection of candidate groups."""

    groups: int
    single_pass_accuracy: float
    random_from_pool_accuracy: float
    verifier_selected_accuracy: float
    oracle_best_of_k_accuracy: float

    @property
    def selection_gain(self) -> float:
        """Measure verifier value beyond uniform selection from the same pool."""
        return self.verifier_selected_accuracy - self.random_from_pool_accuracy

    @property
    def oracle_gap(self) -> float:
        """Measure available improvement that the current verifier misses."""
        return self.oracle_best_of_k_accuracy - self.verifier_selected_accuracy

    def to_dict(self) -> dict:
        """Return JSON-compatible metrics for saving in an experiment report."""
        return {
            "groups": self.groups,
            "single_pass_accuracy": self.single_pass_accuracy,
            "random_from_pool_accuracy": self.random_from_pool_accuracy,
            "verifier_selected_accuracy": self.verifier_selected_accuracy,
            "oracle_best_of_k_accuracy": self.oracle_best_of_k_accuracy,
            "selection_gain": self.selection_gain,
            "oracle_gap": self.oracle_gap,
        }


def _correct(candidate: CandidateTrace) -> float:
    """Convert a candidate label to a metric contribution."""
    return float(bool(candidate.is_correct))


def evaluate_candidate_groups(groups: Iterable[Sequence[CandidateTrace]], random_seed: int = 0) -> SelectionMetrics:
    """Evaluate policy, random, verifier, and oracle selection on fixed pools.

    random_seed remains accepted for compatibility with earlier experiment
    scripts. The random baseline now uses its exact expectation, so no sampled
    seed is needed to make it reproducible.
    """
    usable_groups: List[Sequence[CandidateTrace]] = [list(group) for group in groups if group]
    if not usable_groups:
        raise ValueError("No non-empty candidate groups were supplied for evaluation.")
    if any(any(candidate.is_correct is None for candidate in group) for group in usable_groups):
        raise ValueError("Every candidate needs an is_correct label before evaluation.")
    if any(any(candidate.verifier_score is None for candidate in group) for group in usable_groups):
        raise ValueError("Every candidate needs a verifier_score before verifier evaluation.")

    single_pass = sum(_correct(group[0]) for group in usable_groups)
    # Use the exact expectation of uniformly picking one candidate. This is
    # less noisy than taking one random draw per group and remains equivalent
    # to the random-from-pool baseline in expectation.
    random_score = sum(sum(_correct(candidate) for candidate in group) / len(group) for group in usable_groups)
    verifier_score = sum(_correct(max(group, key=lambda candidate: candidate.verifier_score)) for group in usable_groups)
    oracle_score = sum(float(any(candidate.is_correct for candidate in group)) for group in usable_groups)
    count = len(usable_groups)
    return SelectionMetrics(
        groups=count,
        single_pass_accuracy=single_pass / count,
        random_from_pool_accuracy=random_score / count,
        verifier_selected_accuracy=verifier_score / count,
        oracle_best_of_k_accuracy=oracle_score / count,
    )
