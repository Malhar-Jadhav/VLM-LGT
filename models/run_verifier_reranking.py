# run_verifier_reranking.py
#
# Scores an existing Approach 2 candidate pool with a separately loaded visual
# verifier and reports the comparisons required for the project claim.

import argparse
import json
from pathlib import Path

from PIL import Image

from data.approach2_verifier_reranking.evaluation import evaluate_candidate_groups
from data.approach2_verifier_reranking.schema import CandidateTrace
from data.common.utils import read_jsonl, write_jsonl
from models.load_model import load_qwen3_vl
from models.verifier import PromptedVLMVerifier


def _verifier_image(record: dict, candidate: CandidateTrace) -> Image.Image:
    """Load the crop used by a candidate, falling back to the original image."""
    image_path = candidate.crop_path or record.get("image_path")
    if not image_path:
        raise ValueError(f"Pool record {record.get('example_id', '<unknown>')} has no image_path.")
    return Image.open(image_path).convert("RGB")


def score_pool_records(records: list, verifier: PromptedVLMVerifier) -> list:
    """Attach a score and explanation to every trace in a fixed candidate pool."""
    scored_records = []
    for record_index, record in enumerate(records, start=1):
        scored_candidates = []
        for raw_candidate in record.get("candidates", []):
            candidate = CandidateTrace.from_dict(raw_candidate)
            score, reason = verifier.score(_verifier_image(record, candidate), candidate.question, candidate)
            candidate.verifier_score = score
            candidate.metadata["verifier_reason"] = reason
            scored_candidates.append(candidate.to_dict())
        scored_records.append({**record, "candidates": scored_candidates})
        print(f"[verify] {record_index}/{len(records)} examples")
    return scored_records


def main() -> None:
    """Load a saved pool, score it, and write scored records plus metrics."""
    parser = argparse.ArgumentParser(description="Run Approach 2 verifier reranking on a saved candidate pool.")
    parser.add_argument("--pool", required=True, help="Path to a JSONL candidate pool from build_failure_pool.py.")
    parser.add_argument("--verifier-size", choices=["2b", "4b"], default="2b")
    parser.add_argument("--limit", type=int, default=None, help="Optional local smoke-test cap.")
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--output-dir", default="results/approach2")
    args = parser.parse_args()

    records = read_jsonl(args.pool)
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        raise ValueError("The selected pool is empty.")

    model, processor = load_qwen3_vl(args.verifier_size)
    verifier = PromptedVLMVerifier(model, processor)
    scored_records = score_pool_records(records, verifier)
    groups = [[CandidateTrace.from_dict(candidate) for candidate in record["candidates"]] for record in scored_records]
    metrics = evaluate_candidate_groups(groups, random_seed=args.random_seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(scored_records, str(output_dir / "scored_pool.jsonl"))
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics.to_dict(), handle, indent=2)
    print(json.dumps(metrics.to_dict(), indent=2))


if __name__ == "__main__":
    main()
