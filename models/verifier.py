# verifier.py
#
# Implements the first deployable Approach 2 verifier baseline. It prompts a
# separately loaded VLM to inspect a candidate's image evidence and answer,
# then parses a bounded 0-to-1 support score. This is deliberately separate
# from the policy call. A fine-tuned verifier can replace this class later
# without changing the pool or evaluation formats.

from __future__ import annotations

import json
import re
from typing import Optional

from data.approach2_verifier_reranking.schema import CandidateTrace
from models.run_inference import build_messages


def verifier_prompt(question: str, candidate: CandidateTrace) -> str:
    """Ask for evidence-based scoring of one candidate without exposing labels."""
    crop_description = (
        f"The candidate used normalized crop coordinates {candidate.crop_box}."
        if candidate.crop_box is not None
        else "The candidate used the full image."
    )
    return (
        "You are a strict visual verifier. Judge whether the proposed answer is supported by "
        "the supplied image and question. Do not rely on whether the answer merely sounds plausible. "
        "Return exactly one JSON object with a numeric score from 0.0 to 1.0 and a reason of at most 20 words.\n\n"
        f"Question: {question}\n"
        f"Candidate answer: {candidate.prediction}\n"
        f"Candidate rationale: {candidate.rationale or '(none)'}\n"
        f"Candidate evidence: {crop_description}\n\n"
        'Required format: {"score": 0.0, "reason": "short evidence-based reason"}'
    )


def parse_verifier_score(response: str) -> tuple:
    """Extract a bounded score and explanation from a verifier response."""
    try:
        match = re.search(r"\{.*?\}", response, flags=re.DOTALL)
        value = json.loads(match.group(0) if match else response)
        score = float(value["score"])
        reason = str(value.get("reason", ""))
    except (AttributeError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        score_match = re.search(r"[\"']?score[\"']?\s*[:=]\s*([01](?:\.\d+)?)", response, flags=re.IGNORECASE)
        if not score_match:
            raise ValueError(f"Verifier response has no parseable score: {response!r}")
        score = float(score_match.group(1))
        reason = response.strip()
    return max(0.0, min(1.0, score)), reason


class PromptedVLMVerifier:
    """Score candidate traces with a separately loaded VLM checkpoint."""

    def __init__(self, model, processor):
        self.model = model
        self.processor = processor

    def score(self, image: Image.Image, question: str, candidate: CandidateTrace) -> tuple:
        """Return an evidence-support score and the verifier's short explanation."""
        messages = build_messages(image, verifier_prompt(question, candidate))
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        output_ids = self.model.generate(**inputs, max_new_tokens=256, do_sample=False)
        generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
        response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return parse_verifier_score(response)
