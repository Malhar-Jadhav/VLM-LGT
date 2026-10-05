# candidates.py
#
# Generates and labels policy traces for Approach 2. The default candidate
# strategy changes decoding settings while keeping the same image. A future
# grounding or crop proposal module can pass proposed_boxes without leaking a
# ground-truth box into inference.

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from data.approach2_verifier_reranking.schema import CandidateTrace
from models.run_inference import build_messages


DEFAULT_TEMPERATURES = (0.0, 0.3, 0.7, 1.0)


def normalize_answer(value: Any) -> str:
    """Normalize an answer for simple exact-match labeling.

    Dataset-specific evaluators may be stricter or more permissive. This
    default intentionally stays transparent so a project can replace it when a
    benchmark provides an official scoring script.
    """
    if value is None:
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return re.sub(r"[^\w\s.-]", "", text).strip()


def answer_is_correct(prediction: str, reference: Any) -> bool:
    """Check a prediction against one answer or an iterable of answer aliases."""
    prediction = normalize_answer(prediction)
    if isinstance(reference, dict):
        reference = reference.get("answer") or reference.get("answers") or reference.get("label")
    if isinstance(reference, (list, tuple, set)):
        return prediction in {normalize_answer(answer) for answer in reference}
    return prediction == normalize_answer(reference)


def extract_answer_and_rationale(response: str) -> tuple:
    """Parse the requested FINAL: answer convention without discarding text."""
    match = re.search(r"(?:^|\n)\s*FINAL\s*:\s*(.+)", response, flags=re.IGNORECASE)
    if not match:
        return response.strip(), ""
    answer = match.group(1).strip().splitlines()[0]
    rationale = response[: match.start()].strip()
    return answer, rationale


def _validate_box(box: Sequence[float]) -> List[float]:
    """Validate normalized crop coordinates before an image is cropped."""
    if len(box) != 4:
        raise ValueError("A crop box must have four values: x1, y1, x2, y2.")
    x1, y1, x2, y2 = (float(value) for value in box)
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        raise ValueError("Crop coordinates must satisfy 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1.")
    return [x1, y1, x2, y2]


def crop_image(image: Image.Image, box: Optional[Sequence[float]]) -> Image.Image:
    """Return the full image or a crop specified by normalized coordinates."""
    if box is None:
        return image.copy()
    x1, y1, x2, y2 = _validate_box(box)
    width, height = image.size
    return image.crop((round(x1 * width), round(y1 * height), round(x2 * width), round(y2 * height)))


def default_candidate_specs(k: int, proposed_boxes: Optional[Sequence[Sequence[float]]] = None) -> List[Dict[str, Any]]:
    """Create controlled candidate settings without using label-derived crops."""
    if k < 1:
        raise ValueError("k must be at least 1.")
    boxes = list(proposed_boxes or [])
    specs = []
    for index in range(k):
        specs.append(
            {
                "temperature": DEFAULT_TEMPERATURES[index % len(DEFAULT_TEMPERATURES)],
                "seed": 1000 + index,
                "crop_box": list(boxes[index]) if index < len(boxes) else None,
            }
        )
    return specs


def policy_prompt(question: str) -> str:
    """Ask the policy for inspectable evidence plus a concise final answer."""
    return (
        "Answer the image question using visible evidence only. First write one short sentence "
        "describing the relevant evidence. Then write the final answer on its own line as "
        f"FINAL: <answer>\n\nQuestion: {question}"
    )


def generate_response(model, processor, image: Image.Image, question: str, temperature: float, seed: int) -> str:
    """Generate one policy response with an explicit decoding configuration."""
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    messages = build_messages(image, policy_prompt(question))
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    generation_args = {"max_new_tokens": 128}
    if temperature > 0:
        generation_args.update({"do_sample": True, "temperature": temperature, "top_p": 0.9})
    else:
        generation_args.update({"do_sample": False})
    output_ids = model.generate(**inputs, **generation_args)
    generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


def sample_candidates(
    model,
    processor,
    example: Dict[str, Any],
    k: int,
    crop_dir: Optional[Path] = None,
    proposed_boxes: Optional[Sequence[Sequence[float]]] = None,
) -> List[CandidateTrace]:
    """Generate, save, and label K candidate traces for one local image example."""
    image_path = example.get("image_path")
    if not image_path:
        raise ValueError("Approach 2 expects a normalized example with an image_path.")
    image_id = str(example.get("example_id") or Path(image_path).stem)
    question = str(example.get("question") or "")
    if not question:
        raise ValueError(f"Example {image_id} has no question.")

    from PIL import Image

    source_image = Image.open(image_path).convert("RGB")
    specs = default_candidate_specs(k, proposed_boxes)
    traces = []
    for index, spec in enumerate(specs):
        crop = crop_image(source_image, spec["crop_box"])
        crop_path = None
        if crop_dir is not None:
            crop_dir.mkdir(parents=True, exist_ok=True)
            destination = crop_dir / f"{image_id.replace(':', '_')}_k{index}.png"
            crop.save(destination)
            crop_path = str(destination)

        response = generate_response(
            model=model,
            processor=processor,
            image=crop,
            question=question,
            temperature=spec["temperature"],
            seed=spec["seed"],
        )
        prediction, rationale = extract_answer_and_rationale(response)
        traces.append(
            CandidateTrace(
                candidate_id=f"{image_id}:k{index}",
                image_id=image_id,
                question=question,
                prediction=prediction,
                raw_response=response,
                rationale=rationale,
                crop_box=spec["crop_box"],
                crop_path=crop_path,
                generation_temperature=spec["temperature"],
                generation_seed=spec["seed"],
                is_correct=answer_is_correct(prediction, example.get("answer")),
            )
        )
    return traces


def iter_pool_traces(records: Iterable[Dict[str, Any]]) -> Iterable[CandidateTrace]:
    """Yield traces from pool records for training or error analysis."""
    for record in records:
        for candidate in record.get("candidates", []):
            yield CandidateTrace.from_dict(candidate)
