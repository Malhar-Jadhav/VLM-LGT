# schema.py
#
# Defines the JSON-compatible records shared by candidate generation,
# verification, and evaluation for Approach 2. Keeping this schema in one
# place prevents each script from inventing a slightly different trace format.

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CandidateTrace:
    """One complete policy attempt for an image-question example.

    crop_box uses normalized (x1, y1, x2, y2) coordinates when it is present.
    is_correct is an evaluation label and must never be passed to a verifier at
    inference time.
    """

    candidate_id: str
    image_id: str
    question: str
    prediction: str
    raw_response: str
    rationale: str = ""
    crop_box: Optional[List[float]] = None
    crop_path: Optional[str] = None
    generation_temperature: float = 0.0
    generation_seed: Optional[int] = None
    generator_confidence: Optional[float] = None
    is_correct: Optional[bool] = None
    verifier_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-compatible representation of this trace."""
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "CandidateTrace":
        """Create a trace from a saved JSON record, tolerating extra fields."""
        allowed = {field_name for field_name in cls.__dataclass_fields__}
        return cls(**{key: item for key, item in value.items() if key in allowed})


def make_pool_record(example: Dict[str, Any], candidates: List[CandidateTrace]) -> Dict[str, Any]:
    """Attach serializable candidate traces to one normalized dataset example."""
    record = dict(example)
    record["candidates"] = [candidate.to_dict() for candidate in candidates]
    return record
