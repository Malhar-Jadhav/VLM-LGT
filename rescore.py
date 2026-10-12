import json, re, sys
from collections import Counter
from data.approach2_verifier_reranking.candidates import normalize_answer
from data.approach2_verifier_reranking.evaluation import evaluate_candidate_groups
from data.approach2_verifier_reranking.schema import CandidateTrace

def parse(raw):
    m = re.search(r"FINAL\s*:\s*(.+)", raw, flags=re.IGNORECASE)  # FINAL anywhere, not just line start
    return (m.group(1).strip().splitlines()[0] if m else raw.strip())

def strict(pred, ref):
    return normalize_answer(pred) == normalize_answer(ref)

def lenient(pred, ref):
    p, r = normalize_answer(pred), normalize_answer(ref)
    if p == r:
        return True
    if r in ("yes", "no"):  # yes/no: only count if the prediction starts with it
        return p.split()[:1] == [r]
    return re.search(rf"\b{re.escape(r)}\b", p) is not None

pool = sys.argv[1]
records = [json.loads(l) for l in open(pool) if l.strip()]
results = {}
for name, match in (("strict_reparsed", strict), ("lenient", lenient)):
    groups, mv = [], 0
    for r in records:
        g = []
        for c in r["candidates"]:
            t = CandidateTrace.from_dict(c)
            t.prediction = parse(t.raw_response)
            t.is_correct = match(t.prediction, r.get("answer"))
            g.append(t)
        groups.append(g)
        top = Counter(normalize_answer(t.prediction) for t in g).most_common(1)[0][0]
        mv += any(t.is_correct for t in g if normalize_answer(t.prediction) == top)
    m = evaluate_candidate_groups(groups).to_dict()
    m["majority_vote_accuracy"] = mv / len(groups)
    results[name] = m
print(json.dumps(results, indent=2))
