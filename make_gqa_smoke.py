# Builds a tiny GQA file for the Approach 2 smoke run.
# lmms-lab/GQA keeps questions and images in separate configs, so this joins them.
import argparse, os
from datasets import load_dataset
from data.common.utils import write_jsonl

p = argparse.ArgumentParser()
p.add_argument("--limit", type=int, default=20)
p.add_argument("--subset", default="testdev_balanced")
a = p.parse_args()

def first_split(ds):
    return next(iter(ds.values()))

qs = first_split(load_dataset("lmms-lab/GQA", f"{a.subset}_instructions"))
ims = first_split(load_dataset("lmms-lab/GQA", f"{a.subset}_images"))
print("question columns:", qs.column_names)
print("image columns:", ims.column_names)

qs = qs.select(range(min(a.limit, len(qs))))
need = set(qs["imageId"])
id_to_row = {i: r for r, i in enumerate(ims["id"]) if i in need}

base = "data/approach2_verifier_reranking/processed"
img_dir = os.path.join(base, "images")
os.makedirs(img_dir, exist_ok=True)
records = []
for idx, q in enumerate(qs):
    path = os.path.join(img_dir, f"gqa_{q['imageId']}.png")
    if not os.path.exists(path):
        ims[id_to_row[q["imageId"]]]["image"].convert("RGB").save(path)
    records.append({
        "source": "gqa", "example_id": f"gqa:{a.subset}:{idx}",
        "question": q["question"], "answer": q["answer"], "bbox": None,
        "image_path": os.path.abspath(path),
    })
write_jsonl(records, os.path.join(base, "gqa_train.jsonl"))
print(f"wrote {len(records)} examples")
