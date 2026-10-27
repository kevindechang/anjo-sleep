"""Build a deterministic 200-item multiple-choice general-skills set.

Stdlib only. Fetches public rows from the Hugging Face datasets-server API.
Run from any directory: python3 -I build_general.py
"""
import json
import os
import random
import re
import time
import urllib.parse
import urllib.request

SEED = 20261027
API = "https://datasets-server.huggingface.co/rows"
PAGE = 100
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "general.jsonl")

# (dataset, config, split, pool_rows or None for the whole split, sample size)
MMLU = ("cais/mmlu", "all", "test", None, 80)
ARC = ("allenai/ai2_arc", "ARC-Challenge", "test", None, 60)
HELLASWAG = ("Rowan/hellaswag", "default", "validation", 1000, 60)
LETTER_FOR_DIGIT = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}


def fetch_rows(dataset, config, split, total=None):
    """Return the row dicts of a split, paging 100 at a time."""
    rows, offset = [], 0
    while total is None or offset < total:
        query = urllib.parse.urlencode({
            "dataset": dataset, "config": config, "split": split,
            "offset": offset, "length": PAGE,
        })
        data = get_json(f"{API}?{query}")
        if total is None:
            total = data["num_rows_total"]
        page = [item["row"] for item in data["rows"]]
        if not page:
            break
        rows.extend(page)
        offset += len(page)
        print(f"  {dataset} [{config}/{split}] {offset}/{total}", flush=True)
        time.sleep(0.5)
    return rows[:total]


def get_json(url, attempts=8):
    """GET a JSON document, backing off on rate limits (429) and errors."""
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                return json.load(resp)
        except OSError:
            if attempt == attempts:
                raise
            time.sleep(min(60, 5 * attempt))


def clean_hellaswag(text):
    # Same light preprocessing as the lm-evaluation-harness HellaSwag task.
    text = text.strip().replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def normalize_key(key):
    key = str(key).strip()
    return LETTER_FOR_DIGIT.get(key, key)


def to_items_mmlu(rows):
    for row in rows:
        yield {
            "source": "mmlu",
            "subject": row["subject"],
            "question": row["question"].strip(),
            "choices": [str(c).strip() for c in row["choices"]],
            "answer": int(row["answer"]),
        }


def to_items_arc(rows):
    for row in rows:
        labels = [normalize_key(x) for x in row["choices"]["label"]]
        key = normalize_key(row["answerKey"])
        yield {
            "source": "arc_challenge",
            "subject": "science",
            "question": row["question"].strip(),
            "choices": [str(t).strip() for t in row["choices"]["text"]],
            "answer": labels.index(key),
        }


def to_items_hellaswag(rows):
    for row in rows:
        yield {
            "source": "hellaswag",
            "subject": row["activity_label"].strip(),
            "question": clean_hellaswag(row["ctx"]),
            "choices": [clean_hellaswag(e) for e in row["endings"]],
            "answer": int(row["label"]),
        }


def sample(pool, size, rng):
    if len(pool) < size:
        raise SystemExit(f"pool has {len(pool)} rows, need {size}")
    return rng.sample(pool, size)


def load(spec, convert):
    dataset, config, split, pool_size, size = spec
    rows = fetch_rows(dataset, config, split, total=pool_size)
    return rows, convert(rows)


def main():
    rng = random.Random(SEED)
    # Each source samples from its own pool, in a fixed order, from one RNG.
    items = []
    for spec, convert in ((MMLU, to_items_mmlu), (ARC, to_items_arc),
                          (HELLASWAG, to_items_hellaswag)):
        _, converted = load(spec, convert)
        pool = list(converted)
        items.extend(sample(pool, spec[4], rng))
    rng.shuffle(items)  # interleave sources deterministically

    with open(OUT, "w", encoding="utf-8") as out:
        for n, item in enumerate(items, start=1):
            record = {"id": f"g{n:03d}", **item}
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"wrote {len(items)} items to {OUT}")


if __name__ == "__main__":
    main()
