import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]

DEFAULT_INPUT_PATH = BASE / "experiments" / "predictions" / "controlled_synthetic_predictions.jsonl"
DEFAULT_OUTPUT_PATH = BASE / "experiments" / "results" / "controlled_synthetic_evaluation.jsonl"
DEFAULT_SUMMARY_PATH = BASE / "experiments" / "results" / "controlled_synthetic_summary.json"


def load_jsonl(path):
    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def save_jsonl(items, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def save_json(item, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(item, f, ensure_ascii=False, indent=2)


def normalize_text(value):
    text = "" if value is None else str(value)
    text = text.strip().lower()
    text = re.sub(r"^<answer>\s*|\s*</answer>$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .,:;\"'")


def parse_number(value):
    if value is None:
        return None

    text = str(value).replace(",", "").strip()
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None

    try:
        return float(match.group(0))
    except ValueError:
        return None


def split_answers(value):
    if isinstance(value, list):
        return [str(item) for item in value]
    if value is None:
        return []
    return [part.strip() for part in str(value).split("|") if part.strip()]


def score_prediction(predicted, gold, all_gold_answers=None, numeric_tolerance=1e-3):
    candidates = split_answers(all_gold_answers) or split_answers(gold)
    pred_norm = normalize_text(predicted)

    for candidate in candidates:
        gold_norm = normalize_text(candidate)
        if pred_norm == gold_norm:
            return "exact_match", True

        pred_num = parse_number(predicted)
        gold_num = parse_number(candidate)
        if pred_num is not None and gold_num is not None:
            if math.isclose(pred_num, gold_num, rel_tol=0.0, abs_tol=numeric_tolerance):
                return "numeric_tolerance", True

    return "incorrect", False


def aggregate(rows, keys):
    buckets = defaultdict(lambda: {"total": 0, "correct": 0})

    for row in rows:
        key = tuple(row.get(k) for k in keys)
        buckets[key]["total"] += 1
        buckets[key]["correct"] += int(row["is_correct"])

    output = []
    for key, stats in sorted(buckets.items()):
        total = stats["total"]
        correct = stats["correct"]
        item = {k: v for k, v in zip(keys, key)}
        item.update(
            {
                "total": total,
                "correct": correct,
                "accuracy": round(correct / total, 4) if total else 0.0,
            }
        )
        output.append(item)
    return output


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate inference predictions.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--split-type", default="controlled_synthetic")
    parser.add_argument("--numeric-tolerance", type=float, default=1e-3)
    return parser.parse_args()


def main():
    args = parse_args()
    evaluated = []

    for item in load_jsonl(args.input):
        if args.split_type and item.get("split_type") != args.split_type:
            continue

        score_type, is_correct = score_prediction(
            item.get("predicted_answer"),
            item.get("gold_answer"),
            item.get("all_gold_answers"),
            numeric_tolerance=args.numeric_tolerance,
        )
        evaluated.append(
            {
                **item,
                "score_type": score_type,
                "is_correct": is_correct,
            }
        )

    save_jsonl(evaluated, args.output)

    total = len(evaluated)
    correct = sum(1 for row in evaluated if row["is_correct"])
    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "split_type": args.split_type,
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "score_type_counts": dict(Counter(row["score_type"] for row in evaluated)),
        "by_table_language": aggregate(evaluated, ["table_language"]),
        "by_serialization": aggregate(evaluated, ["serialization"]),
        "by_language_serialization": aggregate(evaluated, ["table_language", "serialization"]),
        "by_scenario": aggregate(evaluated, ["scenario"]),
        "by_scenario_language": aggregate(evaluated, ["scenario", "table_language"]),
        "by_scenario_language_serialization": aggregate(evaluated, ["scenario", "table_language", "serialization"]),
        "by_failure_type": aggregate(evaluated, ["failure_type"]),
    }
    save_json(summary, args.summary)

    print("[evaluation] done")
    print(f"Rows: {total}")
    print(f"Accuracy: {summary['accuracy']}")
    print(f"Output: {args.output}")
    print(f"Summary: {args.summary}")


if __name__ == "__main__":
    main()
