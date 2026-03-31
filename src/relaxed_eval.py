import json
import math
import re
from pathlib import Path
from collections import defaultdict


BASE_DIR = Path(__file__).resolve().parent.parent
PRED_DIR = BASE_DIR / "predictions"


def load_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    return items


# =========================
# Common normalization
# =========================

def normalize_text(s):
    s = str(s).lower().strip()
    s = s.replace(",", "")
    s = re.sub(r"\s+", " ", s)
    return s


def try_parse_number_strict(s):
    s = str(s).replace(",", "").strip()
    try:
        return float(s)
    except Exception:
        return None


def extract_number_relaxed(s):
    s = str(s).replace(",", "")
    nums = re.findall(r"-?\d+\.?\d*", s)
    if not nums:
        return None
    return float(nums[0])


def strip_unit(s):
    return re.sub(r"(명|건|개|%|people|cases|units)", "", str(s))


# =========================
# Strict / Relaxed scorer
# =========================

def is_correct_strict(gold, pred, tol=1e-6):
    gold_num = try_parse_number_strict(gold)
    pred_num = try_parse_number_strict(pred)

    if gold_num is not None and pred_num is not None:
        return math.isclose(gold_num, pred_num, rel_tol=tol, abs_tol=tol)

    return normalize_text(gold) == normalize_text(pred)


def is_correct_relaxed(gold, pred, tol=1e-3):
    gold = strip_unit(gold)
    pred = strip_unit(pred)

    gold_num = extract_number_relaxed(gold)
    pred_num = extract_number_relaxed(pred)

    if gold_num is not None and pred_num is not None:
        return math.isclose(gold_num, pred_num, rel_tol=tol, abs_tol=tol)

    g = normalize_text(gold)
    p = normalize_text(pred)

    return g == p or g in p or p in g


def get_checker(mode="strict"):
    if mode == "strict":
        return is_correct_strict
    elif mode == "relaxed":
        return is_correct_relaxed
    else:
        raise ValueError(f"Unknown mode: {mode}")


# =========================
# Evaluation
# =========================

def evaluate_file(path, mode="strict"):
    rows = load_jsonl(path)
    total = len(rows)
    correct = 0

    by_task = defaultdict(lambda: [0, 0])
    by_dataset = defaultdict(lambda: [0, 0])

    checker = get_checker(mode)

    for r in rows:
        ok = checker(r["gold"], r["pred"])
        correct += int(ok)

        by_task[r["task_type"]][0] += int(ok)
        by_task[r["task_type"]][1] += 1

        by_dataset[r["dataset_name"]][0] += int(ok)
        by_dataset[r["dataset_name"]][1] += 1

    acc = correct / total if total > 0 else 0.0

    print(f"\n=== {path.name} ({mode}) ===")
    print(f"Overall Accuracy: {acc:.4f} ({correct}/{total})")

    print("\nBy Task")
    for k, (c, n) in sorted(by_task.items()):
        print(f"{k}: {c/n:.4f} ({c}/{n})")

    print("\nBy Dataset")
    for k, (c, n) in sorted(by_dataset.items()):
        print(f"{k}: {c/n:.4f} ({c}/{n})")

    return acc


# =========================
# Aggregation analysis
# =========================

def classify_aggregation_error(gold, pred):
    """
    Returns:
      - correct
      - format
      - unit
      - rounding
      - wrong
    """
    if is_correct_relaxed(gold, pred):
        return "correct"

    gold_num = extract_number_relaxed(gold)
    pred_num = extract_number_relaxed(pred)

    pred_str = str(pred)

    if gold_num is not None and pred_num is not None:
        if abs(gold_num - pred_num) < 1e-1:
            return "rounding"
        return "wrong"

    if any(unit in pred_str for unit in ["명", "건", "개", "%", "people", "cases", "units"]):
        return "unit"

    return "format"


def analyze_aggregation_errors(path):
    rows = load_jsonl(path)

    total = 0
    correct = 0
    error_types = {
        "format": 0,
        "unit": 0,
        "rounding": 0,
        "wrong": 0
    }

    samples = {
        "format": [],
        "unit": [],
        "rounding": [],
        "wrong": []
    }

    for r in rows:
        if r["task_type"] != "aggregation":
            continue

        total += 1
        gold = r["gold"]
        pred = r["pred"]

        label = classify_aggregation_error(gold, pred)

        if label == "correct":
            correct += 1
        else:
            error_types[label] += 1
            if len(samples[label]) < 3:
                samples[label].append({
                    "question": r["question"],
                    "gold": gold,
                    "pred": pred,
                    "dataset_name": r["dataset_name"],
                    "language": r["language"],
                    "serialization": r["serialization"],
                })

    print(f"\n=== Aggregation Analysis: {path.name} ===")
    print(f"Relaxed Accuracy: {correct / total if total else 0:.4f} ({correct}/{total})")
    print("Error Types:")
    for k, v in error_types.items():
        print(f"  {k}: {v}")

    print("\nSample Errors")
    for err_type, items in samples.items():
        if not items:
            continue
        print(f"\n[{err_type}]")
        for item in items:
            print(f"- dataset={item['dataset_name']}, lang={item['language']}, ser={item['serialization']}")
            print(f"  Q: {item['question']}")
            print(f"  GOLD: {item['gold']}")
            print(f"  PRED: {item['pred']}")


# =========================
# Summary / Gap
# =========================

def summarize_all_files(mode="strict"):
    files = sorted(PRED_DIR.glob("*.jsonl"))
    if not files:
        print("predictions 폴더에 예측 파일이 없습니다.")
        return {}

    summary = {}
    for path in files:
        acc = evaluate_file(path, mode=mode)
        summary[path.name] = acc

    print(f"\n=== Summary ({mode}) ===")
    for name, acc in summary.items():
        print(f"{name}: {acc:.4f}")

    serializations = ["markdown", "json", "kv"]
    print(f"\n=== EN-KO Gap ({mode}) ===")
    for s in serializations:
        ko_name = f"pred_ko_{s}.jsonl"
        en_name = f"pred_en_{s}.jsonl"
        if ko_name in summary and en_name in summary:
            gap = summary[en_name] - summary[ko_name]
            print(f"{s}: EN-KO gap = {gap:.4f}")

    return summary


def main():
    print("######## STRICT EVALUATION ########")
    summarize_all_files(mode="strict")

    print("\n\n######## RELAXED EVALUATION ########")
    summarize_all_files(mode="relaxed")

    files = sorted(PRED_DIR.glob("*.jsonl"))
    for path in files:
        analyze_aggregation_errors(path)


if __name__ == "__main__":
    main()