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


def normalize_text(s):
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def try_parse_number(s):
    s = str(s).replace(",", "").strip()
    try:
        return float(s)
    except Exception:
        return None


def is_correct(gold, pred, tol=1e-6):
    gold_num = try_parse_number(gold)
    pred_num = try_parse_number(pred)

    if gold_num is not None and pred_num is not None:
        return math.isclose(gold_num, pred_num, rel_tol=tol, abs_tol=tol)

    return normalize_text(gold) == normalize_text(pred)


def evaluate_file(path):
    rows = load_jsonl(path)
    total = len(rows)
    correct = 0

    by_task = defaultdict(lambda: [0, 0])
    by_dataset = defaultdict(lambda: [0, 0])

    for r in rows:
        ok = is_correct(r["gold"], r["pred"])
        correct += int(ok)

        by_task[r["task_type"]][0] += int(ok)
        by_task[r["task_type"]][1] += 1

        by_dataset[r["dataset_name"]][0] += int(ok)
        by_dataset[r["dataset_name"]][1] += 1

    acc = correct / total if total > 0 else 0.0

    print(f"\n=== {path.name} ===")
    print(f"Overall Accuracy: {acc:.4f} ({correct}/{total})")

    print("\nBy Task")
    for k, (c, n) in sorted(by_task.items()):
        print(f"{k}: {c/n:.4f} ({c}/{n})")

    print("\nBy Dataset")
    for k, (c, n) in sorted(by_dataset.items()):
        print(f"{k}: {c/n:.4f} ({c}/{n})")

    return acc


def main():
    files = sorted(PRED_DIR.glob("*.jsonl"))
    if not files:
        print("predictions 폴더에 예측 파일이 없습니다.")
        return

    summary = {}
    for path in files:
        acc = evaluate_file(path)
        summary[path.name] = acc

    print("\n=== Summary ===")
    for name, acc in summary.items():
        print(f"{name}: {acc:.4f}")

    serializations = ["markdown", "json", "kv"]
    print("\n=== EN-KO Gap ===")
    for s in serializations:
        ko_name = f"pred_ko_{s}.jsonl"
        en_name = f"pred_en_{s}.jsonl"
        if ko_name in summary and en_name in summary:
            gap = summary[en_name] - summary[ko_name]
            print(f"{s}: EN-KO gap = {gap:.4f}")


if __name__ == "__main__":
    main()