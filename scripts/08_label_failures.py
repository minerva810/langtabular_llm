import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]

DEFAULT_INPUT_PATH = BASE / "experiments" / "results" / "controlled_synthetic_evaluation.jsonl"
DEFAULT_OUTPUT_PATH = BASE / "experiments" / "results" / "controlled_synthetic_failure_labels.jsonl"
DEFAULT_SUMMARY_PATH = BASE / "experiments" / "results" / "controlled_synthetic_failure_summary.json"


FAILURE_TYPE_TO_LABEL = {
    "header_grounding": "Header grounding",
    "row_selection": "Row selection",
    "filtering_comparison": "Filtering / comparison",
    "arithmetic_aggregation": "Arithmetic / aggregation",
    "serialization_parsing": "Serialization parsing",
    "language_specific_linking": "Language-specific linking",
    "instruction_output_error": "Instruction / output error",
}


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


def infer_failure_label(row):
    raw_response = row.get("raw_response")
    predicted_answer = str(row.get("predicted_answer") or "").strip()

    if row.get("status") != "ok":
        return "Instruction / output error"
    if raw_response is None or not predicted_answer:
        return "Instruction / output error"

    failure_type = row.get("failure_type")
    if failure_type:
        return FAILURE_TYPE_TO_LABEL.get(failure_type, failure_type)

    task_type = row.get("task_type")
    if task_type in {"aggregation"}:
        return "Arithmetic / aggregation"
    if task_type in {"filtering", "comparison", "filtering_comparison"}:
        return "Filtering / comparison"

    return "Uncategorized"


def summarize(rows):
    label_counts = Counter(row["failure_label"] for row in rows)
    by_language = defaultdict(Counter)
    by_serialization = defaultdict(Counter)
    by_scenario = defaultdict(Counter)

    for row in rows:
        label = row["failure_label"]
        by_language[row.get("table_language")][label] += 1
        by_serialization[row.get("serialization")][label] += 1
        by_scenario[row.get("scenario")][label] += 1

    return {
        "total_failures": len(rows),
        "failure_counts": dict(label_counts),
        "by_table_language": {str(k): dict(v) for k, v in by_language.items()},
        "by_serialization": {str(k): dict(v) for k, v in by_serialization.items()},
        "by_scenario": {str(k): dict(v) for k, v in by_scenario.items()},
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Label failure types for incorrect predictions.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--include-correct", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    labeled = []

    for row in load_jsonl(args.input):
        is_correct = bool(row.get("is_correct"))
        if is_correct and not args.include_correct:
            continue

        label = "Correct" if is_correct else infer_failure_label(row)
        labeled.append({**row, "failure_label": label})

    save_jsonl(labeled, args.output)
    save_json(summarize(labeled), args.summary)

    print("[failure_labeling] done")
    print(f"Rows: {len(labeled)}")
    print(f"Output: {args.output}")
    print(f"Summary: {args.summary}")


if __name__ == "__main__":
    main()
