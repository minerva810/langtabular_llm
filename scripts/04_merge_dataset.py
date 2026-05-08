import json
from collections import Counter
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]

META_DIR = BASE / "dataset" / "metadata"
QUESTION_DIR = BASE / "dataset" / "questions"

META_FILES = [
    META_DIR / "controlled_synthetic_metadata.jsonl",
    META_DIR / "real_metadata.jsonl",
    META_DIR / "stress_metadata.jsonl",
]

QUESTION_FILES = [
    QUESTION_DIR / "controlled_synthetic_questions.jsonl",
    QUESTION_DIR / "real_questions.jsonl",
    QUESTION_DIR / "stress_questions.jsonl",
]

OUT_META = META_DIR / "table_metadata.jsonl"
OUT_QUESTIONS = QUESTION_DIR / "base_questions.jsonl"


def load_jsonl(path):
    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")

    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def save_jsonl(items, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def validate_unique(items, key):
    values = [item[key] for item in items]
    duplicated = [k for k, v in Counter(values).items() if v > 1]

    if duplicated:
        raise ValueError(f"Duplicate {key} values found: {duplicated}")


def validate_required_keys(items, required_keys, item_name):
    for item in items:
        missing = [key for key in required_keys if key not in item]
        if missing:
            raise ValueError(f"{item_name} is missing keys {missing}: {item}")


def main():
    all_metadata = []
    all_questions = []

    for path in META_FILES:
        all_metadata.extend(load_jsonl(path))

    for path in QUESTION_FILES:
        all_questions.extend(load_jsonl(path))

    validate_required_keys(
        all_metadata,
        ["table_id", "split_type", "table_en_path", "table_ko_path", "column_mapping"],
        "metadata item",
    )
    validate_required_keys(
        all_questions,
        ["question_id", "table_id", "question", "gold_answer", "task_type", "data_type"],
        "question item",
    )

    validate_unique(all_metadata, "table_id")
    validate_unique(all_questions, "question_id")

    table_ids = {m["table_id"] for m in all_metadata}
    for q in all_questions:
        if q["table_id"] not in table_ids:
            raise ValueError(
                f"question_id={q['question_id']} has no metadata table_id: {q['table_id']}"
            )

    save_jsonl(all_metadata, OUT_META)
    save_jsonl(all_questions, OUT_QUESTIONS)

    print(f"[merged] tables: {len(all_metadata)}")
    print(f"[merged] questions: {len(all_questions)}")

    split_counter = Counter(m["split_type"] for m in all_metadata)
    print("[table split distribution]")
    for k, v in split_counter.items():
        print(f"- {k}: {v}")


if __name__ == "__main__":
    main()
