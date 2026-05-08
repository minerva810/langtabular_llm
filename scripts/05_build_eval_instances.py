import json
from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parents[1]

META_PATH = BASE / "dataset" / "metadata" / "table_metadata.jsonl"
QUESTION_PATH = BASE / "dataset" / "questions" / "base_questions.jsonl"
OUT_PATH = BASE / "dataset" / "eval" / "eval_instances.jsonl"

TABLE_LANGUAGES = ["en", "ko"]
SERIALIZATIONS = ["markdown", "json", "kv"]


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


def read_table(path):
    if not path:
        raise ValueError("Empty table path")

    table_path = Path(path)
    if not table_path.exists():
        raise FileNotFoundError(f"Table CSV does not exist: {table_path}")

    return pd.read_csv(table_path, dtype=str, keep_default_na=False)


def format_cell(value):
    return str(value).replace("\n", " ").strip()


def serialize_markdown(df):
    columns = [format_cell(col) for col in df.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]

    for _, row in df.iterrows():
        values = [format_cell(row[col]) for col in df.columns]
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def serialize_json(df):
    records = []
    for _, row in df.iterrows():
        records.append({str(col): format_cell(row[col]) for col in df.columns})
    return json.dumps(records, ensure_ascii=False)


def serialize_kv(df):
    lines = []
    for row_idx, row in df.iterrows():
        pairs = [f"{format_cell(col)}={format_cell(row[col])}" for col in df.columns]
        lines.append(f"row {row_idx + 1}: " + " | ".join(pairs))
    return "\n".join(lines)


def serialize_table(df, serialization):
    if serialization == "markdown":
        return serialize_markdown(df)
    if serialization == "json":
        return serialize_json(df)
    if serialization == "kv":
        return serialize_kv(df)
    raise ValueError(f"Unsupported serialization: {serialization}")


def table_path_for_language(metadata, table_language):
    key = f"table_{table_language}_path"
    if key not in metadata:
        raise KeyError(f"Missing {key} for table_id={metadata.get('table_id')}")
    return metadata[key]


def gold_answer_for_language(question, table_language):
    if table_language == "ko" and "gold_answer_ko" in question:
        return str(question["gold_answer_ko"])
    return str(question["gold_answer"])


def build_instance(question, metadata, table_language, serialization, serialized_table):
    instance_id = (
        f"{question['question_id']}__"
        f"{table_language}__"
        f"{serialization}"
    )

    return {
        "instance_id": instance_id,
        "question_id": question["question_id"],
        "table_id": question["table_id"],
        "split_type": metadata["split_type"],
        "source": metadata.get("source", metadata["split_type"]),
        "table_language": table_language,
        "serialization": serialization,
        "question": question["question"],
        "gold_answer": gold_answer_for_language(question, table_language),
        "canonical_gold_answer": str(question["gold_answer"]),
        "all_gold_answers": question.get("all_gold_answers"),
        "task_type": question["task_type"],
        "data_type": question["data_type"],
        "difficulty": question.get("difficulty", metadata.get("difficulty")),
        "scenario": question.get("scenario", metadata.get("scenario")),
        "failure_type": question.get("failure_type"),
        "evidence": question.get("evidence", {}),
        "table_path": table_path_for_language(metadata, table_language),
        "num_rows": metadata.get("num_rows"),
        "num_cols": metadata.get("num_cols"),
        "serialized_table": serialized_table,
    }


def main():
    metadata_items = load_jsonl(META_PATH)
    question_items = load_jsonl(QUESTION_PATH)
    metadata_by_table_id = {item["table_id"]: item for item in metadata_items}

    serialized_cache = {}
    instances = []

    for question in question_items:
        table_id = question["table_id"]
        if table_id not in metadata_by_table_id:
            raise ValueError(f"No metadata found for table_id={table_id}")

        metadata = metadata_by_table_id[table_id]

        for table_language in TABLE_LANGUAGES:
            table_path = table_path_for_language(metadata, table_language)
            cache_key = (table_path, table_language)

            if cache_key not in serialized_cache:
                df = read_table(table_path)
                serialized_cache[cache_key] = {
                    serialization: serialize_table(df, serialization)
                    for serialization in SERIALIZATIONS
                }

            for serialization in SERIALIZATIONS:
                instances.append(
                    build_instance(
                        question,
                        metadata,
                        table_language,
                        serialization,
                        serialized_cache[cache_key][serialization],
                    )
                )

    save_jsonl(instances, OUT_PATH)

    print(f"[eval_instances] instances: {len(instances)}")
    print(f"Output: {OUT_PATH}")
    print("[dimensions]")
    print(f"- questions: {len(question_items)}")
    print(f"- table_languages: {len(TABLE_LANGUAGES)}")
    print(f"- serializations: {len(SERIALIZATIONS)}")


if __name__ == "__main__":
    main()
