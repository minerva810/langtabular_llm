import json
from pathlib import Path

import pandas as pd
from datasets import load_dataset


BASE = Path(__file__).resolve().parents[1]

OUT_TABLE_DIR = BASE / "dataset" / "tables" / "real" / "wikitablequestions"
OUT_QUESTION_PATH = BASE / "dataset" / "questions" / "real_questions.jsonl"
OUT_METADATA_PATH = BASE / "dataset" / "metadata" / "real_metadata.jsonl"

MAX_TABLES_PER_SPLIT = 100


def save_jsonl(items, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_wikitablequestions():
    for dataset_name in ["wikitablequestions", "Stanford/wikitablequestions"]:
        try:
            return load_dataset(dataset_name)
        except Exception as exc:
            last_error = exc

    raise RuntimeError("Failed to load WikitableQuestions dataset") from last_error


def make_unique_columns(columns):
    counts = {}
    unique_columns = []

    for column in columns:
        base = str(column).strip() or "column"
        counts[base] = counts.get(base, 0) + 1
        if counts[base] == 1:
            unique_columns.append(base)
        else:
            unique_columns.append(f"{base}_{counts[base]}")

    return unique_columns


def table_to_df(table):
    header = make_unique_columns(table["header"])
    rows = table["rows"]

    valid_rows = [row for row in rows if len(row) == len(header)]
    if len(header) < 2 or len(valid_rows) < 2:
        return None

    df = pd.DataFrame(valid_rows, columns=header)
    df = df.map(lambda x: str(x).strip() if pd.notna(x) else "")
    df = df.replace("", pd.NA)
    df = df.dropna(axis=0, how="all")
    df = df.dropna(axis=1, how="all")
    df = df.fillna("")

    if df.shape[0] < 2 or df.shape[1] < 2:
        return None

    return df


def build_korean_column_mapping(columns):
    return {column: f"열_{idx + 1}" for idx, column in enumerate(columns)}


def infer_data_type(df):
    numeric_cols = 0
    for column in df.columns:
        converted = pd.to_numeric(df[column], errors="coerce")
        if converted.notna().mean() >= 0.7:
            numeric_cols += 1

    if numeric_cols == 0:
        return "text"
    if numeric_cols == len(df.columns):
        return "numeric"
    return "mixed"


def normalize_answer(answers):
    if isinstance(answers, list):
        return " | ".join(str(answer) for answer in answers)
    return str(answers)


def save_pair(table_id, df_en, column_mapping):
    OUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)

    df_ko = df_en.rename(columns=column_mapping)
    en_path = OUT_TABLE_DIR / f"{table_id}_en.csv"
    ko_path = OUT_TABLE_DIR / f"{table_id}_ko.csv"

    df_en.to_csv(en_path, index=False, encoding="utf-8-sig")
    df_ko.to_csv(ko_path, index=False, encoding="utf-8-sig")

    return str(en_path), str(ko_path)


def main():
    OUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)

    ds = load_wikitablequestions()

    questions = []
    metadata = []
    skipped = 0

    for split in ["train", "validation", "test"]:
        saved_in_split = 0

        for idx, sample in enumerate(ds[split]):
            if saved_in_split >= MAX_TABLES_PER_SPLIT:
                break

            df_en = table_to_df(sample["table"])
            if df_en is None:
                skipped += 1
                continue

            table_id = f"real_wtq_{split}_{idx:05d}"
            column_mapping = build_korean_column_mapping(df_en.columns)
            en_path, ko_path = save_pair(table_id, df_en, column_mapping)
            data_type = infer_data_type(df_en)

            questions.append(
                {
                    "question_id": f"{table_id}_q1",
                    "table_id": table_id,
                    "question": sample["question"],
                    "gold_answer": normalize_answer(sample["answers"]),
                    "all_gold_answers": sample["answers"],
                    "evidence": {"source": "wikitablequestions"},
                    "task_type": "unknown",
                    "data_type": data_type,
                    "difficulty": "real",
                }
            )

            metadata.append(
                {
                    "table_id": table_id,
                    "split_type": "real",
                    "source": "wikitablequestions",
                    "split": split,
                    "original_id": sample["id"],
                    "original_table_name": sample["table"].get("name", f"{split}_{idx}"),
                    "table_en_path": en_path,
                    "table_ko_path": ko_path,
                    "num_rows": int(df_en.shape[0]),
                    "num_cols": int(df_en.shape[1]),
                    "columns": list(df_en.columns),
                    "column_mapping": column_mapping,
                    "domain": "wikipedia",
                    "data_type": data_type,
                    "paired_guarantee": True,
                }
            )

            saved_in_split += 1

    save_jsonl(questions, OUT_QUESTION_PATH)
    save_jsonl(metadata, OUT_METADATA_PATH)

    print("[real_wikitablequestions] done")
    print(f"Saved tables: {len(metadata)}")
    print(f"Skipped broken tables: {skipped}")
    print(f"Question file: {OUT_QUESTION_PATH}")
    print(f"Metadata file: {OUT_METADATA_PATH}")


if __name__ == "__main__":
    main()
