import json
from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parents[1]

OUT_TABLE_DIR = BASE / "dataset" / "tables" / "stress"
OUT_META = BASE / "dataset" / "metadata" / "stress_metadata.jsonl"
OUT_Q = BASE / "dataset" / "questions" / "stress_questions.jsonl"

ROWS_PER_TABLE = 35


def save_jsonl(items, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def save_pair(table_id, df_en, df_ko, data_type):
    table_dir = OUT_TABLE_DIR / data_type
    table_dir.mkdir(parents=True, exist_ok=True)

    en_path = table_dir / f"{table_id}_en.csv"
    ko_path = table_dir / f"{table_id}_ko.csv"

    df_en.to_csv(en_path, index=False, encoding="utf-8-sig")
    df_ko.to_csv(ko_path, index=False, encoding="utf-8-sig")

    return str(en_path), str(ko_path)


def main():
    metadata = []
    questions = []

    table_id = "stress_ambiguous_score_001"
    column_mapping = {
        "Name": "이름",
        "Score": "점수",
        "Final Score": "최종 점수",
        "Score Change": "점수 변화",
    }

    names = [f"Student {i:02d}" for i in range(1, ROWS_PER_TABLE + 1)]
    scores = [62 + ((i * 7) % 35) for i in range(ROWS_PER_TABLE)]
    final_scores = [score + ((i * 5) % 11) - 3 for i, score in enumerate(scores)]
    score_changes = [final - score for score, final in zip(scores, final_scores)]

    df_en = pd.DataFrame(
        {
            "Name": names,
            "Score": scores,
            "Final Score": final_scores,
            "Score Change": score_changes,
        }
    )
    df_ko = df_en.rename(columns=column_mapping)

    en_path, ko_path = save_pair(table_id, df_en, df_ko, "numeric")
    max_final_row = df_en.loc[df_en["Final Score"].idxmax()]
    target_row = df_en.iloc[9]

    metadata.append(
        {
            "table_id": table_id,
            "split_type": "stress",
            "table_en_path": en_path,
            "table_ko_path": ko_path,
            "num_rows": ROWS_PER_TABLE,
            "num_cols": 4,
            "data_type": "numeric",
            "difficulty": "hard",
            "stress_type": "ambiguous_header",
            "column_mapping": column_mapping,
            "paired_guarantee": True,
        }
    )

    questions.extend(
        [
            {
                "question_id": f"{table_id}_q1",
                "table_id": table_id,
                "question": "Who has the highest final score?",
                "gold_answer": str(max_final_row["Name"]),
                "evidence": {"column": "Final Score", "operation": "max"},
                "task_type": "comparison",
                "data_type": "numeric",
                "difficulty": "hard",
            },
            {
                "question_id": f"{table_id}_q2",
                "table_id": table_id,
                "question": f"What is the score change of {target_row['Name']}?",
                "gold_answer": str(target_row["Score Change"]),
                "evidence": {"row": target_row["Name"], "column": "Score Change"},
                "task_type": "retrieval",
                "data_type": "numeric",
                "difficulty": "hard",
            },
        ]
    )

    save_jsonl(metadata, OUT_META)
    save_jsonl(questions, OUT_Q)

    print(f"[stress] tables: {len(metadata)}, questions: {len(questions)}")


if __name__ == "__main__":
    main()
