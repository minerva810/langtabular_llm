from pathlib import Path
import pandas as pd
import json
import random

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
QA_DIR = BASE_DIR / "data" / "qa"

QA_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------
# 1. QA templates
# ---------------------------

def gen_aggregation(df):
    text_cols = [c for c in df.columns if df[c].dtype == object]
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    if not text_cols or not num_cols:
        return None

    cat = random.choice(text_cols)
    metric = random.choice(num_cols)

    val = random.choice(df[cat].dropna().unique())
    sub = df[df[cat] == val]

    if len(sub) == 0:
        return None

    answer = float(sub[metric].mean())

    q_en = f"What is the average {metric} for {val}?"
    q_ko = f"{val}의 평균 {metric}은 무엇인가?"

    return q_en, q_ko, answer, "aggregation"


# ---------------------------
# 2. dataset loop
# ---------------------------

def process_dataset(path):
    df = pd.read_csv(path)

    samples = []

    for _ in range(10):  # dataset당 10개
        out = gen_aggregation(df)
        if out is None:
            continue

        q_en, q_ko, ans, task = out

        samples.append({
            "dataset": path.stem,
            "task": task,
            "lang": "en",
            "question": q_en,
            "answer": ans
        })

        samples.append({
            "dataset": path.stem,
            "task": task,
            "lang": "ko",
            "question": q_ko,
            "answer": ans
        })

    return samples


# ---------------------------
# 3. main
# ---------------------------

def main():
    all_samples = []

    for group in ["numeric", "text", "mixed"]:
        for path in (PROCESSED_DIR / group).glob("*_en.csv"):
            samples = process_dataset(path)
            all_samples.extend(samples)

    with open(QA_DIR / "qa_all.jsonl", "w", encoding="utf-8") as f:
        for s in all_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()