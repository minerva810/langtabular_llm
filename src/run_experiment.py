import json
from pathlib import Path

from serialize import serialize_table
from prompt_builder import build_prompt
from model_api import ModelConfig, get_model_client


BASE_DIR = Path(__file__).resolve().parent.parent
BENCHMARK_DIR = BASE_DIR / "benchmark_outputs"
PRED_DIR = BASE_DIR / "predictions"
PRED_DIR.mkdir(exist_ok=True, parents=True)


def load_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    return items


def save_jsonl(path, items):
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def build_table_map(tables):
    return {t["table_id"]: t for t in tables}


def run_language_experiment(
    tables,
    qas,
    language,
    serialization_method,
    model_client,
    output_path,
    limit=None
):
    table_map = build_table_map(tables)
    results = []

    count = 0
    for qa in qas:
        if limit is not None and count >= limit:
            break

        if language == "ko":
            question = qa["question_ko"]
            gold = qa["answer_ko"]
            qid = qa["qid_ko"]
        else:
            question = qa["question_en"]
            gold = qa["answer_en"]
            qid = qa["qid_en"]

        table = table_map[qa["table_id"]]
        table_text = serialize_table(table, method=serialization_method)
        prompt = build_prompt(
            table_text=table_text,
            question=question,
            language=language
        )

        pred = model_client.generate(prompt)

        results.append({
            "qid": qid,
            "table_id": qa["table_id"],
            "dataset_name": qa["dataset_name"],
            "language": language,
            "task_type": qa["task_type"],
            "serialization": serialization_method,
            "question": question,
            "gold": gold,
            "pred": pred,
            "prompt": prompt
        })
        count += 1

    save_jsonl(output_path, results)
    return results


def main():
    tables_ko = load_jsonl(BENCHMARK_DIR / "tables_ko.jsonl")
    tables_en = load_jsonl(BENCHMARK_DIR / "tables_en.jsonl")
    qas = load_jsonl(BENCHMARK_DIR / "qas_paired.jsonl")

    print("tables_ko:", len(tables_ko))
    print("tables_en:", len(tables_en))
    print("qas:", len(qas))

    config = ModelConfig(
        backend="openai",
        model_name="gpt-4.1-mini",
        temperature=0.0,
        max_tokens=64
    )

    print("Using backend:", config.backend)
    model_client = get_model_client(config)

    for serialization_method in ["markdown", "json", "kv"]:
        ko_out = PRED_DIR / f"pred_ko_{serialization_method}.jsonl"
        en_out = PRED_DIR / f"pred_en_{serialization_method}.jsonl"

        run_language_experiment(
            tables=tables_ko,
            qas=qas,
            language="ko",
            serialization_method=serialization_method,
            model_client=model_client,
            output_path=ko_out,
            limit=50
        )

        run_language_experiment(
            tables=tables_en,
            qas=qas,
            language="en",
            serialization_method=serialization_method,
            model_client=model_client,
            output_path=en_out,
            limit=50
        )

        print(f"[Done] {serialization_method}")


if __name__ == "__main__":
    main()
