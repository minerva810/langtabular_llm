import argparse
import json
import os
import re
import time
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]

README_INPUT_PATH = BASE / "dataset" / "serialized" / "eval_instances.jsonl"
FALLBACK_INPUT_PATH = BASE / "dataset" / "eval" / "eval_instances.jsonl"
DEFAULT_OUTPUT_PATH = BASE / "experiments" / "predictions" / "predictions.jsonl"
DEFAULT_LOG_PATH = BASE / "experiments" / "logs" / "inference.log"
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

SYSTEM_PROMPT = (
    "You are a careful table question answering system. "
    "Answer using only the provided table. "
    "Return exactly one short final answer inside <answer>...</answer>. "
    "Do not include explanation."
)

USER_PROMPT_TEMPLATE = """Table language: {table_language}
Serialization: {serialization}

Table:
{serialized_table}

Question:
{question}

Return exactly one short final answer inside <answer>...</answer>."""


def load_jsonl(path):
    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def append_jsonl(item, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def default_input_path():
    if README_INPUT_PATH.exists():
        return README_INPUT_PATH
    return FALLBACK_INPUT_PATH


def append_log(message, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")


def load_completed_instance_ids(path):
    if not path.exists():
        return set()

    completed = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("status") == "ok" and item.get("instance_id"):
                completed.add(item["instance_id"])
    return completed


def build_user_prompt(instance):
    return USER_PROMPT_TEMPLATE.format(
        table_language=instance["table_language"],
        serialization=instance["serialization"],
        serialized_table=instance["serialized_table"],
        question=instance["question"],
    )


def extract_answer(raw_text):
    if raw_text is None:
        return ""

    text = str(raw_text).strip()
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""

    first = lines[0]
    first = re.sub(r"^(final answer|answer)\s*:\s*", "", first, flags=re.IGNORECASE)
    return first.strip().strip('"').strip("'")


def make_result(instance, model, raw_response, latency_sec, status="ok", error=None):
    return {
        "instance_id": instance["instance_id"],
        "question_id": instance["question_id"],
        "table_id": instance["table_id"],
        "model": model,
        "split_type": instance["split_type"],
        "source": instance["source"],
        "table_language": instance["table_language"],
        "serialization": instance["serialization"],
        "task_type": instance["task_type"],
        "data_type": instance["data_type"],
        "difficulty": instance.get("difficulty"),
        "scenario": instance.get("scenario"),
        "failure_type": instance.get("failure_type"),
        "question": instance["question"],
        "gold_answer": instance["gold_answer"],
        "canonical_gold_answer": instance.get("canonical_gold_answer"),
        "all_gold_answers": instance.get("all_gold_answers"),
        "predicted_answer": extract_answer(raw_response),
        "raw_response": raw_response,
        "status": status,
        "error": error,
        "latency_sec": round(latency_sec, 3),
    }


def create_openai_client():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install the OpenAI Python package first: pip install openai") from exc

    return OpenAI()


def response_text(response):
    output_text = getattr(response, "output_text", None)
    if output_text:
        return output_text

    try:
        return response.output[0].content[0].text
    except Exception:
        return str(response)


def call_openai(client, model, system_prompt, user_prompt, max_output_tokens):
    response = client.responses.create(
        model=model,
        temperature=0,
        max_output_tokens=max_output_tokens,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response_text(response)


def run_one(client, instance, model, max_output_tokens, max_retries, retry_sleep):
    user_prompt = build_user_prompt(instance)

    for attempt in range(max_retries + 1):
        start = time.perf_counter()
        try:
            raw_response = call_openai(
                client=client,
                model=model,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                max_output_tokens=max_output_tokens,
            )
            latency_sec = time.perf_counter() - start
            return make_result(instance, model, raw_response, latency_sec)
        except Exception as exc:
            if attempt >= max_retries:
                latency_sec = time.perf_counter() - start
                return make_result(
                    instance,
                    model,
                    raw_response=None,
                    latency_sec=latency_sec,
                    status="error",
                    error=f"{type(exc).__name__}: {exc}",
                )
            time.sleep(retry_sleep * (2**attempt))

    raise RuntimeError("Unreachable retry state")


def parse_args():
    parser = argparse.ArgumentParser(description="Run LLM inference over table QA eval instances.")
    parser.add_argument("--input", type=Path, default=default_input_path())
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--split-type", default=None)
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--table-language", choices=["en", "ko"], default=None)
    parser.add_argument("--serialization", choices=["markdown", "json", "kv"], default=None)
    parser.add_argument("--max-output-tokens", type=int, default=64)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    run_start = time.perf_counter()
    args = parse_args()
    completed = load_completed_instance_ids(args.output) if args.resume else set()
    client = None if args.dry_run else create_openai_client()

    append_log(
        (
            f"start input={args.input} output={args.output} model={args.model} "
            f"limit={args.limit} offset={args.offset} split_type={args.split_type} "
            f"scenario={args.scenario} table_language={args.table_language} "
            f"serialization={args.serialization} resume={args.resume} dry_run={args.dry_run}"
        ),
        args.log,
    )

    seen = 0
    written = 0
    skipped = 0

    for instance in load_jsonl(args.input):
        if args.split_type and instance.get("split_type") != args.split_type:
            continue
        if args.scenario and instance.get("scenario") != args.scenario:
            continue
        if args.table_language and instance.get("table_language") != args.table_language:
            continue
        if args.serialization and instance.get("serialization") != args.serialization:
            continue

        if seen < args.offset:
            seen += 1
            continue
        seen += 1

        if args.resume and instance["instance_id"] in completed:
            skipped += 1
            continue

        if args.limit is not None and written >= args.limit:
            break

        if args.dry_run:
            result = {
                "instance_id": instance["instance_id"],
                "model": args.model,
                "status": "dry_run",
                "prompt": {
                    "system": SYSTEM_PROMPT,
                    "user": build_user_prompt(instance),
                },
            }
        else:
            result = run_one(
                client=client,
                instance=instance,
                model=args.model,
                max_output_tokens=args.max_output_tokens,
                max_retries=args.max_retries,
                retry_sleep=args.retry_sleep,
            )

        append_jsonl(result, args.output)
        written += 1

        if written % 100 == 0:
            progress = f"written={written}, skipped={skipped}, latest={instance['instance_id']}"
            print(f"[progress] {progress}")
            append_log(f"progress {progress}", args.log)

    print("[inference] done")
    print(f"Output: {args.output}")
    print(f"Written: {written}")
    print(f"Skipped: {skipped}")
    elapsed_sec = time.perf_counter() - run_start
    print(f"Elapsed seconds: {elapsed_sec:.3f}")
    append_log(f"done written={written} skipped={skipped} elapsed_sec={elapsed_sec:.3f}", args.log)


if __name__ == "__main__":
    main()
