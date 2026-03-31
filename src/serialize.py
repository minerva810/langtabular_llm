import json
import pandas as pd


def to_markdown_table(columns, rows):
    df = pd.DataFrame(rows, columns=columns)
    return df.to_markdown(index=False)


def to_json_records(columns, rows):
    records = [dict(zip(columns, row)) for row in rows]
    return json.dumps(records, ensure_ascii=False, indent=2)


def to_key_value_text(columns, rows):
    lines = []
    for i, row in enumerate(rows, start=1):
        items = [f"{col}={val}" for col, val in zip(columns, row)]
        lines.append(f"row{i}: " + ", ".join(items))
    return "\n".join(lines)


def serialize_table(table, method="markdown"):
    columns = table["columns"]
    rows = table["rows"]

    if method == "markdown":
        return to_markdown_table(columns, rows)
    if method == "json":
        return to_json_records(columns, rows)
    if method == "kv":
        return to_key_value_text(columns, rows)

    raise ValueError(f"Unknown serialization method: {method}")