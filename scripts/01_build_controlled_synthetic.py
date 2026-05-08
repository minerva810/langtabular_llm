import json
import random
from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parents[1]

TABLE_DIR = BASE / "dataset" / "tables" / "controlled_synthetic"
META_PATH = BASE / "dataset" / "metadata" / "controlled_synthetic_metadata.jsonl"
QUESTION_PATH = BASE / "dataset" / "questions" / "controlled_synthetic_questions.jsonl"

RANDOM_SEED = 42
ROWS_PER_TABLE = 35
MIXED_TABLES_PER_SCENARIO = 50
NUMERIC_TABLES = 50
TEXT_TABLES = 50


# Controlled vocabulary used to create paired EN/KO table values.
# Text values are intentionally translated in KO tables to increase row-selection difficulty.
PRODUCTS = [
    ("Laptop", "노트북"),
    ("Tablet", "태블릿"),
    ("Camera", "카메라"),
    ("Monitor", "모니터"),
    ("Keyboard", "키보드"),
    ("Speaker", "스피커"),
    ("Printer", "프린터"),
    ("Router", "공유기"),
    ("Scanner", "스캐너"),
    ("Projector", "프로젝터"),
]

CATEGORIES = [
    ("Consumer", "소비자용"),
    ("Enterprise", "기업용"),
    ("Education", "교육용"),
    ("Medical", "의료용"),
]

REGIONS = [
    ("North", "북부"),
    ("South", "남부"),
    ("East", "동부"),
    ("West", "서부"),
    ("Central", "중부"),
]

STATUS = [
    ("Delayed", "지연"),
    ("Completed", "완료"),
    ("In Progress", "진행중"),
    ("Paused", "중단"),
]

RISK_LEVELS = [
    ("Low", "낮음"),
    ("Medium", "보통"),
    ("High", "높음"),
]


def save_jsonl(items, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def paired_map(pairs):
    # Convert paired vocab like [("Delayed", "지연")] into {"Delayed": "지연"}.
    return {en: ko for en, ko in pairs}


def make_unique_values(pairs, n):
    # Reuse a small vocabulary while adding numeric suffixes so row keys remain unique.
    en_values = []
    ko_values = []
    for idx in range(n):
        en, ko = pairs[idx % len(pairs)]
        suffix = idx + 1
        en_values.append(f"{en} {suffix}")
        ko_values.append(f"{ko} {suffix}")
    return en_values, ko_values


def save_pair(table_id, data_type, scenario, df_en, df_ko):
    # Store by data type and scenario so later analysis can target each condition directly.
    # Example: dataset/tables/controlled_synthetic/text/text_value_linking_filtering/...
    out_dir = TABLE_DIR / data_type / scenario
    out_dir.mkdir(parents=True, exist_ok=True)

    en_path = out_dir / f"{table_id}_en.csv"
    ko_path = out_dir / f"{table_id}_ko.csv"

    df_en.to_csv(en_path, index=False, encoding="utf-8-sig")
    df_ko.to_csv(ko_path, index=False, encoding="utf-8-sig")

    return str(en_path), str(ko_path)


def make_question(
    table_id,
    idx,
    question,
    gold_answer,
    evidence,
    task_type,
    data_type,
    failure_type,
    scenario,
    gold_answer_ko=None,
):
    # Each question carries the intended failure type so automatic labeling can use
    # the controlled design instead of guessing the error source later.
    item = {
        "question_id": f"{table_id}_q{idx}",
        "table_id": table_id,
        "question": question,
        "gold_answer": str(gold_answer),
        "evidence": evidence,
        "task_type": task_type,
        "data_type": data_type,
        "difficulty": "hard",
        "scenario": scenario,
        "failure_type": failure_type,
    }
    if gold_answer_ko is not None:
        item["gold_answer_ko"] = str(gold_answer_ko)
    return item


def add_metadata(metadata, table_id, data_type, scenario, en_path, ko_path, column_mapping):
    # Metadata defines the EN/KO pair and the column translation map used by eval generation.
    metadata.append(
        {
            "table_id": table_id,
            "split_type": "controlled_synthetic",
            "domain": "synthetic_product_operations",
            "scenario": scenario,
            "table_en_path": en_path,
            "table_ko_path": ko_path,
            "num_rows": ROWS_PER_TABLE,
            "num_cols": len(column_mapping),
            "data_type": data_type,
            "difficulty": "hard",
            "column_mapping": column_mapping,
            "paired_guarantee": True,
        }
    )


def build_base_rows(table_idx):
    # Shared mixed-value base table. Scenario builders select and rename columns from this
    # base to isolate one stress factor at a time while keeping row/value generation consistent.
    products_en, products_ko = make_unique_values(PRODUCTS, ROWS_PER_TABLE)
    category_ko = paired_map(CATEGORIES)
    region_ko = paired_map(REGIONS)
    status_ko = paired_map(STATUS)
    risk_ko = paired_map(RISK_LEVELS)

    rows_en = []
    rows_ko = []
    for row_idx in range(ROWS_PER_TABLE):
        category_en, _ = CATEGORIES[(row_idx + table_idx) % len(CATEGORIES)]
        region_en, _ = REGIONS[(row_idx * 2 + table_idx) % len(REGIONS)]
        status_en, _ = STATUS[(row_idx * 3 + table_idx) % len(STATUS)]
        risk_en, _ = RISK_LEVELS[(row_idx + table_idx * 2) % len(RISK_LEVELS)]

        revenue = 800 + random.randint(0, 5200)
        profit = 80 + random.randint(0, 900)
        defect_rate = round(random.uniform(0.4, 7.5), 2)
        return_rate = round(random.uniform(0.2, 8.0), 2)
        rating = round(random.uniform(2.8, 5.0), 1)
        growth = round(random.uniform(-4.0, 19.0), 1)
        delivery_days = 2 + ((row_idx + table_idx) % 12)

        rows_en.append(
            {
                "Product": products_en[row_idx],
                "Category": category_en,
                "Region": region_en,
                "Status": status_en,
                "Risk Level": risk_en,
                "Revenue": revenue,
                "Profit": profit,
                "Defect Rate": defect_rate,
                "Return Rate": return_rate,
                "Customer Rating": rating,
                "Growth Rate": growth,
                "Delivery Days": delivery_days,
            }
        )
        rows_ko.append(
            {
                "Product": products_ko[row_idx],
                "Category": category_ko[category_en],
                "Region": region_ko[region_en],
                "Status": status_ko[status_en],
                "Risk Level": risk_ko[risk_en],
                "Revenue": revenue,
                "Profit": profit,
                "Defect Rate": defect_rate,
                "Return Rate": return_rate,
                "Customer Rating": rating,
                "Growth Rate": growth,
                "Delivery Days": delivery_days,
            }
        )

    return pd.DataFrame(rows_en), pd.DataFrame(rows_ko)


def build_mixed_semantic_header_gap(table_idx):
    # Mixed scenario A: English question keywords do not directly match Korean headers.
    # Example: "Revenue" in the question corresponds to "거래 금액" in the KO table.
    data_type = "mixed"
    scenario = "semantic_header_gap"
    table_id = f"synthetic_mixed_semantic_gap_{table_idx:03d}"
    base_en, base_ko = build_base_rows(table_idx)

    column_mapping = {
        "Product": "품목명",
        "Revenue": "거래 금액",
        "Profit": "순이익",
        "Revenue Growth": "전년 대비 증가율",
        "Return Rate": "반송 비율",
        "Customer Rating": "고객 만족도",
    }
    df_en = pd.DataFrame(
        {
            "Product": base_en["Product"],
            "Revenue": base_en["Revenue"],
            "Profit": base_en["Profit"],
            "Revenue Growth": base_en["Growth Rate"],
            "Return Rate": base_en["Return Rate"],
            "Customer Rating": base_en["Customer Rating"],
        }
    )
    df_ko = pd.DataFrame(
        {
            "품목명": base_ko["Product"],
            "거래 금액": base_ko["Revenue"],
            "순이익": base_ko["Profit"],
            "전년 대비 증가율": base_ko["Growth Rate"],
            "반송 비율": base_ko["Return Rate"],
            "고객 만족도": base_ko["Customer Rating"],
        }
    )

    target_idx = table_idx % ROWS_PER_TABLE
    target = df_en.iloc[target_idx]
    max_idx = df_en["Revenue"].idxmax()

    questions = [
        make_question(table_id, 1, f"What is the revenue of {target['Product']}?", target["Revenue"], {"row": target["Product"], "column": "Revenue"}, "retrieval", data_type, "header_grounding", scenario),
        make_question(table_id, 2, "Which product has the highest revenue?", df_en.loc[max_idx, "Product"], {"column": "Revenue", "operation": "max"}, "comparison", data_type, "header_grounding", scenario, gold_answer_ko=df_ko.loc[max_idx, "품목명"]),
        make_question(table_id, 3, f"What is the customer rating of {target['Product']}?", target["Customer Rating"], {"row": target["Product"], "column": "Customer Rating"}, "retrieval", data_type, "language_specific_linking", scenario),
    ]
    return table_id, data_type, scenario, df_en, df_ko, column_mapping, questions


def build_mixed_translated_value_linking(table_idx):
    # Mixed scenario B: both headers and text cell values are translated.
    # This makes row selection depend on linking English question terms to Korean values.
    data_type = "mixed"
    scenario = "translated_value_linking"
    table_id = f"synthetic_mixed_value_linking_{table_idx:03d}"
    base_en, base_ko = build_base_rows(table_idx)
    teams = [f"Team {chr(65 + (idx % 26))}" for idx in range(ROWS_PER_TABLE)]

    column_mapping = {
        "Product": "제품명",
        "Category": "분류",
        "Region": "권역",
        "Project Status": "진행 상태",
        "Risk Level": "위험 등급",
        "Owner Team": "담당 팀",
    }
    df_en = pd.DataFrame(
        {
            "Product": base_en["Product"],
            "Category": base_en["Category"],
            "Region": base_en["Region"],
            "Project Status": base_en["Status"],
            "Risk Level": base_en["Risk Level"],
            "Owner Team": teams,
        }
    )
    df_ko = pd.DataFrame(
        {
            "제품명": base_ko["Product"],
            "분류": base_ko["Category"],
            "권역": base_ko["Region"],
            "진행 상태": base_ko["Status"],
            "위험 등급": base_ko["Risk Level"],
            "담당 팀": teams,
        }
    )

    target_idx = table_idx % ROWS_PER_TABLE
    target = df_en.iloc[target_idx]
    target_ko = df_ko.iloc[target_idx]
    delayed_idx = df_en[df_en["Project Status"] == "Delayed"].index[0]

    questions = [
        make_question(table_id, 1, f"What is the project status of {target['Product']}?", target["Project Status"], {"row": target["Product"], "column": "Project Status"}, "retrieval", data_type, "language_specific_linking", scenario, gold_answer_ko=target_ko["진행 상태"]),
        make_question(table_id, 2, "Which product has project status delayed?", df_en.loc[delayed_idx, "Product"], {"column": "Project Status", "condition": "Delayed"}, "filtering", data_type, "row_selection", scenario, gold_answer_ko=df_ko.loc[delayed_idx, "제품명"]),
        make_question(table_id, 3, f"What is the risk level of {target['Product']}?", target["Risk Level"], {"row": target["Product"], "column": "Risk Level"}, "retrieval", data_type, "language_specific_linking", scenario, gold_answer_ko=target_ko["위험 등급"]),
    ]
    return table_id, data_type, scenario, df_en, df_ko, column_mapping, questions


def build_mixed_multi_condition_filtering(table_idx):
    # Mixed scenario C: questions require combining multiple filters before comparison/counting.
    # Example: defect rate below 3% AND status completed, then max revenue.
    data_type = "mixed"
    scenario = "multi_condition_filtering"
    table_id = f"synthetic_mixed_multi_filter_{table_idx:03d}"
    base_en, base_ko = build_base_rows(table_idx)

    column_mapping = {
        "Product": "제품명",
        "Category": "분류",
        "Region": "지역",
        "Status": "상태",
        "Revenue": "매출액",
        "Defect Rate": "불량률",
        "Return Rate": "반품률",
    }
    df_en = base_en[list(column_mapping.keys())].copy()
    df_ko = pd.DataFrame(
        {
            "제품명": base_ko["Product"],
            "분류": base_ko["Category"],
            "지역": base_ko["Region"],
            "상태": base_ko["Status"],
            "매출액": base_ko["Revenue"],
            "불량률": base_ko["Defect Rate"],
            "반품률": base_ko["Return Rate"],
        }
    )

    eligible = df_en[(df_en["Defect Rate"] < 3.0) & (df_en["Status"] == "Completed")]
    if eligible.empty:
        best_idx = df_en["Revenue"].idxmax()
        df_en.loc[best_idx, "Defect Rate"] = 2.1
        df_en.loc[best_idx, "Status"] = "Completed"
        df_ko.loc[best_idx, "불량률"] = 2.1
        df_ko.loc[best_idx, "상태"] = "완료"
        eligible = df_en[(df_en["Defect Rate"] < 3.0) & (df_en["Status"] == "Completed")]

    answer_idx = eligible["Revenue"].idxmax()
    target_region = df_en.loc[answer_idx, "Region"]
    eligible_region = df_en[(df_en["Region"] == target_region) & (df_en["Return Rate"] < 4.0) & (df_en["Status"] != "Delayed")]
    if eligible_region.empty:
        df_en.loc[answer_idx, "Return Rate"] = 2.5
        df_ko.loc[answer_idx, "반품률"] = 2.5
        eligible_region = df_en[(df_en["Region"] == target_region) & (df_en["Return Rate"] < 4.0) & (df_en["Status"] != "Delayed")]
    region_answer_idx = eligible_region["Revenue"].idxmax()

    questions = [
        make_question(table_id, 1, "Which product has the highest revenue among products with a defect rate below 3% and status completed?", df_en.loc[answer_idx, "Product"], {"conditions": {"Defect Rate": "< 3", "Status": "Completed"}, "operation": "argmax Revenue"}, "filtering_comparison", data_type, "filtering_comparison", scenario, gold_answer_ko=df_ko.loc[answer_idx, "제품명"]),
        make_question(table_id, 2, f"Among products in the {target_region} region with return rate below 4% and not delayed, which product has the highest revenue?", df_en.loc[region_answer_idx, "Product"], {"conditions": {"Region": target_region, "Return Rate": "< 4", "Status": "not Delayed"}, "operation": "argmax Revenue"}, "filtering_comparison", data_type, "filtering_comparison", scenario, gold_answer_ko=df_ko.loc[region_answer_idx, "제품명"]),
        make_question(table_id, 3, "How many completed products have a defect rate below 3%?", len(eligible), {"conditions": {"Defect Rate": "< 3", "Status": "Completed"}, "operation": "count"}, "filtering", data_type, "filtering_comparison", scenario),
    ]
    return table_id, data_type, scenario, df_en, df_ko, column_mapping, questions


def build_mixed_korean_unit_expression(table_idx):
    # Mixed scenario D: Korean headers include unit expressions such as 백만달러 and %.
    # This targets header grounding plus unit interpretation.
    data_type = "mixed"
    scenario = "korean_unit_expression"
    table_id = f"synthetic_mixed_korean_units_{table_idx:03d}"
    base_en, base_ko = build_base_rows(table_idx)

    column_mapping = {
        "Product": "제품명",
        "Sales (million USD)": "매출액(백만달러)",
        "Operating Profit (million USD)": "영업이익(백만달러)",
        "Defect Rate (%)": "불량률(%)",
        "Delivery Time (days)": "배송기간(일)",
        "Growth Rate (%)": "성장률(%)",
    }
    df_en = pd.DataFrame(
        {
            "Product": base_en["Product"],
            "Sales (million USD)": base_en["Revenue"],
            "Operating Profit (million USD)": base_en["Profit"],
            "Defect Rate (%)": base_en["Defect Rate"],
            "Delivery Time (days)": base_en["Delivery Days"],
            "Growth Rate (%)": base_en["Growth Rate"],
        }
    )
    df_ko = pd.DataFrame(
        {
            "제품명": base_ko["Product"],
            "매출액(백만달러)": base_ko["Revenue"],
            "영업이익(백만달러)": base_ko["Profit"],
            "불량률(%)": base_ko["Defect Rate"],
            "배송기간(일)": base_ko["Delivery Days"],
            "성장률(%)": base_ko["Growth Rate"],
        }
    )

    max_sales_idx = df_en["Sales (million USD)"].idxmax()
    low_defect = df_en[df_en["Defect Rate (%)"] < 3.0]
    if low_defect.empty:
        df_en.loc[max_sales_idx, "Defect Rate (%)"] = 2.2
        df_ko.loc[max_sales_idx, "불량률(%)"] = 2.2
        low_defect = df_en[df_en["Defect Rate (%)"] < 3.0]
    low_defect_max_idx = low_defect["Sales (million USD)"].idxmax()
    target_idx = table_idx % ROWS_PER_TABLE
    target = df_en.iloc[target_idx]

    questions = [
        make_question(table_id, 1, "Which product has the highest sales in million USD?", df_en.loc[max_sales_idx, "Product"], {"column": "Sales (million USD)", "operation": "max"}, "comparison", data_type, "header_grounding", scenario, gold_answer_ko=df_ko.loc[max_sales_idx, "제품명"]),
        make_question(table_id, 2, "Which product has the highest sales among products with a defect rate below 3%?", df_en.loc[low_defect_max_idx, "Product"], {"conditions": {"Defect Rate (%)": "< 3"}, "operation": "argmax Sales (million USD)"}, "filtering_comparison", data_type, "filtering_comparison", scenario, gold_answer_ko=df_ko.loc[low_defect_max_idx, "제품명"]),
        make_question(table_id, 3, f"What is the delivery time in days of {target['Product']}?", target["Delivery Time (days)"], {"row": target["Product"], "column": "Delivery Time (days)"}, "retrieval", data_type, "header_grounding", scenario),
    ]
    return table_id, data_type, scenario, df_en, df_ko, column_mapping, questions


def build_numeric_challenge(table_idx):
    # Numeric-only tables keep text values minimal but still stress unit-bearing headers
    # and numeric filtering/comparison.
    data_type = "numeric"
    scenario = "numeric_header_units_filtering"
    table_id = f"synthetic_numeric_challenge_{table_idx:03d}"
    units = [f"Unit {table_idx}-{idx}" for idx in range(1, ROWS_PER_TABLE + 1)]

    df_en = pd.DataFrame(
        {
            "Unit": units,
            "Sales (million USD)": [900 + random.randint(0, 4500) for _ in units],
            "Operating Cost (million USD)": [300 + random.randint(0, 1800) for _ in units],
            "Defect Rate (%)": [round(random.uniform(0.3, 7.0), 2) for _ in units],
            "Delivery Time (days)": [2 + ((idx + table_idx) % 12) for idx in range(ROWS_PER_TABLE)],
            "Growth Rate (%)": [round(random.uniform(-5.0, 18.0), 1) for _ in units],
        }
    )
    column_mapping = {
        "Unit": "단위",
        "Sales (million USD)": "매출액(백만달러)",
        "Operating Cost (million USD)": "운영비용(백만달러)",
        "Defect Rate (%)": "불량률(%)",
        "Delivery Time (days)": "배송기간(일)",
        "Growth Rate (%)": "성장률(%)",
    }
    df_ko = df_en.rename(columns=column_mapping)

    max_sales_idx = df_en["Sales (million USD)"].idxmax()
    low_defect = df_en[df_en["Defect Rate (%)"] < 3.0]
    if low_defect.empty:
        df_en.loc[max_sales_idx, "Defect Rate (%)"] = 2.0
        df_ko.loc[max_sales_idx, "불량률(%)"] = 2.0
        low_defect = df_en[df_en["Defect Rate (%)"] < 3.0]
    low_defect_idx = low_defect["Sales (million USD)"].idxmax()
    target = df_en.iloc[table_idx % ROWS_PER_TABLE]

    questions = [
        make_question(table_id, 1, "Which unit has the highest sales in million USD?", df_en.loc[max_sales_idx, "Unit"], {"column": "Sales (million USD)", "operation": "max"}, "comparison", data_type, "header_grounding", scenario),
        make_question(table_id, 2, "Which unit has the highest sales among units with a defect rate below 3%?", df_en.loc[low_defect_idx, "Unit"], {"conditions": {"Defect Rate (%)": "< 3"}, "operation": "argmax Sales (million USD)"}, "filtering_comparison", data_type, "filtering_comparison", scenario),
        make_question(table_id, 3, f"What is the delivery time in days of {target['Unit']}?", target["Delivery Time (days)"], {"row": target["Unit"], "column": "Delivery Time (days)"}, "retrieval", data_type, "header_grounding", scenario),
    ]
    return table_id, data_type, scenario, df_en, df_ko, column_mapping, questions


def build_text_challenge(table_idx):
    # Text-only tables translate all categorical values, making language-specific linking
    # and row selection the main source of difficulty.
    data_type = "text"
    scenario = "text_value_linking_filtering"
    table_id = f"synthetic_text_challenge_{table_idx:03d}"
    products_en, products_ko = make_unique_values(PRODUCTS, ROWS_PER_TABLE)
    category_ko = paired_map(CATEGORIES)
    region_ko = paired_map(REGIONS)
    status_ko = paired_map(STATUS)
    risk_ko = paired_map(RISK_LEVELS)

    rows_en = []
    rows_ko = []
    for row_idx in range(ROWS_PER_TABLE):
        category_en, _ = CATEGORIES[(row_idx + table_idx) % len(CATEGORIES)]
        region_en, _ = REGIONS[(row_idx * 2 + table_idx) % len(REGIONS)]
        status_en, _ = STATUS[(row_idx * 3 + table_idx) % len(STATUS)]
        risk_en, _ = RISK_LEVELS[(row_idx + table_idx) % len(RISK_LEVELS)]
        owner = f"Team {chr(65 + (row_idx % 26))}"

        rows_en.append(
            {
                "Product": products_en[row_idx],
                "Category": category_en,
                "Region": region_en,
                "Project Status": status_en,
                "Risk Level": risk_en,
                "Owner Team": owner,
            }
        )
        rows_ko.append(
            {
                "제품명": products_ko[row_idx],
                "분류": category_ko[category_en],
                "권역": region_ko[region_en],
                "진행 상태": status_ko[status_en],
                "위험 등급": risk_ko[risk_en],
                "담당 팀": owner,
            }
        )

    df_en = pd.DataFrame(rows_en)
    df_ko = pd.DataFrame(rows_ko)
    column_mapping = {
        "Product": "제품명",
        "Category": "분류",
        "Region": "권역",
        "Project Status": "진행 상태",
        "Risk Level": "위험 등급",
        "Owner Team": "담당 팀",
    }

    target = df_en.iloc[table_idx % ROWS_PER_TABLE]
    target_ko = df_ko.iloc[table_idx % ROWS_PER_TABLE]
    delayed_idx = df_en[df_en["Project Status"] == "Delayed"].index[0]
    multi = df_en[
        (df_en["Project Status"] == "Completed")
        & (df_en["Risk Level"] == "Low")
        & (df_en["Region"] == df_en.loc[delayed_idx, "Region"])
    ]
    if multi.empty:
        multi_idx = delayed_idx
        df_en.loc[multi_idx, "Project Status"] = "Completed"
        df_en.loc[multi_idx, "Risk Level"] = "Low"
        df_ko.loc[multi_idx, "진행 상태"] = "완료"
        df_ko.loc[multi_idx, "위험 등급"] = "낮음"
    else:
        multi_idx = multi.index[0]

    questions = [
        make_question(table_id, 1, f"What is the project status of {target['Product']}?", target["Project Status"], {"row": target["Product"], "column": "Project Status"}, "retrieval", data_type, "language_specific_linking", scenario, gold_answer_ko=target_ko["진행 상태"]),
        make_question(table_id, 2, "Which product has project status delayed?", df_en.loc[delayed_idx, "Product"], {"column": "Project Status", "condition": "Delayed"}, "filtering", data_type, "row_selection", scenario, gold_answer_ko=df_ko.loc[delayed_idx, "제품명"]),
        make_question(table_id, 3, f"Which product is completed, low risk, and in the {df_en.loc[multi_idx, 'Region']} region?", df_en.loc[multi_idx, "Product"], {"conditions": {"Project Status": "Completed", "Risk Level": "Low", "Region": df_en.loc[multi_idx, "Region"]}}, "filtering", data_type, "filtering_comparison", scenario, gold_answer_ko=df_ko.loc[multi_idx, "제품명"]),
    ]
    return table_id, data_type, scenario, df_en, df_ko, column_mapping, questions


def main():
    random.seed(RANDOM_SEED)

    # Each builder returns one EN/KO paired table plus three questions.
    # Mixed has four scenario families; numeric/text add dedicated non-mixed controls.
    builders = [
        (MIXED_TABLES_PER_SCENARIO, build_mixed_semantic_header_gap),
        (MIXED_TABLES_PER_SCENARIO, build_mixed_translated_value_linking),
        (MIXED_TABLES_PER_SCENARIO, build_mixed_multi_condition_filtering),
        (MIXED_TABLES_PER_SCENARIO, build_mixed_korean_unit_expression),
        (NUMERIC_TABLES, build_numeric_challenge),
        (TEXT_TABLES, build_text_challenge),
    ]

    metadata = []
    questions = []

    for count, builder in builders:
        for table_idx in range(1, count + 1):
            table_id, data_type, scenario, df_en, df_ko, column_mapping, table_questions = builder(table_idx)
            en_path, ko_path = save_pair(table_id, data_type, scenario, df_en, df_ko)
            add_metadata(metadata, table_id, data_type, scenario, en_path, ko_path, column_mapping)
            questions.extend(table_questions)

    save_jsonl(metadata, META_PATH)
    save_jsonl(questions, QUESTION_PATH)

    print("[controlled_synthetic] built")
    print(f"Tables: {len(metadata)}")
    print(f"Questions: {len(questions)}")
    print("[data type distribution]")
    for data_type in ["numeric", "text", "mixed"]:
        count = sum(item["data_type"] == data_type for item in metadata)
        print(f"- {data_type}: {count}")
    print("[storage]")
    print(TABLE_DIR)


if __name__ == "__main__":
    main()
