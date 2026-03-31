# build_kor_tabular_benchmark.py
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any

import pandas as pd

random.seed(42)

# =========================================================
# 0. Paths
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

DATA_FILES = {
    "child_protection": DATA_DIR / "인천광역시_보호대상아동 발생 및 조치내용_20251231.csv",
    "child_welfare_facility": DATA_DIR / "인천광역시_아동복지시설(통계)현황_20260228.csv",
    "income_percentile": DATA_DIR / "국세청_근로소득 백분위(천분위) 자료_20251231.csv",
    "influenza_detection": DATA_DIR / "질병관리청_인플루엔자 주별 연령별 검출률_20251228.csv",
}

OUT_DIR = Path("./benchmark_outputs")
OUT_DIR.mkdir(exist_ok=True, parents=True)


# =========================================================
# 1. Utility
# =========================================================
def read_csv_auto(path: Path) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "cp949", "euc-kr"]
    last_error = None

    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_error = e

    raise RuntimeError(f"Failed to read CSV: {path}\nLast error: {last_error}")

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def safe_to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False), errors="ignore")


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_columns(df)
    for col in df.columns:
        df[col] = safe_to_numeric(df[col])
    return df


def df_to_table_json(
    df: pd.DataFrame,
    table_id: str,
    language: str,
    dataset_name: str,
    column_map: Dict[str, str] = None
) -> Dict[str, Any]:
    """
    DataFrame -> 실험용 table JSON
    language == "ko": 원본 컬럼명
    language == "en": column_map 적용
    """
    if language == "ko":
        columns = list(df.columns)
    else:
        if column_map is None:
            raise ValueError("column_map is required for English table generation")
        columns = [column_map.get(c, c) for c in df.columns]

    rows = df.values.tolist()

    return {
        "table_id": table_id,
        "dataset_name": dataset_name,
        "language": language,
        "columns": columns,
        "rows": rows
    }


def save_jsonl(path: Path, items: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def is_number(x) -> bool:
    return isinstance(x, (int, float)) and not pd.isna(x)


# =========================================================
# 2. Column mappings
# =========================================================

COLUMN_MAPS = {
    "influenza_detection": {
        "년도": "year",
        "주": "week",
        "연령": "age_group",
        "검출률": "detection_rate",
    },
    "income_percentile": {
        "구분": "percentile_group",
        "인원": "num_people",
        "총급여": "total_salary",
        "근로소득금액": "earned_income",
        "소득공제금액": "income_deduction",
        "과세표준": "tax_base",
        "결정세액": "determined_tax",
    },
    "child_welfare_facility": {
        "군구별": "district",
        "양육시설_시설수": "care_facility_count",
        "양육시설_입소자": "care_admissions",
        "양육시설_퇴소자": "care_discharges",
        "양육시설_현재 생활인원": "care_current_residents",
        "기타_시설수": "other_facility_count",
        "기타_입소자": "other_admissions",
        "기타_퇴소자": "other_discharges",
        "기타_현재 생활인원": "other_current_residents",
    },
    "child_protection": {
        "구분": "district",
        "총아동발생수": "total_child_cases",
        "귀가및연고자인도": "returned_to_guardian",
        "보호대상아동의 발생원인": "protected_child_case_count",
        "유기": "abandonment",
        "미혼부모_혼외자": "unmarried_parent_nonmarital_child",
        "미아": "missing_child",
        "비행_가출_부랑": "delinquency_runaway_vagrancy",
        "학대": "abuse",
        "부모빈곤_실직": "parent_poverty_unemployment",
        "부모사망": "parent_death",
        "부모질병": "parent_illness",
        "부모이혼등": "parent_divorce_etc",
        "성별": "gender_total",
        "남": "male",
        "여": "female",
        "장애여부": "disability_total",
        "비장애": "non_disabled",
        "장애": "disabled",
        "양육시설": "care_facility",
        "일시보호시설": "temporary_protection_facility",
        "장애아동시설": "facility_for_children_with_disabilities",
        "공동생활가정": "group_home",
        "기타시설": "other_facility",
        "소년소녀가정": "child_headed_household",
        "입양": "adoption",
        "보호출산": "protected_birth",
        "가정위탁": "foster_care",
        "입양전위탁": "pre_adoption_foster_care",
    }
}


# =========================================================
# 3. Dataset-specific preprocessing
# =========================================================

def preprocess_influenza(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_dataframe(df)
    df = df.dropna(subset=["년도", "주", "연령", "검출률"]).copy()
    df["년도"] = df["년도"].astype(int)
    df["주"] = df["주"].astype(int)
    df["검출률"] = pd.to_numeric(df["검출률"], errors="coerce")
    df = df.sort_values(["년도", "주", "연령"]).reset_index(drop=True)
    return df


def preprocess_income(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_dataframe(df)
    if " 구분 " in df.columns:
        df = df.rename(columns={" 구분 ": "구분"})
    numeric_cols = ["인원", "총급여", "근로소득금액", "소득공제금액", "과세표준", "결정세액"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["구분"]).reset_index(drop=True)
    return df


def preprocess_child_welfare(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_dataframe(df)
    numeric_cols = [c for c in df.columns if c != "군구별"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["군구별"]).reset_index(drop=True)
    return df


def preprocess_child_protection(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_dataframe(df)
    numeric_cols = [c for c in df.columns if c != "구분"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["구분"]).reset_index(drop=True)
    return df


PREPROCESSORS = {
    "influenza_detection": preprocess_influenza,
    "income_percentile": preprocess_income,
    "child_welfare_facility": preprocess_child_welfare,
    "child_protection": preprocess_child_protection,
}


# =========================================================
# 4. Value translation helpers
# =========================================================

AGE_GROUP_MAP = {
    "0-6세": "0-6 years",
    "7-12세": "7-12 years",
    "13-18세": "13-18 years",
    "19-49세": "19-49 years",
    "50-64세": "50-64 years",
    "65세 이상": "65 years and older",
}

DISTRICT_MAP = {
    "인천시": "Incheon",
    "중구": "Junggu",
    "동구": "Donggu",
    "미추홀구": "Michuhol-gu",
    "연수구": "Yeonsu-gu",
    "남동구": "Namdong-gu",
    "부평구": "Bupyeong-gu",
    "계양구": "Gyeyang-gu",
    "서구": "Seo-gu",
    "강화군": "Ganghwa-gun",
    "옹진군": "Ongjin-gun",
}

PERCENTILE_GROUP_MAP = {
    # 원본 문자열을 그대로 쓰되 영어 질문에서는 label 그대로 재사용해도 됨
    # 필요하면 여기서 더 자연스럽게 번역 가능
}


def translate_value(dataset_name: str, col_ko: str, value: Any) -> Any:
    if dataset_name == "influenza_detection" and col_ko == "연령":
        return AGE_GROUP_MAP.get(value, value)

    if dataset_name in ["child_welfare_facility", "child_protection"]:
        key_col = "군구별" if dataset_name == "child_welfare_facility" else "구분"
        if col_ko == key_col:
            return DISTRICT_MAP.get(value, value)

    return value


def make_english_rows(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    df_en = df.copy()
    for col in df.columns:
        df_en[col] = df_en[col].apply(lambda x: translate_value(dataset_name, col, x))
    return df_en


# =========================================================
# 5. QA generation
# =========================================================

def generate_qa_influenza(df_ko: pd.DataFrame, df_en: pd.DataFrame, table_id: str):
    qas = []

    # lookup
    sample = df_ko.sample(1, random_state=42).iloc[0]
    ko_q = f"{int(sample['년도'])}년 {int(sample['주'])}주 {sample['연령']}의 검출률은?"
    en_age = translate_value("influenza_detection", "연령", sample["연령"])
    en_q = f"What is the detection rate for {en_age} in year {int(sample['년도'])}, week {int(sample['주'])}?"
    ans = float(sample["검출률"])

    qas.append(("lookup", ko_q, en_q, ans, ans))

    # argmax within year-week
    sample2 = df_ko.sample(1, random_state=7).iloc[0]
    year_, week_ = int(sample2["년도"]), int(sample2["주"])
    sub_ko = df_ko[(df_ko["년도"] == year_) & (df_ko["주"] == week_)]
    sub_en = df_en[(df_en["년도"] == year_) & (df_en["주"] == week_)]
    best_ko = sub_ko.loc[sub_ko["검출률"].idxmax()]
    best_en = sub_en.loc[sub_en["검출률"].idxmax()]

    ko_q = f"{year_}년 {week_}주차에 검출률이 가장 높은 연령대는?"
    en_q = f"Which age group has the highest detection rate in year {year_}, week {week_}?"
    qas.append(("argmax", ko_q, en_q, best_ko["연령"], best_en["연령"]))

    # aggregation
    year_choice = int(df_ko["년도"].sample(1, random_state=11).iloc[0])
    age_choice = df_ko["연령"].sample(1, random_state=13).iloc[0]
    sub_ko = df_ko[(df_ko["년도"] == year_choice) & (df_ko["연령"] == age_choice)]
    mean_val = round(float(sub_ko["검출률"].mean()), 4)

    ko_q = f"{year_choice}년 {age_choice}의 평균 검출률은?"
    en_q = f"What is the average detection rate for {translate_value('influenza_detection', '연령', age_choice)} in year {year_choice}?"
    qas.append(("aggregation", ko_q, en_q, mean_val, mean_val))

    # comparison
    years = sorted(df_ko["년도"].unique())
    if len(years) >= 2:
        y1, y2 = years[0], years[-1]
        age_choice = df_ko["연령"].sample(1, random_state=21).iloc[0]
        m1 = df_ko[(df_ko["년도"] == y1) & (df_ko["연령"] == age_choice)]["검출률"].mean()
        m2 = df_ko[(df_ko["년도"] == y2) & (df_ko["연령"] == age_choice)]["검출률"].mean()

        ko_q = f"{age_choice}의 평균 검출률은 {y1}년보다 {y2}년에 더 높은가? yes 또는 no로 답하라."
        en_q = f"Is the average detection rate for {translate_value('influenza_detection', '연령', age_choice)} higher in {y2} than in {y1}? Answer yes or no."
        answer = "yes" if m2 > m1 else "no"
        qas.append(("comparison", ko_q, en_q, answer, answer))

    return build_qa_items(table_id, "influenza_detection", qas)


def generate_qa_income(df_ko: pd.DataFrame, df_en: pd.DataFrame, table_id: str):
    qas = []

    sample = df_ko.sample(1, random_state=42).iloc[0]
    group = sample["구분"]

    ko_q = f"{group}의 결정세액은?"
    en_q = f"What is the determined tax for {group}?"
    ans = int(sample["결정세액"])
    qas.append(("lookup", ko_q, en_q, ans, ans))

    best_ko = df_ko.loc[df_ko["총급여"].idxmax()]
    best_en = df_en.loc[df_en["총급여"].idxmax()]
    ko_q = "총급여가 가장 큰 구간은?"
    en_q = "Which percentile group has the highest total salary?"
    qas.append(("argmax", ko_q, en_q, best_ko["구분"], best_en["구분"]))

    mean_tax = round(float(df_ko["결정세액"].mean()), 4)
    ko_q = "전체 구간의 평균 결정세액은?"
    en_q = "What is the average determined tax across all groups?"
    qas.append(("aggregation", ko_q, en_q, mean_tax, mean_tax))

    sample2 = df_ko.sample(2, random_state=100)
    g1, g2 = sample2.iloc[0]["구분"], sample2.iloc[1]["구분"]
    t1 = sample2.iloc[0]["결정세액"]
    t2 = sample2.iloc[1]["결정세액"]

    ko_q = f"{g1}와 {g2} 중 결정세액이 더 큰 구간은?"
    en_q = f"Between {g1} and {g2}, which group has the higher determined tax?"
    answer = g1 if t1 > t2 else g2
    qas.append(("comparison", ko_q, en_q, answer, answer))

    return build_qa_items(table_id, "income_percentile", qas)


def generate_qa_child_welfare(df_ko: pd.DataFrame, df_en: pd.DataFrame, table_id: str):
    qas = []

    sample = df_ko.sample(1, random_state=42).iloc[0]
    district_ko = sample["군구별"]
    district_en = translate_value("child_welfare_facility", "군구별", district_ko)

    ko_q = f"{district_ko}의 양육시설 수는?"
    en_q = f"What is the number of care facilities in {district_en}?"
    ans = int(sample["양육시설_시설수"])
    qas.append(("lookup", ko_q, en_q, ans, ans))

    best_ko = df_ko.loc[df_ko["양육시설_현재 생활인원"].idxmax()]
    best_en = df_en.loc[df_en["양육시설_현재 생활인원"].idxmax()]
    ko_q = "양육시설 현재 생활인원이 가장 많은 군구는?"
    en_q = "Which district has the highest number of current residents in care facilities?"
    qas.append(("argmax", ko_q, en_q, best_ko["군구별"], best_en["군구별"]))

    mean_val = round(float(df_ko["기타_시설수"].mean()), 4)
    ko_q = "기타 시설수의 평균은?"
    en_q = "What is the average number of other facilities?"
    qas.append(("aggregation", ko_q, en_q, mean_val, mean_val))

    sample2 = df_ko.sample(2, random_state=77)
    d1, d2 = sample2.iloc[0]["군구별"], sample2.iloc[1]["군구별"]
    v1, v2 = sample2.iloc[0]["양육시설_현재 생활인원"], sample2.iloc[1]["양육시설_현재 생활인원"]

    ko_q = f"{d1}와 {d2} 중 양육시설 현재 생활인원이 더 많은 군구는?"
    en_q = (
        f"Between {translate_value('child_welfare_facility', '군구별', d1)} "
        f"and {translate_value('child_welfare_facility', '군구별', d2)}, "
        f"which district has more current residents in care facilities?"
    )
    answer_ko = d1 if v1 > v2 else d2
    answer_en = translate_value("child_welfare_facility", "군구별", answer_ko)
    qas.append(("comparison", ko_q, en_q, answer_ko, answer_en))

    return build_qa_items(table_id, "child_welfare_facility", qas)


def generate_qa_child_protection(df_ko: pd.DataFrame, df_en: pd.DataFrame, table_id: str):
    qas = []

    sample = df_ko.sample(1, random_state=42).iloc[0]
    district_ko = sample["구분"]
    district_en = translate_value("child_protection", "구분", district_ko)

    ko_q = f"{district_ko}의 학대 건수는?"
    en_q = f"What is the number of abuse cases in {district_en}?"
    ans = int(sample["학대"])
    qas.append(("lookup", ko_q, en_q, ans, ans))

    best_ko = df_ko.loc[df_ko["가정위탁"].idxmax()]
    best_en = df_en.loc[df_en["가정위탁"].idxmax()]
    ko_q = "가정위탁 수가 가장 많은 지역은?"
    en_q = "Which district has the highest foster care count?"
    qas.append(("argmax", ko_q, en_q, best_ko["구분"], best_en["구분"]))

    mean_val = round(float(df_ko["총아동발생수"].mean()), 4)
    ko_q = "전체 지역의 평균 총아동발생수는?"
    en_q = "What is the average total child case count across all districts?"
    qas.append(("aggregation", ko_q, en_q, mean_val, mean_val))

    sample2 = df_ko.sample(2, random_state=88)
    d1, d2 = sample2.iloc[0]["구분"], sample2.iloc[1]["구분"]
    v1, v2 = sample2.iloc[0]["학대"], sample2.iloc[1]["학대"]

    ko_q = f"{d1}와 {d2} 중 학대 건수가 더 많은 지역은?"
    en_q = (
        f"Between {translate_value('child_protection', '구분', d1)} "
        f"and {translate_value('child_protection', '구분', d2)}, "
        f"which district has more abuse cases?"
    )
    answer_ko = d1 if v1 > v2 else d2
    answer_en = translate_value("child_protection", "구분", answer_ko)
    qas.append(("comparison", ko_q, en_q, answer_ko, answer_en))

    return build_qa_items(table_id, "child_protection", qas)


def build_qa_items(table_id: str, dataset_name: str, qas: List[Tuple[str, str, str, Any, Any]]):
    """
    qas = [(task_type, ko_q, en_q, ko_answer, en_answer), ...]
    """
    qa_items = []
    for idx, (task_type, ko_q, en_q, ko_a, en_a) in enumerate(qas):
        qa_items.append({
            "qid_ko": f"{table_id}_ko_{idx}",
            "qid_en": f"{table_id}_en_{idx}",
            "table_id": table_id,
            "dataset_name": dataset_name,
            "task_type": task_type,
            "question_ko": ko_q,
            "question_en": en_q,
            "answer_ko": ko_a,
            "answer_en": en_a,
        })
    return qa_items


QA_GENERATORS = {
    "influenza_detection": generate_qa_influenza,
    "income_percentile": generate_qa_income,
    "child_welfare_facility": generate_qa_child_welfare,
    "child_protection": generate_qa_child_protection,
}


# =========================================================
# 6. Main
# =========================================================

def main():
    all_tables_ko = []
    all_tables_en = []
    all_qas = []

    for dataset_name, path in DATA_FILES.items():
        print(f"\n[Processing] {dataset_name}")

        df_raw = read_csv_auto(path)
        df = PREPROCESSORS[dataset_name](df_raw)
        df_en_values = make_english_rows(df, dataset_name)

        table_id = dataset_name

        table_ko = df_to_table_json(
            df=df,
            table_id=table_id,
            language="ko",
            dataset_name=dataset_name
        )

        table_en = df_to_table_json(
            df=df_en_values,
            table_id=table_id,
            language="en",
            dataset_name=dataset_name,
            column_map=COLUMN_MAPS[dataset_name]
        )

        qas = QA_GENERATORS[dataset_name](df, df_en_values, table_id)

        all_tables_ko.append(table_ko)
        all_tables_en.append(table_en)
        all_qas.extend(qas)

        print(f"  rows={len(df)}, cols={len(df.columns)}, qas={len(qas)}")

    save_jsonl(OUT_DIR / "tables_ko.jsonl", all_tables_ko)
    save_jsonl(OUT_DIR / "tables_en.jsonl", all_tables_en)
    save_jsonl(OUT_DIR / "qas_paired.jsonl", all_qas)

    with open(OUT_DIR / "column_maps.json", "w", encoding="utf-8") as f:
        json.dump(COLUMN_MAPS, f, ensure_ascii=False, indent=2)

    print("\nSaved:")
    print(f"- {OUT_DIR / 'tables_ko.jsonl'}")
    print(f"- {OUT_DIR / 'tables_en.jsonl'}")
    print(f"- {OUT_DIR / 'qas_paired.jsonl'}")
    print(f"- {OUT_DIR / 'column_maps.json'}")


if __name__ == "__main__":
    main()