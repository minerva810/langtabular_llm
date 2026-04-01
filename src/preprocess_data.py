import json
import re
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
METADATA_DIR = DATA_DIR / "metadata"

for sub in ["numeric", "text", "mixed"]:
    (PROCESSED_DIR / sub).mkdir(parents=True, exist_ok=True)

(METADATA_DIR / "dataset_info").mkdir(parents=True, exist_ok=True)
(METADATA_DIR / "column_maps").mkdir(parents=True, exist_ok=True)
(METADATA_DIR / "value_maps").mkdir(parents=True, exist_ok=True)


# ---------------------------
# 1. Utility
# ---------------------------

def read_csv_auto(path: Path) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "cp949", "euc-kr"]
    last_error = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_error = e
    raise RuntimeError(f"Failed to read {path}\nLast error: {last_error}")


def clean_column_name(col: str) -> str:
    col = str(col).strip()
    col = re.sub(r"\s+", " ", col)
    return col


def normalize_text_value(x):
    if pd.isna(x):
        return x
    x = str(x).strip()
    x = re.sub(r"\s+", " ", x)
    return x


def try_convert_numeric(series: pd.Series) -> pd.Series:
    """
    쉼표, 공백 제거 후 숫자형 변환 시도.
    숫자로 변환 안 되는 값은 원래 값 유지.
    """
    raw = series.astype(str).str.replace(",", "", regex=False).str.strip()
    converted = pd.to_numeric(raw, errors="coerce")
    return converted.where(~converted.isna(), series)


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [clean_column_name(c) for c in df.columns]

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(normalize_text_value)

    for col in df.columns:
        df[col] = try_convert_numeric(df[col])

    return df


def infer_column_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    text_cols, numeric_cols = [], []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            text_cols.append(col)
    return text_cols, numeric_cols


def drop_empty_rows_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    return df


def remove_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates().reset_index(drop=True)


# ---------------------------
# 2. Dataset-specific maps
# ---------------------------
COLUMN_MAPS = {
    "국세청_근로소득 백분위(천분위) 자료_20251231": {
        "구분": "percentile_group",
        "인원": "num_people",
        "총급여": "total_salary",
        "근로소득금액": "earned_income",
        "소득공제금액": "income_deduction",
        "과세표준": "tax_base",
        "결정세액": "determined_tax",
    },

    "질병관리청_인플루엔자 주별 연령별 검출률_20251228": {
        "년도": "year",
        "주": "week",
        "연령": "age_group",
        "검출률": "detection_rate",
    },

    "한국도로교통공단_도로종류별 기상상태별 교통사고 통계_20241231": {
        "도로종류": "road_type",
        "기상상태": "weather_condition",
        "사고건수": "num_accidents",
        "사망자수": "num_deaths",
        "중상자수": "serious_injuries",
        "경상자수": "minor_injuries",
        "부상신고자수": "reported_injuries",
    },

    "한국도로교통공단_사고유형별 교통사고 통계_20241231": {
        "사고유형대분류": "accident_type_major",
        "사고유형중분류": "accident_type_mid",
        "사고유형": "accident_type",
        "사고건수": "num_accidents",
        "사망자수": "num_deaths",
        "중상자수": "serious_injuries",
        "경상자수": "minor_injuries",
        "부상신고자수": "reported_injuries",
    },

    "StudentsPerformance": {
        "gender": "gender",
        "race/ethnicity": "race_ethnicity",
        "parental level of education": "parental_education",
        "lunch": "lunch",
        "test preparation course": "test_preparation_course",
        "math score": "math_score",
        "reading score": "reading_score",
        "writing score": "writing_score",
    },

    "heart": {
        "age": "age",
        "sex": "sex",
        "cp": "chest_pain_type",
        "trestbps": "resting_blood_pressure",
        "chol": "cholesterol",
        "fbs": "fasting_blood_sugar",
        "restecg": "resting_ecg",
        "thalach": "max_heart_rate",
        "exang": "exercise_induced_angina",
        "oldpeak": "oldpeak",
        "slope": "st_slope",
        "ca": "num_major_vessels",
        "thal": "thalassemia",
        "target": "heart_disease_score",
    },

    "cars": {
        "buying": "buying_price",
        "maint": "maintenance_cost",
        "doors": "num_doors",
        "persons": "capacity",
        "lug_boot": "luggage_boot_size",
        "safety": "safety_level",
        "class": "car_acceptability",
    },

    "laptop_price - dataset": {
        "Company": "company",
        "Product": "product",
        "TypeName": "type_name",
        "Inches": "screen_size_inches",
        "ScreenResolution": "screen_resolution",
        "CPU_Company": "cpu_company",
        "CPU_Type": "cpu_type",
        "CPU_Frequency (GHz)": "cpu_frequency_ghz",
        "RAM (GB)": "ram_gb",
        "Memory": "memory",
        "GPU_Company": "gpu_company",
        "GPU_Type": "gpu_type",
        "OpSys": "operating_system",
        "Weight (kg)": "weight_kg",
        "Price (Euro)": "price_euro",
    },

    "Thyroid_Diff": {
        "Age": "age",
        "Gender": "gender",
        "Smoking": "smoking",
        "Hx Smoking": "history_of_smoking",
        "Hx Radiothreapy": "history_of_radiotherapy",
        "Thyroid Function": "thyroid_function",
        "Physical Examination": "physical_examination",
        "Adenopathy": "adenopathy",
        "Pathology": "pathology",
        "Focality": "focality",
        "Risk": "risk",
        "T": "t_stage",
        "N": "n_stage",
        "M": "m_stage",
        "Stage": "stage",
        "Response": "response",
        "Recurred": "recurred",
    },
}

VALUE_MAPS = {
    "질병관리청_인플루엔자 주별 연령별 검출률_20251228": {
        "연령": {
            "0-6": "0-6 years",
            "7-12": "7-12 years",
            "13-18": "13-18 years",
            "19-49": "19-49 years",
            "50-64": "50-64 years",
            "65+": "65 years and older",
            "65이상": "65 years and older",
        }
    },

    "한국도로교통공단_도로종류별 기상상태별 교통사고 통계_20241231": {
        "도로종류": {
            "고속도로": "expressway",
            "일반국도": "national highway",
            "특별광역시도": "metropolitan city road",
            "지방도": "provincial road",
            "시도": "city road",
            "군도": "county road",
            "기타": "other",
        },
        "기상상태": {
            "맑음": "clear",
            "흐림": "cloudy",
            "비": "rain",
            "안개": "fog",
            "눈": "snow",
            "기타/불명": "other_or_unknown",
        }
    },

    "한국도로교통공단_사고유형별 교통사고 통계_20241231": {
        "사고유형대분류": {
            "차대사람": "vehicle_to_pedestrian",
            "차대차": "vehicle_to_vehicle",
            "차량단독": "single_vehicle",
        },
        "사고유형중분류": {
            "횡단중": "crossing",
            "차도통행중": "walking_on_roadway",
            "길가장자리구역통행중": "walking_on_roadside_zone",
            "보도통행중": "walking_on_sidewalk",
            "기타": "other",
            "정면충돌": "head_on_collision",
            "추돌": "rear_end_collision",
            "측면충돌": "side_collision",
            "기타": "other",
            "공작물충돌": "collision_with_structure",
            "도로이탈": "road_departure",
            "전도전복": "rollover",
            "주/정차차량 충돌": "collision_with_parked_vehicle",
            "기타": "other",
        },
        "사고유형": {
            "횡단중": "crossing",
            "차도통행중": "walking_on_roadway",
            "길가장자리구역통행중": "walking_on_roadside_zone",
            "보도통행중": "walking_on_sidewalk",
            "기타": "other",
            "정면충돌": "head_on_collision",
            "추돌": "rear_end_collision",
            "측면충돌": "side_collision",
            "공작물충돌": "collision_with_structure",
            "도로이탈": "road_departure",
            "전도전복": "rollover",
            "주/정차차량 충돌": "collision_with_parked_vehicle",
        }
    },

    "StudentsPerformance": {
        "gender": {
            "female": "여성",
            "male": "남성",
        },
        "race/ethnicity": {
            "group A": "A 그룹",
            "group B": "B 그룹",
            "group C": "C 그룹",
            "group D": "D 그룹",
            "group E": "E 그룹",
        },
        "parental level of education": {
            "some high school": "고등학교 일부 이수",
            "high school": "고등학교 졸업",
            "some college": "대학 일부 이수",
            "associate's degree": "전문학사",
            "bachelor's degree": "학사",
            "master's degree": "석사",
        },
        "lunch": {
            "standard": "일반식",
            "free/reduced": "무상/감면 급식",
        },
        "test preparation course": {
            "none": "이수 안 함",
            "completed": "이수함",
        }
    },

    "heart": {
        "sex": {
            0: "female",
            1: "male",
        },
        "cp": {
            0: "typical_angina_type_0",
            1: "atypical_angina_type_1",
            2: "non_anginal_pain_type_2",
            3: "asymptomatic_type_3",
        },
        "fbs": {
            0: "false",
            1: "true",
        },
        "restecg": {
            0: "normal",
            1: "st_t_wave_abnormality",
            2: "left_ventricular_hypertrophy",
        },
        "exang": {
            0: "no",
            1: "yes",
        },
        "slope": {
            0: "downsloping",
            1: "flat",
            2: "upsloping",
        },
        "thal": {
            0: "unknown",
            1: "normal",
            2: "fixed_defect",
            3: "reversible_defect",
        }
    },

    "cars": {
        "buying": {
            "vhigh": "very_high",
            "high": "high",
            "med": "medium",
            "low": "low",
        },
        "maint": {
            "vhigh": "very_high",
            "high": "high",
            "med": "medium",
            "low": "low",
        },
        "doors": {
            "2": "2",
            "3": "3",
            "4": "4",
            "5more": "5_or_more",
        },
        "persons": {
            "2": "2",
            "4": "4",
            "more": "more_than_4",
        },
        "lug_boot": {
            "small": "small",
            "med": "medium",
            "big": "big",
        },
        "safety": {
            "low": "low",
            "med": "medium",
            "high": "high",
        },
        "class": {
            "unacc": "unacceptable",
            "acc": "acceptable",
            "good": "good",
            "vgood": "very_good",
        }
    },

    "laptop_price - dataset": {
        "TypeName": {
            "Ultrabook": "ultrabook",
            "Notebook": "notebook",
            "Gaming": "gaming_laptop",
            "2 in 1 Convertible": "2_in_1_convertible",
            "Workstation": "workstation",
            "Netbook": "netbook",
        },
        "CPU_Company": {
            "Intel": "intel",
            "AMD": "amd",
            "Samsung": "samsung",
        },
        "GPU_Company": {
            "Intel": "intel",
            "AMD": "amd",
            "Nvidia": "nvidia",
            "ARM": "arm",
        },
        "OpSys": {
            "Windows 10": "windows_10",
            "Windows 10 S": "windows_10_s",
            "No OS": "no_os",
            "Linux": "linux",
            "macOS": "macos",
            "Mac OS X": "mac_os_x",
            "Chrome OS": "chrome_os",
            "Android": "android",
        }
    },

    "Thyroid_Diff": {
        "Gender": {
            "F": "female",
            "M": "male",
        },
        "Smoking": {
            "Yes": "yes",
            "No": "no",
        },
        "Hx Smoking": {
            "Yes": "yes",
            "No": "no",
        },
        "Hx Radiothreapy": {
            "Yes": "yes",
            "No": "no",
        },
        "Thyroid Function": {
            "Euthyroid": "euthyroid",
            "Clinical Hyperthyroidism": "clinical_hyperthyroidism",
            "Clinical Hypothyroidism": "clinical_hypothyroidism",
            "Subclinical Hyperthyroidism": "subclinical_hyperthyroidism",
            "Subclinical Hypothyroidism": "subclinical_hypothyroidism",
        },
        "Physical Examination": {
            "Single nodular goiter-left": "single_nodular_goiter_left",
            "Single nodular goiter-right": "single_nodular_goiter_right",
            "Multinodular goiter": "multinodular_goiter",
            "Diffuse goiter": "diffuse_goiter",
            "Normal": "normal",
        },
        "Adenopathy": {
            "No": "no",
            "Right": "right",
            "Left": "left",
            "Bilateral": "bilateral",
            "Posterior": "posterior",
            "Extensive": "extensive",
        },
        "Pathology": {
            "Micropapillary": "micropapillary",
            "Papillary": "papillary",
            "Follicular": "follicular",
            "Hurthel cell": "hurthle_cell",
        },
        "Focality": {
            "Uni-Focal": "uni_focal",
            "Multi-Focal": "multi_focal",
        },
        "Risk": {
            "Low": "low",
            "Intermediate": "intermediate",
            "High": "high",
        },
        "Response": {
            "Excellent": "excellent",
            "Indeterminate": "indeterminate",
            "Structural Incomplete": "structural_incomplete",
            "Biochemical Incomplete": "biochemical_incomplete",
        },
        "Recurred": {
            "Yes": "yes",
            "No": "no",
        }
    }
}

# ---------------------------
# 3. Translation helpers
# ---------------------------

def get_dataset_stem(path: Path) -> str:
    return path.stem


def translate_columns(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    df = df.copy()
    cmap = COLUMN_MAPS.get(dataset_name, {})
    df = df.rename(columns=cmap)
    return df


def translate_values(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    df = df.copy()
    vmaps = VALUE_MAPS.get(dataset_name, {})

    for col, mapping in vmaps.items():
        if col in df.columns and mapping:
            df[col] = df[col].apply(lambda x: mapping.get(x, mapping.get(str(x), x)))
    return df

# ---------------------------
# 4. Dataset-specific cleanups
# ---------------------------

def dataset_specific_cleanup(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    df = df.copy()

    if dataset_name == "질병관리청_인플루엔자 주별 연령별 검출률_20251228":
        needed = ["년도", "주", "연령", "검출률"]
        df = df[[c for c in needed if c in df.columns]]
        df = df.dropna(subset=["년도", "주", "연령", "검출률"])
        df["년도"] = pd.to_numeric(df["년도"], errors="coerce")
        df["주"] = pd.to_numeric(df["주"], errors="coerce")
        df["검출률"] = pd.to_numeric(df["검출률"], errors="coerce")
        df = df.dropna().reset_index(drop=True)

    elif dataset_name == "국세청_근로소득 백분위(천분위) 자료_20251231":
        df = df.dropna(subset=["구분"])
        numeric_cols = [c for c in df.columns if c != "구분"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    elif dataset_name in [
        "한국도로교통공단_도로종류별 기상상태별 교통사고 통계_20241231",
        "한국도로교통공단_사고유형별 교통사고 통계_20241231",
    ]:
        numeric_cols = [c for c in df.columns if "수" in c or "건수" in c]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.reset_index(drop=True)


# ---------------------------
# 5. Main processing
# ---------------------------

def process_one_file(path: Path, group_name: str):
    dataset_name = get_dataset_stem(path)

    df = read_csv_auto(path)
    df = basic_clean(df)
    df = drop_empty_rows_cols(df)
    df = remove_duplicate_rows(df)
    df = dataset_specific_cleanup(df, dataset_name)

    text_cols, numeric_cols = infer_column_types(df)

    # KO cleaned
    out_ko = PROCESSED_DIR / group_name / f"{dataset_name}_ko.csv"
    df.to_csv(out_ko, index=False, encoding="utf-8-sig")

    # EN cleaned
    df_en = translate_values(df, dataset_name)
    df_en = translate_columns(df_en, dataset_name)
    out_en = PROCESSED_DIR / group_name / f"{dataset_name}_en.csv"
    df_en.to_csv(out_en, index=False, encoding="utf-8-sig")

    # metadata
    dataset_info = {
        "dataset_name": dataset_name,
        "group": group_name,
        "num_rows": int(len(df)),
        "num_columns": int(len(df.columns)),
        "text_columns_ko": text_cols,
        "numeric_columns_ko": numeric_cols,
        "columns_ko": list(df.columns),
        "columns_en": list(df_en.columns),
    }

    with open(METADATA_DIR / "dataset_info" / f"{dataset_name}.json", "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)

    with open(METADATA_DIR / "column_maps" / f"{dataset_name}.json", "w", encoding="utf-8") as f:
        json.dump(COLUMN_MAPS.get(dataset_name, {}), f, ensure_ascii=False, indent=2)

    with open(METADATA_DIR / "value_maps" / f"{dataset_name}.json", "w", encoding="utf-8") as f:
        json.dump(VALUE_MAPS.get(dataset_name, {}), f, ensure_ascii=False, indent=2)

    print(f"[DONE] {dataset_name} ({group_name}) rows={len(df)} cols={len(df.columns)}")


def main():
    groups = ["numeric", "text", "mixed"]

    for group in groups:
        src_dir = RAW_DIR / group
        if not src_dir.exists():
            print(f"[SKIP] Missing folder: {src_dir}")
            continue

        for path in sorted(src_dir.glob("*.csv")):
            process_one_file(path, group)


if __name__ == "__main__":
    main()