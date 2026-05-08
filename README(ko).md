# Cross-lingual Table Understanding Benchmark

## 개요 (Overview)

본 프로젝트는 **테이블의 언어(영어 vs 한국어)**가 LLM의 정형 데이터 이해 성능에 미치는 영향을 분석하는 것을 목표로 합니다.

기존의 multilingual QA와 달리, 본 연구는 **질문을 영어로 고정**하고 **테이블 언어만 변화**시켜 다음을 정밀하게 분석합니다:

* cross-lingual table grounding
* serialization 방식의 영향 (Markdown / JSON / KV)
* tabular reasoning에서의 failure 유형

---

## 핵심 아이디어

본 실험은 다음 설정을 사용합니다:

```text
(Q_EN, Table_EN) vs (Q_EN, Table_KO)
```

* 질문: 영어로 고정
* 테이블: 영어 / 한국어 (paired)
* 구조 및 값: 동일

👉 즉, **table representation의 언어 효과만 분리하여 측정**

---

## 데이터셋 구조

```text
dataset/
├─ tables/
│  ├─ controlled_synthetic/
│  ├─ real/
│  └─ stress/
│
├─ metadata/
│  └─ table_metadata.jsonl
│
├─ questions/
│  └─ base_questions.jsonl
│
└─ serialized/
   └─ eval_instances.jsonl
```

---

## 데이터 분할 (비율)

| 구분                   | 비율  | 설명                    |
| -------------------- | --- | --------------------- |
| Controlled Synthetic | 60% | 실험 통제를 위한 직접 생성 데이터   |
| Real Data            | 30% | 외부 CSV 기반 (일반화 성능 평가) |
| Stress Data          | 10% | 실패 유도용 데이터            |

---

## Paired Table 설계

각 테이블은 다음과 같이 쌍으로 존재합니다:

```text
table_xxx_en.csv
table_xxx_ko.csv
```

### 필수 조건

* row 동일
* 값 동일
* 구조 동일
* column name만 번역

👉 실험 변수: **table language only**

---

## 전체 파이프라인

```text
1. Dataset 구축
2. Serialization 변환 (Markdown / JSON / KV)
3. Evaluation Instance 생성
4. LLM Inference
5. 자동 평가
6. Failure 분석
```


project/
├─ dataset/
│  ├─ tables/
│  │  ├─ controlled_synthetic/
│  │  │  ├─ synthetic_001_en.csv
│  │  │  ├─ synthetic_001_ko.csv
│  │  │  └─ ...
│  │  ├─ real/
│  │  │  ├─ real_001_en.csv
│  │  │  ├─ real_001_ko.csv
│  │  │  └─ ...
│  │  └─ stress/
│  │     ├─ stress_001_en.csv
│  │     ├─ stress_001_ko.csv
│  │     └─ ...
│  │
│  ├─ metadata/
│  │  ├─ controlled_synthetic_metadata.jsonl
│  │  ├─ real_metadata.jsonl
│  │  ├─ stress_metadata.jsonl
│  │  └─ table_metadata.jsonl        # 최종 통합
│  │
│  ├─ questions/
│  │  ├─ controlled_synthetic_questions.jsonl
│  │  ├─ real_questions.jsonl
│  │  ├─ stress_questions.jsonl
│  │  └─ base_questions.jsonl        # 최종 통합
│  │
│  └─ serialized/                 
│     ├─ markdown/
│     ├─ json/
│     └─ kv/
│
├─ external_data/
│  └─ raw/
│     ├─ iris.csv
│     ├─ ...
│
├─ scripts/
│  ├─ 01_build_controlled_synthetic.py
│  ├─ 02_ingest_real_datasets.py
│  ├─ 03_build_stress_data.py
│  ├─ 04_merge_dataset.py
│  ├─ 05_build_eval_instances.py   
│  └─ 06_run_inference.py          
│
├─ experiments/
│  ├─ logs/
│  ├─ results/
│  └─ configs/
│
├─ analysis/
│  ├─ notebooks/
│  └─ plots/
│
├─ docs/
│  ├─ dataset_description.md
│  ├─ failure_taxonomy.md
│  └─ experiment_plan.md
│
├─ requirements.txt
└─ README.md
---

## 실행 방법

### 1. 데이터셋 생성

```bash
python scripts/01_build_controlled_synthetic.py
python scripts/02_ingest_real_datasets.py
python scripts/03_build_stress_data.py
python scripts/04_merge_dataset.py
```

---

### 2. Evaluation Instance 생성

```bash
python scripts/05_build_eval_instances.py
```

---

### 3. Inference 실행

```bash
python scripts/06_run_inference.py
```

---

## 평가 단위

각 실험 단위는 다음 조합으로 생성됩니다:

```text
question × table_language × serialization
```

* Language: EN / KO
* Serialization: Markdown / JSON / KV

---

## 출력 결과

### Eval Instances

```text
dataset/serialized/eval_instances.jsonl
```

### Prediction 결과

```text
experiments/predictions/predictions.jsonl
```

---

## 평가 지표

* Exact Match
* Numeric Tolerance
* Accuracy

---

## Failure Taxonomy

다음과 같은 오류 유형으로 분류합니다:

* Header grounding
* Row selection
* Filtering / comparison
* Arithmetic / aggregation
* Serialization parsing
* Language-specific linking
* Instruction / output error

---

## 연구 질문 (RQ)

1. 테이블 언어가 LLM 성능에 영향을 미치는가?
2. Serialization 방식이 이 효과를 어떻게 변화시키는가?
3. 데이터 타입별로 영향이 달라지는가?
4. 성능 저하는 어떤 failure 유형에서 발생하는가?

---

## 설치

```bash
pip install pandas openai tabulate
```

---

## 주의사항

* 모든 질문은 영어로 고정
* 테이블은 EN/KO paired 구조
* 외부 데이터는 변환 후 사용
* 모든 실험은 재현 가능하도록 구성

---

## 기여 (Contribution)

본 연구는 단순 multilingual QA가 아니라:

> **cross-lingual table understanding 문제를 분석**

다음 기여를 포함합니다:

* 통제된 데이터셋 설계
* serialization 기반 비교 분석
* failure taxonomy 기반 정밀 분석

---

## 라이선스

외부 데이터셋은 각 source의 라이선스를 따릅니다.
