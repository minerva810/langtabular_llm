# Cross-lingual Table Understanding Benchmark

## Overview

This project investigates how **table language (English vs Korean)** affects LLM performance in tabular understanding tasks.

Unlike typical multilingual QA setups, this work **fixes the question in English** and varies only the **table language**, enabling controlled analysis of:

* cross-lingual table grounding
* serialization effects (Markdown / JSON / KV)
* failure modes in tabular reasoning

---

## Key Idea

We evaluate the following setting:

```
(Q_EN, Table_EN) vs (Q_EN, Table_KO)
```

* Question: fixed (English)
* Table: English vs Korean (paired)
* Structure & values: identical

This isolates the effect of **table language representation**.

---

## Dataset Structure

```
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

### Data Splits

| Split                | Ratio | Description                           |
| -------------------- | ----- | ------------------------------------- |
| Controlled Synthetic | 60%   | Fully controlled tables (main result) |
| Real Data            | 30%   | External datasets (generalization)    |
| Stress Data          | 10%   | Failure-inducing tables               |

---

## Paired Table Design

Each table exists as a pair:

```
table_xxx_en.csv
table_xxx_ko.csv
```

Constraints:

* identical rows
* identical values
* identical structure
* only column names translated

---

## Pipeline

```
1. Dataset Construction
2. Serialization (Markdown / JSON / KV)
3. Evaluation Instance Generation
4. LLM Inference
5. Automatic Evaluation
6. Failure Analysis
```

---

## Scripts

### 1. Build Dataset

```bash
python scripts/01_build_controlled_synthetic.py
python scripts/02_ingest_real_datasets.py
python scripts/03_build_stress_data.py
python scripts/04_merge_dataset.py
```

### 2. Build Evaluation Instances

```bash
python scripts/05_build_eval_instances.py
```

### 3. Run Inference

```bash
python scripts/06_run_inference.py
```

---

## Evaluation Setup

Each instance:

```
question × table_language × serialization
```

Total combinations:

* Language: EN / KO
* Serialization: Markdown / JSON / KV

---

## Output

### Eval Instances

```
dataset/serialized/eval_instances.jsonl
```

### Predictions

```
experiments/predictions/predictions.jsonl
```

---

## Metrics

* Exact Match
* Numeric Tolerance
* Accuracy

---

## Failure Taxonomy

We categorize errors into:

* Header grounding
* Row selection
* Filtering / comparison
* Arithmetic / aggregation
* Serialization parsing
* Language-specific linking
* Instruction / output error

---

## Research Questions

1. Does table language affect LLM performance?
2. Does serialization modulate this effect?
3. Does the effect vary across data types?
4. What failure modes explain the gap?

---

## Requirements

```bash
pip install pandas openai tabulate
```

---

## Notes

* All questions are in English
* Tables are paired (EN/KO)
* External datasets are preserved and transformed
* Reproducibility is ensured via scripts

---

## Contribution

This project focuses on:

> **cross-lingual table understanding**, not general multilingual QA

It provides:

* controlled dataset construction
* serialization-aware evaluation
* fine-grained failure analysis

---

## License

Check each real dataset source for license details.


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