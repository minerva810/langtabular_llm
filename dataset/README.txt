dataset/
├─ tables/
│  ├─ controlled_synthetic/
│  ├─ real/
│  └─ stress/
├─ questions/
│  └─ base_questions.jsonl
└─ metadata/
   └─ table_metadata.jsonl

scripts/
├─ 01_build_controlled_synthetic.py
├─ 02_load_real_datasets.py
├─ 03_build_stress_data.py
└─ 04_merge_dataset.py


01_build_controlled_synthetic.py
→ 직접 설계한 synthetic EN/KO CSV 생성
→ controlled_synthetic_metadata.jsonl
→ controlled_synthetic_questions.jsonl

02_load_real_datasets.py
→ 외부 CSV 불러오기
→ column subset 선택
→ EN/KO paired CSV 생성
→ real_metadata.jsonl
→ real_questions.jsonl

03_build_stress_data.py
→ ambiguity, 유사 column, 긴 text 등 stress CSV 생성
→ stress_metadata.jsonl
→ stress_questions.jsonl

04_merge_dataset.py
→ 위 3개 metadata/question 파일 병합
→ dataset/metadata/table_metadata.jsonl
→ dataset/questions/base_questions.jsonl