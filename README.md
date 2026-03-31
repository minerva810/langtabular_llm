# langtabular_llm

한국어를 포함한 언어적 차이가 LLM의 tabular understanding 성능에 어떤 영향을 미치는지 분석하기 위한 실험 저장소.

## Overview

이 프로젝트는 표 데이터를 텍스트로 직렬화 (serialization)하여 LLM에 입력할 때,  
언어(예: 영어 / 한국어) 및 표현 방식의 차이가 모델의 해석 성능과 오류 패턴에 미치는 영향을 분석하는 것을 목표로 합니다.

주요 관심사는 다음과 같습니다.

- 언어별 table serialization 방식 비교
- 한국어 환경에서의 tabular understanding 성능 분석
- 성능 저하가 발생하는 failure case 탐색
- benchmark 결과 및 prediction 결과 관리

## Repository Structure
```bash

project/
├─ data/
│  ├─ 국세청_근로소득 백분위....csv
│  ├─ 인천광역시_보호대상...csv
│  ├─ 인천광역시_아동복지...csv
│  ├─ 질병관리청_인플루엔자...csv
├─ prompts/
│  ├─ base_prompt.txt
├─ src/
│  ├─ serialize.py
│  ├─ build_kor_tabular_benchmark.py
│  ├─ model_api.py
│  ├─ evaluate.py
│  ├─ run_experiment.py
│  └─ error_analysis.py
├─ benchmark_outputs/
│   ├─ tables_ko.jsonl
│   ├─ tables_en.jsonl
│   ├─ qas_paired.jsonl
│   └─ column_maps.json
├─ configs/
├─ experiments/
├─ requirements.txt       # 실행 환경 패키지
├─ README.md             # 프로젝트 설명
└─ .gitignore