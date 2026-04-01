# [2026-3-29] EXP-001: Baseline
evaluate.py 실행

### Setup
- limit= 5
- model= gpt-4o-mini
- task
    - aggregation
    - argmax
    - comparison
    - lookup

## Summary 
pred_en_json.jsonl: 0.8000
pred_en_kv.jsonl: 0.8000
pred_en_markdown.jsonl: 0.8000
pred_ko_json.jsonl: 0.8000
pred_ko_kv.jsonl: 0.8000
pred_ko_markdown.jsonl: 0.6000

=== EN-KO Gap ===
markdown: EN-KO gap = 0.2000
json: EN-KO gap = 0.0000
kv: EN-KO gap = 0.0000

=== pred_en_json.jsonl ===
Overall Accuracy: 0.8000 (4/5)

### Insight
1. serializaiton 성능은 언어에 따라 달라짐
2. aggregation = 0 (모든 경우 실패) → LLM은 "구조 이해"는 되지만 "수치 집계 reasoning"은 약함

### By Task
aggregation: 0.0000 (0/1)
argmax: 1.0000 (1/1)
comparison: 1.0000 (1/1)
lookup: 1.0000 (2/2)

By Dataset
child_protection: 0.7500 (3/4)
child_welfare_facility: 1.0000 (1/1)

=== pred_en_kv.jsonl ===
Overall Accuracy: 0.8000 (4/5)

By Task
aggregation: 0.0000 (0/1)
argmax: 1.0000 (1/1)
comparison: 1.0000 (1/1)
lookup: 1.0000 (2/2)

By Dataset
child_protection: 0.7500 (3/4)
child_welfare_facility: 1.0000 (1/1)

=== pred_en_markdown.jsonl ===
Overall Accuracy: 0.8000 (4/5)

By Task
aggregation: 0.0000 (0/1)
argmax: 1.0000 (1/1)
comparison: 1.0000 (1/1)
lookup: 1.0000 (2/2)

By Dataset
child_protection: 0.7500 (3/4)
child_welfare_facility: 1.0000 (1/1)

=== pred_ko_json.jsonl ===
Overall Accuracy: 0.8000 (4/5)

By Task
aggregation: 0.0000 (0/1)
argmax: 1.0000 (1/1)
comparison: 1.0000 (1/1)
lookup: 1.0000 (2/2)

By Dataset
child_protection: 0.7500 (3/4)
child_welfare_facility: 1.0000 (1/1)

=== pred_ko_kv.jsonl ===
Overall Accuracy: 0.8000 (4/5)

By Task
aggregation: 0.0000 (0/1)
argmax: 1.0000 (1/1)
comparison: 1.0000 (1/1)
lookup: 1.0000 (2/2)

By Dataset
child_protection: 0.7500 (3/4)
child_welfare_facility: 1.0000 (1/1)

=== pred_ko_markdown.jsonl ===
Overall Accuracy: 0.6000 (3/5)

By Task
aggregation: 0.0000 (0/1)
argmax: 0.0000 (0/1)
comparison: 1.0000 (1/1)
lookup: 1.0000 (2/2)

By Dataset
child_protection: 0.5000 (2/4)
child_welfare_facility: 1.0000 (1/1)

