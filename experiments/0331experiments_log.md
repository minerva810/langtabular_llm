# [2026-3-31] EXP-002: Baseline + limit up
evaluation.py

## Setup
- limit= 50
- model= gpt-4o-mini
- task
    - aggregation
    - argmax
    - comparison
    - lookup



### Summary
pred_en_json.jsonl: 0.6250
pred_en_kv.jsonl: 0.6875
pred_en_markdown.jsonl: 0.6875
pred_ko_json.jsonl: 0.6875
pred_ko_kv.jsonl: 0.6250
pred_ko_markdown.jsonl: 0.6875

=== EN-KO Gap ===
markdown: EN-KO gap = 0.0000
json: EN-KO gap = -0.0625
kv: EN-KO gap = 0.0625


=== pred_en_json.jsonl ===
Overall Accuracy: 0.6250 (10/16)

By Task
aggregation: 0.0000 (0/4)
argmax: 0.7500 (3/4)
comparison: 0.7500 (3/4)
lookup: 1.0000 (4/4)

By Dataset
child_protection: 0.7500 (3/4)
child_welfare_facility: 0.7500 (3/4)
income_percentile: 0.2500 (1/4)
influenza_detection: 0.7500 (3/4)

=== pred_en_kv.jsonl ===
Overall Accuracy: 0.6875 (11/16)

By Task
aggregation: 0.2500 (1/4)
argmax: 0.7500 (3/4)
comparison: 0.7500 (3/4)
lookup: 1.0000 (4/4)

By Dataset
child_protection: 0.7500 (3/4)
child_welfare_facility: 1.0000 (4/4)
income_percentile: 0.2500 (1/4)
influenza_detection: 0.7500 (3/4)

=== pred_en_markdown.jsonl ===
Overall Accuracy: 0.6875 (11/16)

By Task
aggregation: 0.2500 (1/4)
argmax: 0.5000 (2/4)
comparison: 1.0000 (4/4)
lookup: 1.0000 (4/4)

By Dataset
child_protection: 0.7500 (3/4)
child_welfare_facility: 0.7500 (3/4)
income_percentile: 0.5000 (2/4)
influenza_detection: 0.7500 (3/4)

=== pred_ko_json.jsonl ===
Overall Accuracy: 0.6875 (11/16)

By Task
aggregation: 0.0000 (0/4)
argmax: 0.7500 (3/4)
comparison: 1.0000 (4/4)
lookup: 1.0000 (4/4)

By Dataset
child_protection: 0.7500 (3/4)
child_welfare_facility: 0.7500 (3/4)
income_percentile: 0.5000 (2/4)
influenza_detection: 0.7500 (3/4)

=== pred_ko_kv.jsonl ===
Overall Accuracy: 0.6250 (10/16)

By Task
aggregation: 0.0000 (0/4)
argmax: 0.5000 (2/4)
comparison: 1.0000 (4/4)
lookup: 1.0000 (4/4)

By Dataset
child_protection: 0.7500 (3/4)
child_welfare_facility: 0.7500 (3/4)
income_percentile: 0.5000 (2/4)
influenza_detection: 0.5000 (2/4)

=== pred_ko_markdown.jsonl ===
Overall Accuracy: 0.6875 (11/16)

By Task
aggregation: 0.2500 (1/4)
argmax: 0.5000 (2/4)
comparison: 1.0000 (4/4)
lookup: 1.0000 (4/4)

By Dataset
child_protection: 0.5000 (2/4)
child_welfare_facility: 1.0000 (4/4)
income_percentile: 0.5000 (2/4)
influenza_detection: 0.7500 (3/4)
