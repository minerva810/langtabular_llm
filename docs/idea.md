# Research Idea

## Problem

기존 Tabular + LLM 연구는 대부분 영어 기반으로 진행됨.

하지만 실제 환경에서는:
- 한국어
- 다국어
- 혼합 언어

등 다양한 형태의 table이 존재한다.

👉 질문:
"LLM은 언어가 달라져도 table 구조를 동일하게 이해하는가?"

---

## Hypothesis

1. LLM은 table structure 자체를 이해하는 것이 아니라  
   → language-dependent pattern을 활용한다

2. 따라서:
   - English → 높은 성능
   - Korean → 성능 저하

3. 특히:
   - numeric value → 영향 적음
   - categorical/text value → 영향 큼

---

## Core Research Question

1. Language change가 tabular understanding에 미치는 영향은?
2. 어떤 조건에서 failure가 발생하는가?
3. 구조 이해 vs 언어 이해 중 무엇이 더 중요한가?

---

## Key Idea

"Structure-Preserving Table Serialization"이  
실제로는 language-invariant하지 않을 수 있음

---

## Expected Contribution

- Tabular LLM의 language bias 분석
- 한국어 환경에서의 failure case 제시
- serialization 방식의 한계 분석

---

## Experimental Direction

- English vs Korean 비교
- Header vs Value 분리 실험
- Numeric vs Categorical 분리
- Mixed language 실험

---

## Possible Extension

- multilingual serialization template
- language-agnostic representation
- structured encoding 방식 비교

---