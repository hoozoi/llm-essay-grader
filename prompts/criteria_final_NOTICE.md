# 최종 채점 기준 안내 (criteria_final.md)

## 무엇인가
- 과제: ASAP 2.0 "Car-free cities" (자동차 사용을 줄이는 것의 장점에 대한 자료 기반 설명문), 1~6점 종합 채점
- 채점 모델(Claude Haiku 5.5)에 이 기준 전문을 그대로 넣어 사용한다 (prompts/asap_grade_system.md의 {criteria})

## 어떻게 만들어졌나
1. 출발점(v0): PERSUADE 2.0 자료 기반 Holistic Rating Form 원문
2. 자동 개선 루프: 개선용 200개의 사람 점수와 비교 → Claude Sonnet 5.5가 기준 수정 (v1~v7)
3. 선정: v4 (후보 v3·v4·v7을 2회씩 채점한 평균 QWK 최고, 0.703)
4. 정리: v4에 들어간 학생 글 표현 3곳("maybe this is good" 등)을 일반적인 표현으로 교체. 그 외 문장은 v4와 동일
   - 기준 안의 점수 비율 안내(1점 약 7% …)는 사람 점수 200개의 분포에서 수정 모델이 가져온 보정 정보

## 성능 (v4 기준, 확인용 200개 = 루프에서 한 번도 안 본 글)
- QWK 0.697 [95% 구간 0.62~0.76], 정확 일치 52%, ±1 이내 96%
- 공식 루브릭 그대로(v0) 0.392 → +0.305

## 출처·라이선스
- 원 루브릭: Crossley et al., PERSUADE 2.0 corpus (https://github.com/scrosseye/persuade_corpus_2.0),
  sat_rubric_only_source_based.pdf — **CC BY-NC-SA 4.0**
- 이 기준은 위 루브릭을 고친 2차 저작물이므로 **CC BY-NC-SA 4.0**으로 공개한다 (비상업적 이용, 출처 표기, 동일 조건 변경 허락)
- 데이터: ASAP 2.0 (The Learning Agency, Kaggle), CC BY 4.0
