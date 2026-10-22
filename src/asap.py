"""
ASAP 2.0 공용 함수 (API 호출 없음)
- load(): 데이터 읽기
- task_info(): 주제별 과제 문구 + 지문
- build_messages(): 채점 프롬프트 조립 (과제·지문·기준 = system, 학생 글 = user → 캐싱)
- parse(): AI 응답(JSON)에서 점수 꺼내기
- surface_features(): 표면 특징 7개 (길이, 어휘 다양성, 문장 수, 지문 베끼기 등) — 기준선용
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "asap" / "ASAP2_train_sourcetexts.csv"
PROMPTS = ROOT / "prompts"


def load() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    df["words"] = df.full_text.str.split().str.len()
    return df


def task_info(df, prompt_name):
    d = df[df.prompt_name == prompt_name]
    sources = []
    for i in range(1, 5):
        s = d[f"source_text_{i}"].dropna()
        if len(s):
            sources.append(f"--- Source {i} ---\n{s.iloc[0].strip()}")
    return d.assignment.iloc[0].strip(), "\n\n".join(sources)


def build_messages(criteria, assignment, sources, essay):
    tpl = (PROMPTS / "asap_grade_system.md").read_text(encoding="utf-8")
    system = tpl.format(assignment=assignment, sources=sources, criteria=criteria.strip())
    user = f"[Student essay]\n{essay.strip()}"
    return system, user


def parse(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("JSON 없음")
    d = json.loads(m.group(0))
    s = d["score"]
    if s not in (1, 2, 3, 4, 5, 6):
        raise ValueError(f"점수 범위 오류: {s}")
    return int(s), d.get("rationale", "")


def _ngrams(t, n=5):
    w = re.findall(r"[a-z']+", t.lower())
    return set(zip(*[w[i:] for i in range(n)]))


def surface_features(texts, sources_text):
    sg = _ngrams(sources_text)
    X = []
    for t in texts:
        w = t.split()
        s = [x for x in re.split(r"[.!?]+", t) if x.strip()]
        g = _ngrams(t)
        X.append([len(w), len(set(x.lower() for x in w)) / max(len(w), 1),
                  np.mean([len(x) for x in w]) if w else 0, len(s), len(w) / max(len(s), 1),
                  t.count("\n\n") + 1, len(g & sg) / max(len(g), 1)])
    return np.array(X)


def rank_map(pred, train_scores):
    """예측값의 순위를 학습 데이터 점수 분포 비율대로 1~6점에 배정 (기준선용 보정)"""
    n = len(pred)
    order = pd.Series(pred).rank(method="first").values / n
    cum = pd.Series(train_scores).value_counts(normalize=True).sort_index().cumsum()
    return cum.index.values[np.clip(np.searchsorted(cum.values, order), 0, len(cum) - 1)]
