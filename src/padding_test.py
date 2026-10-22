"""
길이 부풀리기 강건성 테스트: 내용 없이 길이만 늘린 글에 각 채점기가 속는가?

- 대상: 자동채점용(pool) 묶음의 사람 1~2점 글 30개 (1점 10, 2점 20) — 학습·기준 개선에 안 쓴 글
- 부풀리기 (원본 길이만큼 덧붙여 약 2배)
  A. 지문 짜깁기: 지문 문장을 무작위로 그대로 이어 붙임
  B. 자기 반복: 그 글의 문장을 순서만 섞어 한 번 더 붙임
- 채점기: 표면특징 모델(개선용 200개로 학습, 1~6 반올림) vs AI(criteria_final.md, Haiku)

  python src/padding_test.py --out <폴더> [--estimate]
"""

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from asap import load, surface_features, task_info  # noqa: E402
from llm import spent_so_far  # noqa: E402
from loop import COST_PER_GRADE, TOPIC, grade_one  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RNG = np.random.default_rng(11)


def sentences(t):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", t.strip()) if len(s.split()) >= 4]


def pad_sources(essay, src_sents):
    target, add = len(essay.split()), []
    while sum(len(s.split()) for s in add) < target:
        add.append(src_sents[RNG.integers(len(src_sents))])
    return essay.rstrip() + "\n\n" + " ".join(add)


def pad_repeat(essay):
    s = sentences(essay) or [essay]
    return essay.rstrip() + "\n\n" + " ".join(RNG.permutation(s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--estimate", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    pool = [json.loads(l) for l in open(ROOT / "data/processed/carfree_pool.jsonl", encoding="utf-8")]
    picks = []
    for lv, k in [(1, 10), (2, 20)]:
        cand = [r for r in pool if r["score"] == lv]
        picks += [cand[i] for i in RNG.permutation(len(cand))[:k]]
    print(f"대상 {len(picks)}개 × 3버전 = {len(picks) * 3}회 채점, 예상 ${len(picks) * 3 * COST_PER_GRADE * 1.3:.3f} "
          f"| 현재 누적 ${spent_so_far():.3f}")
    if args.estimate:
        return

    df = load()
    assignment, sources = task_info(df, TOPIC)
    src_sents = sentences(re.sub(r"--- Source \d+ ---", " ", sources))
    items = []
    for r in picks:
        for kind, text in [("원본", r["text"]), ("A 지문 짜깁기", pad_sources(r["text"], src_sents)),
                           ("B 자기 반복", pad_repeat(r["text"]))]:
            items.append({"id": f"{r['id']}|{kind}", "base": r["id"], "kind": kind, "human": r["score"],
                          "text": text, "words": len(text.split())})

    # 단순 모델 (개선용 200개로 학습, 절대 점수로 반올림)
    imp = [json.loads(l) for l in open(ROOT / "data/processed/carfree_improve.jsonl", encoding="utf-8")]
    m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20))).fit(
        surface_features([r["text"] for r in imp], sources), [r["score"] for r in imp])
    raw = m.predict(surface_features([it["text"] for it in items], sources))
    for it, p in zip(items, raw):
        it["surface_raw"], it["surface"] = float(p), int(np.clip(np.rint(p), 1, 6))

    # AI (최종 기준)
    crit = (ROOT / "prompts/criteria_final.md").read_text(encoding="utf-8")
    recs = [{"id": it["id"], "text": it["text"]} for it in items]
    first = grade_one(crit, assignment, sources, recs[0], "padding")
    with ThreadPoolExecutor(max_workers=8) as ex:
        res = [first] + list(ex.map(lambda r: grade_one(crit, assignment, sources, r, "padding"), recs[1:]))
    by = {g["id"]: g for g in res}
    for it in items:
        it["ai"], it["ai_rationale"] = by[it["id"]]["ai"], by[it["id"]]["rationale"]
    with open(out / "padding_items.jsonl", "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps({k: v for k, v in it.items() if k != "text"}, ensure_ascii=False) + "\n")

    # 요약
    L = ["길이 부풀리기 테스트 — 사람 1~2점 글 30개 (자동채점용 묶음, 학습·개선에 안 쓴 글)\n",
         f"{'버전':<12}{'평균 단어':>8}{'단순모델 평균':>12}{'AI 평균':>9}{'단순모델 오른 비율':>14}{'AI 오른 비율':>12}"]
    base = {it["base"]: it for it in items if it["kind"] == "원본"}
    for kind in ["원본", "A 지문 짜깁기", "B 자기 반복"]:
        g = [it for it in items if it["kind"] == kind and it["ai"] is not None]
        s_up = np.mean([it["surface"] > base[it["base"]]["surface"] for it in g]) if kind != "원본" else 0
        a_up = np.mean([it["ai"] > base[it["base"]]["ai"] for it in g if base[it["base"]]["ai"]]) if kind != "원본" else 0
        L.append(f"{kind:<12}{np.mean([it['words'] for it in g]):>8.0f}{np.mean([it['surface'] for it in g]):>12.2f}"
                 f"{np.mean([it['ai'] for it in g]):>9.2f}{s_up:>14.0%}{a_up:>12.0%}")
    L.append(f"\n사람 점수 평균: {np.mean([b['human'] for b in base.values()]):.2f}")
    for kind in ["A 지문 짜깁기", "B 자기 반복"]:
        g = [it for it in items if it["kind"] == kind and it["ai"] is not None]
        ds = [it["surface"] - base[it["base"]]["surface"] for it in g]
        da = [it["ai"] - base[it["base"]]["ai"] for it in g if base[it["base"]]["ai"]]
        L.append(f"{kind}: 점수 변화 평균 — 단순모델 {np.mean(ds):+.2f}점, AI {np.mean(da):+.2f}점 "
                 f"| 4점 이상 받은 글 — 단순모델 {sum(it['surface'] >= 4 for it in g)}개, AI {sum(it['ai'] >= 4 for it in g)}개")
    txt = "\n".join(L)
    print("\n" + txt)
    (out / "padding_report.txt").write_text(txt, encoding="utf-8")
    print(f"\n누적 비용 ${spent_so_far():.3f}")


if __name__ == "__main__":
    main()
