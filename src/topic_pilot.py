"""
0.5단계: 주제 3개 × 100개를 공식 루브릭(v0)으로 AI 채점 → 사람 점수·기준선과 비교

  python src/topic_pilot.py --estimate   (무료 토큰 계산으로 예상 비용만)
  python src/topic_pilot.py              (실제 채점, 비용 발생)

비교 대상 (같은 100개 글 기준)
  - 길이만 기준선
  - 표면특징 모델: 이 100개를 뺀 나머지에서 200개로 학습
결과: runs/topic_pilot/
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import cohen_kappa_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from asap import build_messages, load, parse, rank_map, surface_features, task_info  # noqa: E402
from llm import GRADER_MODEL, PRICES, call, count_input_tokens, spent_so_far  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "topic_pilot"
TOPICS = ["Car-free cities", "Does the electoral college work?", "The Face on Mars"]
N = 100
SEED = 7


def sample(d, n, seed):
    """점수별 비율을 유지하며 n개 (층화)"""
    rng = np.random.default_rng(seed)
    parts = []
    for s, g in d.groupby("score"):
        k = max(1, round(n * len(g) / len(d)))
        parts.append(g.iloc[rng.permutation(len(g))[:k]])
    out = __import__("pandas").concat(parts)
    return out.iloc[rng.permutation(len(out))[:n]]


def grade(criteria, assignment, sources, row, tag):
    system, user = build_messages(criteria, assignment, sources, row.full_text)
    err = None
    for _ in range(2):
        raw = call(GRADER_MODEL, system, user, max_tokens=400, tag=f"{tag}:{row.essay_id}")
        try:
            s, why = parse(raw)
            return {"id": row.essay_id, "ai": s, "human": int(row.score), "rationale": why, "error": None}
        except Exception as e:
            err = str(e)
    return {"id": row.essay_id, "ai": None, "human": int(row.score), "rationale": None, "error": err}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--estimate", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    df = load()
    # 루브릭 원문만 사용 (파일 맨 위의 한국어 출처·라이선스 메모 줄은 제외)
    criteria = "\n".join(l for l in (ROOT / "prompts" / "rubric_official.md").read_text(encoding="utf-8").splitlines()
                         if not l.startswith(("#", ">")))

    plan = {}
    total_est = 0
    p = PRICES[GRADER_MODEL]
    for t in TOPICS:
        d = df[df.prompt_name == t]
        smp = sample(d, N, SEED)
        assignment, sources = task_info(df, t)
        sys_txt, _ = build_messages(criteria, assignment, sources, "x")
        sys_tok = count_input_tokens(GRADER_MODEL, sys_txt, "x")
        ess_tok = int(smp.words.mean() * 1.35)
        est = (sys_tok * p["cache_write"] + sys_tok * (N - 1) * p["cache_read"]
               + ess_tok * N * p["input"] + 180 * N * p["output"]) / 1e6
        total_est += est
        plan[t] = (d, smp, assignment, sources)
        print(f"{t[:32]:<34} 고정부분 {sys_tok}토큰, 글 평균 약 {ess_tok}토큰 → 예상 ${est:.3f}")
    print(f"합계 예상 ${total_est:.3f} | 현재 누적 ${spent_so_far():.3f}")
    if args.estimate:
        return

    lines = [f"공식 루브릭(v0) 채점 vs 기준선 — 주제별 {N}개 (점수 비율 유지 표본)\n"]
    for t, (d, smp, assignment, sources) in plan.items():
        path = OUT / f"{t[:20].replace(' ', '_').replace('?', '')}.jsonl"
        if path.exists():
            res = [json.loads(l) for l in open(path, encoding="utf-8")]
        else:
            rows = list(smp.itertuples())
            first = grade(criteria, assignment, sources, rows[0], "pilot")
            with ThreadPoolExecutor(max_workers=8) as ex:
                res = [first] + list(ex.map(lambda r: grade(criteria, assignment, sources, r, "pilot"), rows[1:]))
            with open(path, "w", encoding="utf-8") as f:
                for g in res:
                    f.write(json.dumps(g, ensure_ascii=False) + "\n")
            print(f"  {t} 채점 완료, 누적 ${spent_so_far():.3f}")

        ok = [g for g in res if g["error"] is None]
        ai = np.array([g["ai"] for g in ok])
        hu = np.array([g["human"] for g in ok])
        ids = [g["id"] for g in ok]
        # 기준선 (같은 글)
        X_all = surface_features(d.full_text.tolist(), sources)
        idx_of = {e: i for i, e in enumerate(d.essay_id)}
        te = np.array([idx_of[i] for i in ids])
        pool = np.setdiff1d(np.arange(len(d)), te)
        rng = np.random.default_rng(0)
        tr = rng.choice(pool, 200, replace=False)
        y = d.score.values
        m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20))).fit(X_all[tr], y[tr])
        q_surface = cohen_kappa_score(hu, rank_map(m.predict(X_all[te]), y[tr]), weights="quadratic")
        q_len = cohen_kappa_score(hu, rank_map(X_all[te, 0], y[tr]), weights="quadratic")
        q_ai = cohen_kappa_score(hu, ai, weights="quadratic")
        q_ai_cal = cohen_kappa_score(hu, rank_map(ai + np.random.default_rng(1).random(len(ai)) * 1e-3, y[tr]),
                                     weights="quadratic")
        r = np.corrcoef(ai, hu)[0, 1]
        r_len = np.corrcoef(ai, X_all[te, 0])[0, 1]
        lines.append(
            f"[{t}] n={len(ok)} (형식오류 {len(res) - len(ok)})\n"
            f"  AI(v0) QWK {q_ai:.3f} | 순위 보정 시 {q_ai_cal:.3f} | 상관 {r:.3f} | 정확 일치 {np.mean(ai == hu):.0%} "
            f"| ±1 일치 {np.mean(abs(ai - hu) <= 1):.0%} | 평균 AI {ai.mean():.2f} vs 사람 {hu.mean():.2f}\n"
            f"  기준선: 길이만 QWK {q_len:.3f} | 표면특징(200개 학습) QWK {q_surface:.3f}\n"
            f"  AI 점수와 글 길이의 상관 {r_len:.2f}  (사람 점수와 길이의 상관 {np.corrcoef(hu, X_all[te, 0])[0, 1]:.2f})\n"
            f"  AI 점수 분포 {dict(zip(*np.unique(ai, return_counts=True)))} / 사람 {dict(zip(*np.unique(hu, return_counts=True)))}")
    txt = "\n".join(lines)
    print("\n" + txt)
    (OUT / "report.txt").write_text(txt, encoding="utf-8")
    print(f"\n누적 비용 ${spent_so_far():.3f}")


if __name__ == "__main__":
    main()
