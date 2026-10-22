"""
검증 단계
 ① 흔들림 측정: 최고 버전(v4)으로 개선용 200개를 한 번 더 채점 → 1회차와 비교
 ② 최종 평가: 확인용 200개(루프에서 한 번도 안 본 글)를 v0(공식 루브릭)과 최고 버전으로 채점
    + 기준선(길이만 / 표면특징 모델: 개선용 200개로 학습)과 비교, 부트스트랩 95% 신뢰구간

  python src/evaluate.py --estimate
  python src/evaluate.py
결과: runs/eval/
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
from asap import load, rank_map, surface_features, task_info  # noqa: E402
from llm import spent_so_far  # noqa: E402
from loop import COST_PER_GRADE, TOPIC, grade_one  # noqa: E402
from stats import compute_stats  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LOOP = ROOT / "runs" / "loop"
OUT = ROOT / "runs" / "eval"
HUMAN_QWK = 0.745  # 사람끼리 (ASAP 2.0 논문 보고값)
RNG = np.random.default_rng(0)


def read(name):
    return [json.loads(l) for l in open(ROOT / "data/processed" / f"carfree_{name}.jsonl", encoding="utf-8")]


def run_grades(path, criteria, assignment, sources, rows, tag):
    if path.exists():
        return [json.loads(l) for l in open(path, encoding="utf-8")]
    first = grade_one(criteria, assignment, sources, rows[0], tag)
    with ThreadPoolExecutor(max_workers=8) as ex:
        res = [first] + list(ex.map(lambda r: grade_one(criteria, assignment, sources, r, tag), rows[1:]))
    with open(path, "w", encoding="utf-8") as f:
        for g in res:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
    print(f"  {path.name} 완료, 누적 ${spent_so_far():.3f}")
    return res


def dist(a):
    return {int(k): int(v) for k, v in zip(*np.unique(a, return_counts=True))}


def qwk(a, b):
    return cohen_kappa_score(a, b, weights="quadratic")


def boot_ci(h, preds, n=2000):
    """preds: {name: array}. 각 QWK의 95% 구간 + 같은 표본에서 짝지은 차이 계산용 표본 반환"""
    idx = RNG.integers(0, len(h), (n, len(h)))
    samples = {k: np.array([qwk(h[i], p[i]) for i in idx]) for k, p in preds.items()}
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--estimate", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    best = (LOOP / "best.txt").read_text().strip()
    improve, test = read("improve"), read("test")
    n_calls = (0 if (OUT / f"{best}_improve_run2.jsonl").exists() else len(improve)) \
        + (0 if (OUT / "v0_test.jsonl").exists() else len(test)) \
        + (0 if (OUT / f"{best}_test.jsonl").exists() else len(test))
    print(f"최고 버전 {best} | 채점 {n_calls}회 예상 ${n_calls * COST_PER_GRADE:.2f} | 현재 누적 ${spent_so_far():.3f}")
    if args.estimate:
        return

    df = load()
    assignment, sources = task_info(df, TOPIC)
    crit = {v: (LOOP / f"criteria_{v}.md").read_text(encoding="utf-8") for v in ["v0", best]}
    L = []

    # ① 흔들림
    r1 = {g["id"]: g for g in (json.loads(l) for l in open(LOOP / f"{best}_grades.jsonl", encoding="utf-8"))}
    r2 = {g["id"]: g for g in run_grades(OUT / f"{best}_improve_run2.jsonl", crit[best], assignment, sources,
                                         improve, f"noise_{best}")}
    hum_i = {r["id"]: r for r in improve}
    ids = [i for i in r1 if r1[i]["error"] is None and r2.get(i, {}).get("error") is None]
    a1 = np.array([r1[i]["ai"] for i in ids]); a2 = np.array([r2[i]["ai"] for i in ids])
    hi = np.array([hum_i[i]["score"] for i in ids])
    L.append(f"① 흔들림 측정 — {best}, 개선용 {len(ids)}개를 두 번 채점")
    L.append(f"  1회차 QWK {qwk(hi, a1):.3f} | 2회차 QWK {qwk(hi, a2):.3f} | 차이 {qwk(hi, a2) - qwk(hi, a1):+.3f}")
    L.append(f"  두 회차가 같은 점수를 준 비율 {np.mean(a1 == a2):.0%} | 회차끼리 QWK {qwk(a1, a2):.3f}")

    # ② 최종 평가 (확인용)
    t_v0 = run_grades(OUT / "v0_test.jsonl", crit["v0"], assignment, sources, test, "test_v0")
    t_best = run_grades(OUT / f"{best}_test.jsonl", crit[best], assignment, sources, test, f"test_{best}")
    hum_t = {r["id"]: r for r in test}
    ok = [r["id"] for r in test if all(g["error"] is None for g in
                                         [next(x for x in t_v0 if x["id"] == r["id"]),
                                          next(x for x in t_best if x["id"] == r["id"])])]
    gv0 = {g["id"]: g for g in t_v0}; gb = {g["id"]: g for g in t_best}
    h = np.array([hum_t[i]["score"] for i in ok])
    p_v0 = np.array([gv0[i]["ai"] for i in ok]); p_b = np.array([gb[i]["ai"] for i in ok])

    # 기준선: 개선용 200개(사람 점수 있는 유일한 데이터)로 학습
    Xi = surface_features([r["text"] for r in improve], sources); yi = np.array([r["score"] for r in improve])
    Xt = surface_features([hum_t[i]["text"] for i in ok], sources)
    m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20))).fit(Xi, yi)
    p_surf = rank_map(m.predict(Xt), yi)
    p_len = rank_map(Xt[:, 0], yi)
    p_mode = np.full(len(h), int(np.bincount(yi).argmax()))

    preds = {"v0 (공식 루브릭)": p_v0, f"{best} (자동 개선)": p_b, "표면특징 모델": p_surf, "길이만": p_len}
    S = boot_ci(h, preds)
    L.append(f"\n② 최종 평가 — 확인용 {len(ok)}개 (루프에서 한 번도 안 본 글)")
    L.append(f"  {'방법':<16}{'QWK [95% 구간]':<24}{'정확':>6}{'±1':>6}{'2점↑틀림':>9}{'상관':>7}")
    for k, p in preds.items():
        lo, hi_ = np.percentile(S[k], [2.5, 97.5])
        L.append(f"  {k:<16}{qwk(h, p):.3f} [{lo:.2f}~{hi_:.2f}]{'':<5}{np.mean(p == h):>6.0%}"
                 f"{np.mean(abs(p - h) <= 1):>6.0%}{np.mean(abs(p - h) >= 2):>9.0%}{np.corrcoef(p, h)[0, 1]:>7.3f}")
    L.append(f"  {'최빈값 찍기':<16}{qwk(h, p_mode):.3f}{'':<20}{np.mean(p_mode == h):>6.0%}")
    L.append(f"  참고: 사람끼리 QWK {HUMAN_QWK} (논문 보고값)")
    kb, k0, ks = f"{best} (자동 개선)", "v0 (공식 루브릭)", "표면특징 모델"
    for a, b in [(kb, k0), (kb, ks)]:
        d = S[a] - S[b]
        L.append(f"  {a} − {b}: {np.mean(d):+.3f} [95% 구간 {np.percentile(d, 2.5):+.3f}~{np.percentile(d, 97.5):+.3f}], "
                 f"더 높을 확률 {np.mean(d > 0):.0%}")
    L.append(f"  사람 대비 수준: {best} {qwk(h, p_b) / HUMAN_QWK:.0%}, v0 {qwk(h, p_v0) / HUMAN_QWK:.0%}")

    L.append("\n  사람 점수별 정확 일치 (v0 → 최고 버전)")
    for lv in range(1, 7):
        mk = h == lv
        if mk.sum():
            L.append(f"   사람 {lv}점 ({mk.sum()}개): {np.mean(p_v0[mk] == lv):.0%} → {np.mean(p_b[mk] == lv):.0%} "
                     f"| 평균 {p_v0[mk].mean():.2f} → {p_b[mk].mean():.2f}")
    words = np.array([hum_t[i]["words"] for i in ok])
    L.append(f"\n  글 길이와의 상관: 사람 {np.corrcoef(h, words)[0, 1]:.2f} | v0 {np.corrcoef(p_v0, words)[0, 1]:.2f} "
             f"| {best} {np.corrcoef(p_b, words)[0, 1]:.2f} | 표면특징 {np.corrcoef(p_surf, words)[0, 1]:.2f}")
    L.append(f"  점수 분포: 사람 {dist(h)}")
    L.append(f"            {best} {dist(p_b)}")
    L.append(f"            v0 {dist(p_v0)}")

    txt = "\n".join(L)
    print("\n" + txt)
    (OUT / "report.txt").write_text(txt, encoding="utf-8")
    json.dump({"best": best, "test_ids": ok}, open(OUT / "meta.json", "w"))
    print(f"\n누적 비용 ${spent_so_far():.3f}")


if __name__ == "__main__":
    main()
