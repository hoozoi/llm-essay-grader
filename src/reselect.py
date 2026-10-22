"""
상위 후보 재선정: 흔들림을 줄이려고 후보마다 개선용 200개를 2회 채점한 평균 QWK로 다시 고릅니다.
(확인용 데이터는 고르는 데 쓰지 않음)

  python src/reselect.py --estimate
  python src/reselect.py
결과: runs/eval/reselect.txt, runs/loop/best_avg.txt
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from asap import load, task_info  # noqa: E402
from llm import spent_so_far  # noqa: E402
from loop import COST_PER_GRADE, TOPIC, grade_one  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LOOP, OUT = ROOT / "runs" / "loop", ROOT / "runs" / "eval"
CANDIDATES = ["v3", "v4", "v7"]
RUNS = 2


def run_files(v):
    files = [LOOP / f"{v}_grades.jsonl"]
    files += sorted(OUT.glob(f"{v}_improve_run*.jsonl"))
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--estimate", action="store_true")
    args = ap.parse_args()
    improve = [json.loads(l) for l in open(ROOT / "data/processed/carfree_improve.jsonl", encoding="utf-8")]
    need = {v: max(0, RUNS - len(run_files(v))) for v in CANDIDATES}
    n = sum(need.values()) * len(improve)
    print(f"추가 채점: {need} → {n}회, 예상 ${n * COST_PER_GRADE:.2f} | 현재 누적 ${spent_so_far():.3f}")
    if args.estimate:
        return

    assignment, sources = task_info(load(), TOPIC)
    for v, k in need.items():
        crit = (LOOP / f"criteria_{v}.md").read_text(encoding="utf-8")
        for _ in range(k):
            idx = len(run_files(v)) + 1
            first = grade_one(crit, assignment, sources, improve[0], f"reselect_{v}")
            with ThreadPoolExecutor(max_workers=8) as ex:
                res = [first] + list(ex.map(lambda r: grade_one(crit, assignment, sources, r, f"reselect_{v}"),
                                            improve[1:]))
            with open(OUT / f"{v}_improve_run{idx}.jsonl", "w", encoding="utf-8") as f:
                for g in res:
                    f.write(json.dumps(g, ensure_ascii=False) + "\n")
            print(f"  {v} {idx}회차 완료, 누적 ${spent_so_far():.3f}")

    human = {r["id"]: r["score"] for r in improve}
    L = [f"상위 후보 재선정 — 개선용 200개, 후보당 {RUNS}회 채점"]
    means = {}
    for v in CANDIDATES:
        qs = []
        for fp in run_files(v)[:RUNS]:
            g = [json.loads(l) for l in open(fp, encoding="utf-8")]
            g = [x for x in g if x["error"] is None]
            qs.append(cohen_kappa_score([human[x["id"]] for x in g], [x["ai"] for x in g], weights="quadratic"))
        means[v] = float(np.mean(qs))
        L.append(f"  {v}: 회차별 QWK {', '.join(f'{q:.3f}' for q in qs)} → 평균 {means[v]:.3f}")
    best = max(means, key=means.get)
    gap = sorted(means.values())[-1] - sorted(means.values())[-2]
    L.append(f"\n평균 기준 최고: {best} (2위와 차이 {gap:.3f})")
    txt = "\n".join(L)
    print("\n" + txt)
    (OUT / "reselect.txt").write_text(txt, encoding="utf-8")
    (LOOP / "best_avg.txt").write_text(f"{best}\n", encoding="utf-8")
    print(f"누적 비용 ${spent_so_far():.3f}")


if __name__ == "__main__":
    main()
