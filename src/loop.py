"""
채점 기준 자동 개선 루프 (ASAP 2.0, Car-free cities)

  v0(공식 루브릭)으로 개선용 200개 채점 → 통계 → 지금까지 최고 버전을 Sonnet이 수정
  (수정 이력·통계·크게 틀린 사례 참고) → 새 버전 채점 → ... → v{iters}
  최고 버전 = QWK가 가장 높은 버전 (동점이면 정확 일치율)

실행
  python src/loop.py --estimate        예상 비용만 (API 비용 없음)
  python src/loop.py --upto 1          v1까지만 (단계별 확인용)
  python src/loop.py                   전체 v0~v10. 이미 끝난 부분은 건너뜀(이어하기)

결과: runs/loop/
  criteria_v{k}.md  v{k}_changes.md  v{k}.diff  v{k}_parent.txt
  v{k}_grades.jsonl v{k}_stats.json  summary.csv  best.txt
"""

import argparse
import csv
import difflib
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from asap import build_messages, load, parse, task_info  # noqa: E402
from llm import GRADER_MODEL, REVISER_MODEL, BudgetExceeded, call, spent_so_far  # noqa: E402
from stats import compute_stats, format_case, format_stats, select_cases  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "loop"
PROMPTS = ROOT / "prompts"
TOPIC = "Car-free cities"

COST_PER_GRADE = 0.00025      # 0.5단계 실측 기반 (글 1개)
REVISE_EST = 0.12             # 기준 수정 1회 예상
REVISE_WORST = 0.25           # 기준 수정 1회 최악
REVISE_MAX_OUTPUT = 16_000


def load_rows():
    rows = [json.loads(l) for l in open(ROOT / "data/processed/carfree_improve.jsonl", encoding="utf-8")]
    return rows, {r["id"]: r for r in rows}


def official_rubric():
    txt = (PROMPTS / "rubric_official.md").read_text(encoding="utf-8")
    return "\n".join(l for l in txt.splitlines() if not l.startswith(("#", ">"))).strip()


# ------------------------------------------------------------------ 채점
def grade_one(criteria, assignment, sources, rec, tag):
    system, user = build_messages(criteria, assignment, sources, rec["text"])
    err, raw = None, ""
    for attempt in range(2):
        raw = call(GRADER_MODEL, system, user, max_tokens=400,
                   tag=f"{tag}:{rec['id']}" + ("" if attempt == 0 else ":retry"))
        try:
            s, why = parse(raw)
            return {"id": rec["id"], "ai": s, "rationale": why, "error": None}
        except Exception as e:
            err = str(e)
    return {"id": rec["id"], "ai": None, "rationale": None, "error": err, "raw": raw}


def grade_version(k, criteria, assignment, sources, rows, workers):
    path = OUT / f"v{k}_grades.jsonl"
    done = {}
    if path.exists():
        for l in open(path, encoding="utf-8"):
            g = json.loads(l)
            done[g["id"]] = g
    todo = [r for r in rows if r["id"] not in done]
    if not todo:
        print(f"  v{k}: 이미 채점 완료 ({len(done)}개) → 건너뜀")
        return [done[r["id"]] for r in rows]
    print(f"  v{k}: {len(todo)}개 채점 시작")
    lock = threading.Lock()
    with open(path, "a", encoding="utf-8") as f:
        def save(g):
            with lock:
                f.write(json.dumps(g, ensure_ascii=False) + "\n")
                f.flush()
                done[g["id"]] = g
        save(grade_one(criteria, assignment, sources, todo[0], f"loop_v{k}"))  # 캐시 생성
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(grade_one, criteria, assignment, sources, r, f"loop_v{k}") for r in todo[1:]]
            for i, fu in enumerate(as_completed(futs), 2):
                save(fu.result())
                if i % 100 == 0 or i == len(todo):
                    print(f"    {i}/{len(todo)}  누적 ${spent_so_far():.3f}")
    return [done[r["id"]] for r in rows]


# ------------------------------------------------------------------ 이력·수정
def parent_of(k):
    p = OUT / f"v{k}_parent.txt"
    return int(p.read_text().strip()) if p.exists() else None


def format_history(all_stats):
    L = ["| version | revised from | QWK | exact | within ±1 | model mean (human 3.10) | model std (human ~1.0) |",
         "|---|---|---|---|---|---|---|"]
    for v, s in sorted(all_stats.items()):
        par = parent_of(v)
        L.append(f"| v{v} | {'-' if par is None else f'v{par}'} | {s['qwk']:.3f} | {s['exact']:.0%} | "
                 f"{s['adjacent']:.0%} | {s['ai_mean']:.2f} | {s['ai_std']:.2f} |")
    for v in sorted(all_stats):
        ch = OUT / f"v{v}_changes.md"
        if ch.exists():
            par = parent_of(v)
            delta = all_stats[v]["qwk"] - all_stats[par]["qwk"] if par in all_stats else 0
            L.append(f"\nChanges in v{v} (from v{par}, QWK {delta:+.3f}):\n{ch.read_text(encoding='utf-8').strip()}")
    return "\n".join(L)


def revise(base, new_k, criteria, grades, stats, human, assignment, sources, n_cases, all_stats):
    cases = select_cases(grades, human, k=n_cases)
    cases_txt = "\n\n".join(format_case(i, g, human[g["id"]]) for i, g in enumerate(cases, 1))
    system = (PROMPTS / "asap_revise_system.md").read_text(encoding="utf-8")
    user = (PROMPTS / "asap_revise_user.md").read_text(encoding="utf-8").format(
        assignment=assignment, sources=sources, history=format_history(all_stats),
        version=f"v{base}", criteria=criteria.strip(), stats=format_stats(stats),
        n_cases=len(cases), cases=cases_txt)
    (OUT / f"revise_v{new_k}_input.md").write_text(user, encoding="utf-8")
    for attempt in range(2):
        raw = call(REVISER_MODEL, system, user, max_tokens=REVISE_MAX_OUTPUT,
                   tag=f"revise_v{new_k}" + ("" if attempt == 0 else ":retry"), thinking=True)
        (OUT / f"revise_v{new_k}_raw.txt").write_text(raw, encoding="utf-8")
        m_c = re.search(r"<criteria>(.*?)</criteria>", raw, re.S)
        m_ch = re.search(r"<changes>(.*?)</changes>", raw, re.S)
        if m_c:
            return m_c.group(1).strip(), (m_ch.group(1).strip() if m_ch else "(요약 없음)")
        print(f"  [경고] 수정 응답 형식 오류 (시도 {attempt + 1})")
    raise RuntimeError("수정 모델 응답에서 <criteria>를 찾지 못함")


# ------------------------------------------------------------------ 기록
COLS = ["version", "parent", "qwk", "pearson", "exact", "adjacent", "big_error",
        "over_rate", "under_rate", "ai_mean", "ai_std", "ai_len_corr", "n_error", "criteria_words"]


def write_summary(all_stats, lens):
    with open(OUT / "summary.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLS)
        for k, s in sorted(all_stats.items()):
            par = parent_of(k)
            w.writerow([f"v{k}", "" if par is None else f"v{par}"]
                       + [f"{s[c]:.4f}" if isinstance(s[c], float) else s[c] for c in COLS[2:-1]] + [lens[k]])


def print_row(k, s):
    print(f"  v{k}: QWK {s['qwk']:.3f} | 상관 {s['pearson']:.3f} | 정확 {s['exact']:.0%} | ±1 {s['adjacent']:.0%} "
          f"| 평균 {s['ai_mean']:.2f}(사람 {s['human_mean']:.2f}) | 표준편차 {s['ai_std']:.2f}(사람 {s['human_std']:.2f}) "
          f"| 점수분포 {s['ai_dist']} | 형식오류 {s['n_error']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--upto", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cases", type=int, default=15)
    ap.add_argument("--estimate", action="store_true")
    args = ap.parse_args()
    last = args.iters if args.upto is None else min(args.upto, args.iters)

    OUT.mkdir(parents=True, exist_ok=True)
    rows, human = load_rows()
    assignment, sources = task_info(load(), TOPIC)

    n_grade = sum(len(rows) - (sum(1 for _ in open(OUT / f"v{k}_grades.jsonl", encoding="utf-8"))
                               if (OUT / f"v{k}_grades.jsonl").exists() else 0) for k in range(last + 1))
    n_rev = sum(1 for k in range(1, last + 1) if not (OUT / f"criteria_v{k}.md").exists())
    est = n_grade * COST_PER_GRADE + n_rev * REVISE_EST
    worst = n_grade * COST_PER_GRADE * 1.5 + n_rev * REVISE_WORST
    print(f"작업: v0~v{last} | 채점 {n_grade}회 + 기준 수정 {n_rev}회")
    print(f"예상 ${est:.2f} (최악 ${worst:.2f}) | 현재 누적 ${spent_so_far():.3f} / 예산 $12")
    if args.estimate:
        return

    if not (OUT / "criteria_v0.md").exists():
        (OUT / "criteria_v0.md").write_text(official_rubric(), encoding="utf-8")

    all_stats, all_grades, lens = {}, {}, {}

    def best_so_far():
        return max(all_stats, key=lambda v: (all_stats[v]["qwk"], all_stats[v]["exact"]))

    try:
        for k in range(last + 1):
            criteria = (OUT / f"criteria_v{k}.md").read_text(encoding="utf-8")
            grades = grade_version(k, criteria, assignment, sources, rows, args.workers)
            s = compute_stats(grades, human)
            (OUT / f"v{k}_stats.json").write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding="utf-8")
            all_stats[k], all_grades[k], lens[k] = s, grades, len(criteria.split())
            write_summary(all_stats, lens)
            print_row(k, s)

            if k < last and not (OUT / f"criteria_v{k + 1}.md").exists():
                b = best_so_far()
                base = (OUT / f"criteria_v{b}.md").read_text(encoding="utf-8")
                print(f"  최고 버전 v{b} → v{k + 1} 기준 수정 중 (Sonnet)...")
                new, changes = revise(b, k + 1, base, all_grades[b], all_stats[b], human,
                                      assignment, sources, args.cases, all_stats)
                (OUT / f"criteria_v{k + 1}.md").write_text(new, encoding="utf-8")
                (OUT / f"v{k + 1}_changes.md").write_text(changes, encoding="utf-8")
                (OUT / f"v{k + 1}_parent.txt").write_text(f"{b}\n", encoding="utf-8")
                diff = difflib.unified_diff(base.splitlines(), new.splitlines(),
                                            f"criteria_v{b}.md", f"criteria_v{k + 1}.md", lineterm="")
                (OUT / f"v{k + 1}.diff").write_text("\n".join(diff), encoding="utf-8")
                print(f"  v{k + 1} 수정 요약:\n" + "\n".join("    " + l for l in changes.splitlines()))
    except BudgetExceeded as e:
        print(f"\n[예산 정지] {e}")

    if all_stats:
        b = best_so_far()
        (OUT / "best.txt").write_text(f"v{b}\n", encoding="utf-8")
        print(f"\n지금까지 최고 버전: v{b} (QWK {all_stats[b]['qwk']:.3f})")
    print(f"누적 비용 ${spent_so_far():.3f}")


if __name__ == "__main__":
    main()
