"""
ASAP 채점 결과 → 일치도·오답 통계, 틀린 사례 고르기 (API 호출 없음)

grades: [{"id", "ai", "rationale", "error"}]
human:  {id: {"score", "text", "words"}}
오답 기준: AI 점수 ≠ 사람 점수 (정확 일치 기준). 2점 이상 차이 = 크게 틀림.
"""

import numpy as np
from sklearn.metrics import cohen_kappa_score


def compute_stats(grades, human):
    ok = [g for g in grades if g.get("error") is None]
    ai = np.array([g["ai"] for g in ok])
    hu = np.array([human[g["id"]]["score"] for g in ok])
    words = np.array([human[g["id"]]["words"] for g in ok])
    d = ai - hu
    s = {
        "n": len(grades), "n_error": len(grades) - len(ok),
        "qwk": float(cohen_kappa_score(hu, ai, weights="quadratic")) if len(set(ai)) > 1 else 0.0,
        "pearson": float(np.corrcoef(ai, hu)[0, 1]) if ai.std() > 0 else 0.0,
        "exact": float(np.mean(d == 0)),
        "adjacent": float(np.mean(np.abs(d) <= 1)),
        "big_error": float(np.mean(np.abs(d) >= 2)),
        "over_rate": float(np.mean(d > 0)), "under_rate": float(np.mean(d < 0)),
        "mean_diff": float(d.mean()),
        "ai_mean": float(ai.mean()), "ai_std": float(ai.std()),
        "human_mean": float(hu.mean()), "human_std": float(hu.std()),
        "ai_len_corr": float(np.corrcoef(ai, words)[0, 1]) if ai.std() > 0 else 0.0,
        "human_len_corr": float(np.corrcoef(hu, words)[0, 1]),
        "ai_dist": {int(k): int((ai == k).sum()) for k in range(1, 7)},
        "by_level": [],
    }
    for lv in range(1, 7):
        m = hu == lv
        if m.sum() == 0:
            continue
        s["by_level"].append({"human": lv, "n": int(m.sum()), "ai_mean": float(ai[m].mean()),
                              "exact": float(np.mean(ai[m] == lv)),
                              "ai_dist": {int(k): int((ai[m] == k).sum()) for k in range(1, 7) if (ai[m] == k).sum()}})
    return s


def format_stats(s) -> str:
    L = [f"- Essays scored: {s['n']} (format errors excluded: {s['n_error']})",
         f"- QWK {s['qwk']:.3f} | Pearson r {s['pearson']:.3f}",
         f"- Exact agreement {s['exact']:.0%} | within ±1 {s['adjacent']:.0%} | off by 2+ {s['big_error']:.0%}",
         f"- Over-scored {s['over_rate']:.0%} | under-scored {s['under_rate']:.0%} | mean difference (model - human) {s['mean_diff']:+.2f}",
         f"- Score mean: model {s['ai_mean']:.2f} vs human {s['human_mean']:.2f} | "
         f"std: model {s['ai_std']:.2f} vs human {s['human_std']:.2f}",
         f"- Model score distribution: {s['ai_dist']}",
         "", "[By human score level]"]
    for b in s["by_level"]:
        L.append(f"- Human score {b['human']} ({b['n']} essays): model mean {b['ai_mean']:.2f}, "
                 f"exact {b['exact']:.0%}, model gave {b['ai_dist']}")
    return "\n".join(L)


def select_cases(grades, human, k=15):
    """크게 틀린 순으로 k개. 과대·과소평가가 둘 다 있으면 비율대로 섞되 각 최소 3개."""
    over, under = [], []
    for g in grades:
        if g.get("error"):
            continue
        diff = g["ai"] - human[g["id"]]["score"]
        if diff > 0:
            over.append((diff, g))
        elif diff < 0:
            under.append((-diff, g))
    # 같은 차이면 사람 점수 극단(1·5·6) 우선 → 점수 압축 사례가 잘 보이게
    key = lambda x: (-x[0], -abs(human[x[1]["id"]]["score"] - 3.5))
    over.sort(key=key)
    under.sort(key=key)
    total = len(over) + len(under)
    if total == 0:
        return []
    k_over = round(k * len(over) / total)
    if over and under:
        k_over = min(max(k_over, 3), k - 3)
    k_over = min(k_over, len(over))
    k_under = min(k - k_over, len(under))
    k_over = min(k - k_under, len(over))
    return [g for _, g in over[:k_over]] + [g for _, g in under[:k_under]]


def format_case(i, g, h) -> str:
    kind = "OVER-scored" if g["ai"] > h["score"] else "UNDER-scored"
    return (f"### Case {i} ({kind}): human {h['score']} / model {g['ai']} / {h['words']} words\n"
            f"Model rationale: {g['rationale']}\n"
            f"Essay:\n{h['text'].strip()}")
