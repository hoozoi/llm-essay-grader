"""
Car-free cities 글을 개선용 / 확인용 / 자동채점용으로 나눕니다. (API 호출 없음)

- 개선용 200: 채점 기준을 고치는 데 사용
- 확인용 200: 마지막에 한 번만 사용 (최종 성능 보고)
- 자동채점용 나머지: "사람이 채점 안 한 글"로 보고 자동 채점 시연 (실제 점수는 효과 측정에만)
점수(1~6)별 비율을 유지하며 뽑습니다(층화). SEED 고정 → 다시 돌려도 같은 결과.

실행: python src/split.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from asap import load  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "processed"
TOPIC = "Car-free cities"
SIZES = {"improve": 200, "test": 200}
SEED = 42


def main():
    df = load()
    d = df[df.prompt_name == TOPIC].sort_values("essay_id").reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    split = np.array(["pool"] * len(d), dtype=object)
    for s, g in d.groupby("score"):
        idx = g.index.values[rng.permutation(len(g))]
        k_imp = round(len(g) * SIZES["improve"] / len(d))
        k_test = round(len(g) * SIZES["test"] / len(d))
        split[idx[:k_imp]] = "improve"
        split[idx[k_imp:k_imp + k_test]] = "test"
    # 반올림 오차 보정: 정확히 200개씩
    for name in ["improve", "test"]:
        cur = np.where(split == name)[0]
        if len(cur) > SIZES[name]:
            split[rng.choice(cur, len(cur) - SIZES[name], replace=False)] = "pool"
        elif len(cur) < SIZES[name]:
            pool = np.where(split == "pool")[0]
            split[rng.choice(pool, SIZES[name] - len(cur), replace=False)] = name
    d["split"] = split

    OUT.mkdir(parents=True, exist_ok=True)
    for name in ["improve", "test", "pool"]:
        part = d[d.split == name]
        with open(OUT / f"carfree_{name}.jsonl", "w", encoding="utf-8") as f:
            for r in part.itertuples():
                f.write(json.dumps({"id": r.essay_id, "score": int(r.score), "text": r.full_text,
                                    "words": int(r.words)}, ensure_ascii=False) + "\n")
        dist = part.score.value_counts().sort_index().to_dict()
        print(f"{name:<8} {len(part):>5}개  평균 {part.score.mean():.2f}  점수별 {dist}")
    sets = [set(d[d.split == n].essay_id) for n in ["improve", "test", "pool"]]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]), "겹침 발생!"
    print("겹침 없음 확인")


if __name__ == "__main__":
    main()
