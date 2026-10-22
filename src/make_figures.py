"""
README용 그림 2개 (라이트/다크 각각) → docs/
  1) improvement_curve: 버전별 QWK (개선용 200개)
  2) final_results: 확인용 200개 최종 비교

실행: python src/make_figures.py
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager

matplotlib.use("Agg")
ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
LOOP = ROOT / "runs" / "loop"

for f in font_manager.findSystemFonts():
    if "NotoSansCJK" in f and ("Regular" in f or "Medium" in f):
        font_manager.fontManager.addfont(f)
plt.rcParams["font.family"] = "Noto Sans CJK KR"
plt.rcParams["axes.unicode_minus"] = False

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#9a988f", grid="#e6e4dd", series="#2a78d6"),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#7d7b73", grid="#33322f", series="#3987e5"),
}


def base(ax, t):
    ax.set_facecolor(t["surface"])
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(t["grid"])
    ax.tick_params(colors=t["ink2"], labelsize=10, length=0)
    ax.yaxis.grid(True, color=t["grid"], linewidth=1)
    ax.set_axisbelow(True)


def curve(t, mode):
    stats = {v: json.load(open(LOOP / f"v{v}_stats.json")) for v in range(8)}
    q = {v: stats[v]["qwk"] for v in stats}
    kept, dropped = [0, 1, 2, 3, 4], [5, 6, 7]
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=160)
    fig.patch.set_facecolor(t["surface"])
    base(ax, t)
    ax.plot(kept, [q[v] for v in kept], color=t["series"], linewidth=2, zorder=3)
    ax.scatter(kept, [q[v] for v in kept], s=60, color=t["series"], edgecolor=t["surface"], linewidth=2, zorder=4)
    for v in dropped:  # v4에서 고쳤지만 v4를 넘지 못해 버린 시도
        ax.plot([4, v], [q[4], q[v]], color=t["muted"], linewidth=1, linestyle=(0, (2, 2)), zorder=2)
    ax.scatter(dropped, [q[v] for v in dropped], s=60, facecolor=t["surface"], edgecolor=t["muted"], linewidth=2, zorder=4)
    for v in [0, 4]:
        ax.annotate(f"{q[v]:.3f}", (v, q[v]), textcoords="offset points", xytext=(0, 10 if v else -16),
                    ha="center", color=t["ink"], fontsize=10, fontweight="bold")
    ax.text(6, 0.637, "v4를 넘지 못해 버린 버전", color=t["ink2"], fontsize=9, ha="center")
    ax.set_xticks(range(8), [f"v{v}" for v in range(8)])
    ax.set_ylim(0.45, 0.75)
    ax.set_xlim(-0.4, 7.6)
    ax.set_ylabel("QWK · 기준 개선용 200개", color=t["ink2"], fontsize=10)
    ax.set_title("버전별 사람 채점과의 일치도", color=t["ink"], fontsize=13, loc="left", pad=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(DOCS / f"improvement_curve_{mode}.png", facecolor=t["surface"])
    plt.close(fig)


def final(t, mode):
    rows = [("공식 루브릭", 0.392, (0.28, 0.49)), ("길이 기준선", 0.634, None),
            ("표면특징 모델", 0.656, (0.56, 0.73)), ("자동 개선", 0.697, (0.62, 0.76))]
    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=160)
    fig.patch.set_facecolor(t["surface"])
    base(ax, t)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=t["grid"], linewidth=1)
    ax.spines["bottom"].set_visible(False)
    y = range(len(rows))
    for i, (name, v, ci) in enumerate(rows):
        col = t["series"] if name == "자동 개선" else t["muted"]
        ax.barh(i, v, height=0.5, color=col, zorder=3)
        if ci:
            ax.plot(ci, [i, i], color=t["ink2"], linewidth=1.2, zorder=4)
        ax.text(0.80, i, f"{v:.3f}", va="center", ha="left", color=t["ink"], fontsize=10,
                fontweight="bold" if name == "자동 개선" else "normal")  # 값은 오른쪽 한 열에 정렬
    ax.set_yticks(list(y), [r[0] for r in rows], color=t["ink"])
    ax.set_xlim(0, 0.86)
    ax.set_xlabel("QWK · 최종 평가용 200개 · 가는 선은 95% 구간", color=t["ink2"], fontsize=10)
    ax.set_title("처음 보는 글에서의 최종 성능", color=t["ink"], fontsize=13, loc="left", pad=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(DOCS / f"final_results_{mode}.png", facecolor=t["surface"])
    plt.close(fig)


if __name__ == "__main__":
    DOCS.mkdir(exist_ok=True)
    for mode, t in THEMES.items():
        curve(t, mode)
        final(t, mode)
    print("저장:", sorted(p.name for p in DOCS.glob("*.png")))
