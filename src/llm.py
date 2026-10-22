"""
Claude API 호출 + 비용 기록 + 예산 정지 장치

- 모든 호출의 토큰 수와 금액을 runs/cost_log.csv 에 한 줄씩 기록
- 누적 금액이 BUDGET_USD 를 넘을 것 같으면 호출하기 전에 멈춤 (BudgetExceeded)
- system 프롬프트(기준+제시문)는 캐싱 → 두 번째 호출부터 그 부분이 1/10 가격
- 여러 스레드에서 동시에 불러도 안전 (기록·누적 금액을 잠금으로 보호)
- count_input_tokens(): 무료 토큰 계산 API
"""

import csv
import os
import threading
from datetime import datetime
from pathlib import Path

import anthropic
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
COST_LOG = ROOT / "runs" / "cost_log.csv"
BUDGET_USD = 12.0  # 전체 2만 원 이내 (CLAUDE.md 14번)

GRADER_MODEL = "claude-haiku-5-5"
REVISER_MODEL = "claude-sonnet-5-5"

# 1백만 토큰당 달러 (공식 가격표, 2026-10 확인)
# input = 캐시 안 된 입력, cache_write = 5분 캐시 저장, cache_read = 캐시 재사용, output = 출력
PRICES = {
    "claude-haiku-5-5": {"input": 0.10, "cache_write": 0.125, "cache_read": 0.01, "output": 0.50},
    "claude-sonnet-5-5": {"input": 2.00, "cache_write": 2.50, "cache_read": 0.10, "output": 10.00},
}
HAIKU_LONG_PROMPT = 100_000  # Haiku 5.5는 프롬프트가 이보다 길면 5배 가격

load_dotenv(ROOT / ".env")
_client = None
_lock = threading.Lock()
_spent = None      # 누적 금액 (처음 한 번 파일에서 읽고 이후 메모리에서 갱신)
_reserved = 0.0    # 지금 진행 중인 호출들의 최대 예상 금액 (동시 호출 대비)


class BudgetExceeded(Exception):
    pass


def client():
    global _client
    if _client is None:
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError(".env 에 ANTHROPIC_API_KEY 가 없습니다")
        _client = anthropic.Anthropic(max_retries=6)  # 속도 제한(429) 등은 자동 재시도
    return _client


def _read_spent() -> float:
    if not COST_LOG.exists():
        return 0.0
    with open(COST_LOG, encoding="utf-8") as f:
        return sum(float(r["cost_usd"]) for r in csv.DictReader(f))


def spent_so_far() -> float:
    global _spent
    with _lock:
        if _spent is None:
            _spent = _read_spent()
        return _spent


def _cost(model, usage) -> float:
    p = PRICES[model]
    return (usage.input_tokens * p["input"]
            + (usage.cache_creation_input_tokens or 0) * p["cache_write"]
            + (usage.cache_read_input_tokens or 0) * p["cache_read"]
            + usage.output_tokens * p["output"]) / 1_000_000


def _log(tag, model, usage, cost):
    """_lock 안에서 호출할 것."""
    COST_LOG.parent.mkdir(parents=True, exist_ok=True)
    new = not COST_LOG.exists()
    with open(COST_LOG, "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "tag", "model", "input", "cache_write", "cache_read", "output", "cost_usd"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), tag, model,
                    usage.input_tokens, usage.cache_creation_input_tokens or 0,
                    usage.cache_read_input_tokens or 0, usage.output_tokens, f"{cost:.6f}"])


def count_input_tokens(model, system, user) -> int:
    """무료 토큰 계산 API (비용 없음)."""
    r = client().messages.count_tokens(
        model=model, system=system, messages=[{"role": "user", "content": user}])
    return r.input_tokens


def call(model, system, user, *, max_tokens=2000, tag="", thinking=False) -> str:
    """
    API 1회 호출. 예산 초과 위험이 있으면 호출 전에 BudgetExceeded.

    - 5.5 모델은 temperature를 지정할 수 없음 (같은 글도 매번 조금 다르게 채점될 수 있음)
    - 5.5 모델은 thinking(내부 추론)이 기본으로 켜져 있고 그 토큰도 출력 요금이 붙음.
      채점은 프롬프트에서 '근거 먼저 쓰고 점수' 방식으로 추론하게 하므로 기본은 끔.
    """
    global _spent, _reserved
    p = PRICES[model]
    # 최악의 경우(캐시 없음 + 출력 최대)를 가정한 이번 호출 상한 (한국어 1글자 ≈ 최대 2토큰으로 넉넉히)
    worst = (len(system) + len(user)) * 2 * p["cache_write"] / 1e6 + max_tokens * p["output"] / 1e6
    spent_so_far()
    with _lock:
        if _spent + _reserved + worst > BUDGET_USD:
            raise BudgetExceeded(f"누적 ${_spent:.3f} + 진행 중 ${_reserved:.3f} + 이번 최대 ${worst:.3f} > 예산 ${BUDGET_USD}")
        _reserved += worst
    try:
        resp = client().messages.create(
            model=model,
            max_tokens=max_tokens,
            # 생각 끄기 값이 모델마다 다름: Haiku 5.5 = disabled, Sonnet 5.5 = between_tools
            thinking={"type": "adaptive"} if thinking else (
                {"type": "between_tools"} if model == REVISER_MODEL else {"type": "disabled"}),
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
        )
    finally:
        with _lock:
            _reserved -= worst

    u = resp.usage
    cost = _cost(model, u)
    with _lock:
        _spent += cost
        _log(tag, model, u, cost)
    total_in = u.input_tokens + (u.cache_creation_input_tokens or 0) + (u.cache_read_input_tokens or 0)
    if model == GRADER_MODEL and total_in > HAIKU_LONG_PROMPT:
        print(f"[경고] {tag}: 프롬프트 {total_in} 토큰 > 100k, 고가 구간 요금이 적용됐을 수 있음")
    if resp.stop_reason == "max_tokens":
        print(f"[경고] {tag}: 출력이 max_tokens({max_tokens})에서 잘림")
    return "".join(b.text for b in resp.content if b.type == "text")
