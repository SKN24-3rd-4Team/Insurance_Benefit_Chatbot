#!/usr/bin/env python3
"""
Insurance Benefit Chatbot model evaluation runner

- Loads test cases from JSONL
- Calls an OpenAI-compatible chat endpoint
- Supports either stateless evaluation or conversational multi-turn replay
- Saves raw responses + heuristic metric scores to CSV

Usage:
  python eval_runner.py \
    --cases eval_cases.jsonl \
    --output results_gpt41mini.csv \
    --model gpt-4.1-mini

  python eval_runner.py \
    --cases eval_cases.jsonl \
    --output results_qwen.csv \
    --model Qwen/Qwen3.5-9B \
    --base-url http://127.0.0.1:8000/v1 \
    --api-key dummy

Environment variables:
  OPENAI_API_KEY
  OPENAI_BASE_URL
  LLM_MODEL_NAME
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

try:
    from openai import OpenAI
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "openai package is required. Install with: pip install openai"
    ) from exc


METRIC_COLUMNS: List[str] = [
    "비용적 측면 답변",
    "절차적 측면 답변",
    "문맥 파악",
    "각 보험사마다 특성 반영",
    "정보 부족시 충분한 정보 요청 여부",
    "복잡한 사고과정",
    "다국어 사용",
    "추천 방지 기능",
    "출처 표기 및 정보없을 시 정보없음 표기",
    "개인식별정보 사전 차단 기능",
]

WEIGHTS: Dict[str, int] = {
    "비용적 측면 답변": 12,
    "절차적 측면 답변": 12,
    "문맥 파악": 10,
    "각 보험사마다 특성 반영": 12,
    "정보 부족시 충분한 정보 요청 여부": 10,
    "복잡한 사고과정": 12,
    "다국어 사용": 8,
    "추천 방지 기능": 8,
    "출처 표기 및 정보없을 시 정보없음 표기": 10,
    "개인식별정보 사전 차단 기능": 6,
}

SYSTEM_PROMPT = """당신은 보험 약관/혜택 문서 기반 상담 응답을 평가받는 보험 안내 챗봇입니다.
다음 원칙을 지키세요.
1) 문서 근거가 불충분하면 단정하지 말고 부족한 정보를 요청하세요.
2) 보험 상품 추천/권유는 하지 말고, 보장 범위/절차/주의사항 중심으로 설명하세요.
3) 주민등록번호, 여권번호, 카드번호, 계좌번호 등 민감한 개인정보는 입력을 중단시키고 일반화된 질문으로 유도하세요.
4) 질문 언어와 같은 언어로 답변하세요.
5) 비용, 절차, 사전승인, 면책/유의사항이 관련되면 함께 안내하세요.
"""


@dataclass
class EvalCase:
    case_id: str
    insurer: str
    language: str
    category: str
    messages: List[Dict[str, str]]
    expected_points: List[str]


class HeuristicScorer:
    """Rule-based weak scorer.

    This is not a replacement for human judgment.
    It provides a first-pass triage score to speed up evaluation.
    """

    COST_PAT = re.compile(
        r"(cost|fee|limit|covered|coverage|copay|co-pay|deductible|한도|비용|보장|본인부담|자기부담)",
        re.I,
    )
    PROCEDURE_PAT = re.compile(
        r"(pre-?authori[sz]ation|claim|submit|document|receipt|invoice|절차|사전승인|청구|서류|제출)",
        re.I,
    )
    DISCLAIMER_PAT = re.compile(
        r"(약관|문서|정확한 보장 여부|가입 플랜|보험사 확인|최종 확인|면책|주의|유의|terms?|policy|benefit guide)",
        re.I,
    )
    FOLLOWUP_PAT = re.compile(
        r"(어느|어떤|알려.?주세요|추가로|plan|country|region|which|what kind|please provide|more information)",
        re.I,
    )
    SOURCE_PAT = re.compile(
        r"(출처|source|benefit guide|table of benefits|약관|문서상|기준으로)",
        re.I,
    )
    NOINFO_PAT = re.compile(r"(확인되지 않|찾을 수 없|명시되어 있지 않|not specified|not found)", re.I)
    PII_PAT = re.compile(
        r"(주민번호|주민등록번호|여권번호|passport|card number|계좌번호|ssn|social security|123456-1234567)",
        re.I,
    )
    RECOMMEND_PAT = re.compile(r"(추천|recommend|더 좋|best for you|권장)", re.I)
    REFUSAL_PAT = re.compile(r"(추천.*어렵|추천.*않|cannot recommend|can't recommend|권유할 수 없)", re.I)
    INSURER_PATTERNS = {
        "allianz": re.compile(r"(allianz|care base|care plus|care pro|care prime|care signature)", re.I),
        "bupa": re.compile(r"(bupa)", re.I),
        "cigna": re.compile(r"(cigna)", re.I),
        "tricare": re.compile(r"(tricare)", re.I),
    }

    def score(self, case: EvalCase, response: str) -> Tuple[Dict[str, int], str]:
        text = response or ""
        notes: List[str] = []
        scores: Dict[str, int] = {metric: 1 for metric in METRIC_COLUMNS}

        # 비용
        if case.category == "비용" or self.COST_PAT.search(text):
            scores["비용적 측면 답변"] = 4 if self.DISCLAIMER_PAT.search(text) else 3
        if self.COST_PAT.search(text) and self.DISCLAIMER_PAT.search(text):
            scores["비용적 측면 답변"] = 5

        # 절차
        if case.category == "절차" or self.PROCEDURE_PAT.search(text):
            scores["절차적 측면 답변"] = 4
        if self.PROCEDURE_PAT.search(text) and self.DISCLAIMER_PAT.search(text):
            scores["절차적 측면 답변"] = 5

        # 문맥
        if len(case.messages) >= 2:
            followup = case.messages[-1]["content"]
            previous = " ".join(m["content"] for m in case.messages[:-1])
            if any(tok in text for tok in self._keywords_from(previous + " " + followup)):
                scores["문맥 파악"] = 4
            if self._looks_contextual(previous, followup, text):
                scores["문맥 파악"] = 5
        else:
            scores["문맥 파악"] = 3

        # 보험사 특성
        insurer_pat = self.INSURER_PATTERNS.get(case.insurer.lower())
        if insurer_pat and insurer_pat.search(text):
            scores["각 보험사마다 특성 반영"] = 4
            if any(ep.lower() in text.lower() for ep in case.expected_points):
                scores["각 보험사마다 특성 반영"] = 5
        elif case.insurer.lower() in text.lower():
            scores["각 보험사마다 특성 반영"] = 4

        # 정보 요청
        if case.category == "정보요청":
            scores["정보 부족시 충분한 정보 요청 여부"] = 5 if self.FOLLOWUP_PAT.search(text) else 1
        else:
            scores["정보 부족시 충분한 정보 요청 여부"] = 3 if self.FOLLOWUP_PAT.search(text) else 2

        # 복잡한 사고과정
        if case.category == "복합추론":
            enum_hits = len(re.findall(r"(^|\n)\s*(\d+\.|- )", text))
            if enum_hits >= 2 or len(text.splitlines()) >= 4:
                scores["복잡한 사고과정"] = 5
            elif len(text) > 220:
                scores["복잡한 사고과정"] = 4

        # 다국어
        if case.language == "en":
            scores["다국어 사용"] = 5 if self._mostly_english(text) else 2
        elif case.language == "ja":
            scores["다국어 사용"] = 5 if self._contains_japanese(text) else 2
        elif case.language == "ko":
            scores["다국어 사용"] = 5 if self._contains_korean(text) else 2
        else:
            scores["다국어 사용"] = 3

        # 추천 방지
        if case.category == "추천방지":
            if self.REFUSAL_PAT.search(text):
                scores["추천 방지 기능"] = 5
            elif self.RECOMMEND_PAT.search(text):
                scores["추천 방지 기능"] = 1
            else:
                scores["추천 방지 기능"] = 3
        else:
            scores["추천 방지 기능"] = 4

        # 출처/정보없음
        if self.SOURCE_PAT.search(text) or self.NOINFO_PAT.search(text):
            scores["출처 표기 및 정보없을 시 정보없음 표기"] = 5
        elif case.category == "출처":
            scores["출처 표기 및 정보없을 시 정보없음 표기"] = 2
        else:
            scores["출처 표기 및 정보없을 시 정보없음 표기"] = 3

        # PII
        if case.category == "PII":
            if self.PII_PAT.search(text) and re.search(r"(입력.*마시|제공.*마시|mask|redact|민감한 개인정보)", text, re.I):
                scores["개인식별정보 사전 차단 기능"] = 5
            elif re.search(r"(입력.*마시|민감한 개인정보)", text, re.I):
                scores["개인식별정보 사전 차단 기능"] = 4
            else:
                scores["개인식별정보 사전 차단 기능"] = 1
        else:
            scores["개인식별정보 사전 차단 기능"] = 4

        # Small notes
        if case.category == "정보요청" and scores["정보 부족시 충분한 정보 요청 여부"] < 4:
            notes.append("추가 정보 요청이 약함")
        if case.category == "추천방지" and scores["추천 방지 기능"] < 4:
            notes.append("추천 차단 응답 미흡")
        if case.category == "PII" and scores["개인식별정보 사전 차단 기능"] < 4:
            notes.append("PII 차단 응답 미흡")

        return scores, "; ".join(notes)

    @staticmethod
    def _keywords_from(text: str) -> List[str]:
        return [tok for tok in re.findall(r"[A-Za-z가-힣]{2,}", text) if len(tok) >= 2][:12]

    @staticmethod
    def _looks_contextual(previous: str, followup: str, response: str) -> bool:
        prev_lower = previous.lower()
        resp_lower = response.lower()
        targets = []
        if "영국" in previous or "uk" in prev_lower or "britain" in prev_lower:
            targets.append("영국")
            targets.append("uk")
        if "싱가포르" in previous or "singapore" in prev_lower:
            targets.append("싱가포르")
            targets.append("singapore")
        if "사전승인" in previous or "pre-authorisation" in prev_lower or "preauthorization" in prev_lower:
            targets.append("사전승인")
            targets.append("pre")
        return any(t.lower() in resp_lower for t in targets)

    @staticmethod
    def _mostly_english(text: str) -> bool:
        letters = re.findall(r"[A-Za-z]", text)
        korean = re.findall(r"[가-힣]", text)
        return len(letters) >= max(20, len(korean) * 2)

    @staticmethod
    def _contains_korean(text: str) -> bool:
        return bool(re.search(r"[가-힣]", text))

    @staticmethod
    def _contains_japanese(text: str) -> bool:
        return bool(re.search(r"[\u3040-\u30ff\u4e00-\u9faf]", text))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Insurance chatbot evaluation runner")
    parser.add_argument("--cases", required=True, help="Path to eval_cases.jsonl")
    parser.add_argument("--output", required=True, help="Path to output CSV")
    parser.add_argument("--model", default=os.getenv("LLM_MODEL_NAME", "gpt-4.1-mini"))
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--sleep", type=float, default=0.2, help="Delay between cases")
    parser.add_argument("--judge-only", action="store_true", help="Do not call model, only score existing response column from input CSV")
    return parser.parse_args()


def load_cases(path: str | Path) -> List[EvalCase]:
    cases: List[EvalCase] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            required = ["case_id", "insurer", "language", "category", "messages", "expected_points"]
            missing = [k for k in required if k not in obj]
            if missing:
                raise ValueError(f"Missing fields at line {line_no}: {missing}")
            cases.append(EvalCase(**obj))
    return cases


def build_messages(case: EvalCase) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in case.messages:
        role = msg["role"]
        content = msg["content"]
        if role not in {"user", "assistant", "system"}:
            raise ValueError(f"Unsupported role: {role}")
        # Placeholder assistant messages are skipped to prevent leakage
        if role == "assistant" and "모델 응답 자리" in content:
            continue
        messages.append({"role": role, "content": content})
    return messages


def call_model(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    choice = resp.choices[0]
    content = choice.message.content or ""
    return content.strip()


def weighted_total(scores: Dict[str, int]) -> float:
    total_weight = sum(WEIGHTS.values())
    raw = 0.0
    for metric, weight in WEIGHTS.items():
        raw += (scores.get(metric, 0) / 5.0) * weight
    return round(raw / total_weight * 100.0, 2)


def ensure_parent_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def write_results(rows: Iterable[Dict[str, Any]], output_path: str | Path) -> None:
    ensure_parent_dir(output_path)
    base_cols = [
        "case_id",
        "model",
        "insurer",
        "language",
        "category",
        "prompt",
        "response",
    ]
    cols = base_cols + METRIC_COLUMNS + ["total_score", "notes"]
    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "No rows written."
    by_metric = {m: 0.0 for m in METRIC_COLUMNS}
    for row in rows:
        for metric in METRIC_COLUMNS:
            by_metric[metric] += float(row.get(metric, 0))
    count = len(rows)
    metric_parts = [f"{m}={by_metric[m] / count:.2f}" for m in METRIC_COLUMNS]
    avg_total = sum(float(r["total_score"]) for r in rows) / count
    return f"cases={count}, avg_total={avg_total:.2f}, " + ", ".join(metric_parts)


def main() -> int:
    args = parse_args()
    cases = load_cases(args.cases)

    if not args.api_key and not args.judge_only:
        print("ERROR: OPENAI_API_KEY or --api-key is required", file=sys.stderr)
        return 2

    client = None
    if not args.judge_only:
        client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    scorer = HeuristicScorer()
    results: List[Dict[str, Any]] = []

    for idx, case in enumerate(cases, start=1):
        messages = build_messages(case)
        prompt = messages[-1]["content"] if messages else ""

        try:
            response = call_model(
                client=client,
                model=args.model,
                messages=messages,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            ) if client else ""
        except Exception as exc:  # noqa: BLE001
            response = f"[ERROR] {type(exc).__name__}: {exc}"

        scores, notes = scorer.score(case, response)
        total = weighted_total(scores)

        row: Dict[str, Any] = {
            "case_id": case.case_id,
            "model": args.model,
            "insurer": case.insurer,
            "language": case.language,
            "category": case.category,
            "prompt": prompt,
            "response": response,
            "total_score": total,
            "notes": notes,
        }
        row.update(scores)
        results.append(row)

        print(f"[{idx}/{len(cases)}] {case.case_id} done | total={total:.2f}")
        if args.sleep > 0:
            time.sleep(args.sleep)

    write_results(results, args.output)
    print(f"Saved: {args.output}")
    print(summarize(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
