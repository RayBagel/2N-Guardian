"""
Reward manager for RL training.

Design goals:
1) Reuse project's production reward pipeline (not toy keyword scoring)
2) Deterministic, auditable outputs for each sample
3) Easy integration with GRPO rollout loop
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is importable when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_engine.filter_data import RewardScorer

DEFAULT_SYSTEM_PROMPT = (
    "你是劳动争议法律分析助手。"
    "请按IRAC结构输出，并基于事实、法条、结论给出完整推理。"
)


@dataclass
class RewardResult:
    total_reward: float
    score_fact: float
    score_law: float
    score_process: float
    process_ring1_pass: float
    process_ring2_pass: float
    raw_components: Dict[str, float]
    explanation: str


class RewardManager:
    """
    Adapter layer around project's core reward stack.

    Input:
    - sample: one frozen dataset sample (contains facts/outcome/...)
    - completion: model generated answer text

    Output:
    - RewardResult with PRM sub-scores and explainable diagnostics
    """

    def __init__(self, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> None:
        self.system_prompt = system_prompt
        self._scorer = RewardScorer()

    @staticmethod
    def _clip01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    @staticmethod
    def _safe_float(obj: Dict[str, Any], key: str, default: float = 0.0) -> float:
        try:
            return float(obj.get(key, default))
        except Exception:
            return float(default)

    def _build_row(self, sample: Dict[str, Any], completion: str) -> Dict[str, Any]:
        """
        Convert RL sample + model output into the row format expected by RewardScorer.
        """
        facts = str(sample.get("facts") or sample.get("input") or "").strip()
        outcome = str(sample.get("outcome") or sample.get("label") or "unknown").strip()
        seed_id = str(sample.get("seed_id") or sample.get("id") or "unknown_seed")

        # Keep deterministic case_id for easier traceability in logs.
        trace_key = hashlib.sha256((seed_id + facts).encode("utf-8")).hexdigest()[:12]
        case_id = f"rl_{seed_id}_{trace_key}"

        return {
            "case_id": case_id,
            "system": self.system_prompt,
            "user": facts,
            "assistant": str(completion or "").strip(),
            "outcome": outcome,
        }

    def score(self, sample: Dict[str, Any], completion: str) -> RewardResult:
        row = self._build_row(sample, completion)
        total_reward, breakdown = self._scorer.score_row(row)
        components = dict(breakdown.get("reward_components", {}) or {})

        return RewardResult(
            total_reward=self._clip01(total_reward),
            score_fact=self._clip01(self._safe_float(components, "score_fact", 0.0)),
            score_law=self._clip01(self._safe_float(components, "score_law", 0.0)),
            score_process=self._clip01(self._safe_float(components, "score_process", 0.0)),
            process_ring1_pass=self._clip01(self._safe_float(components, "process_ring1_pass", 0.0)),
            process_ring2_pass=self._clip01(self._safe_float(components, "process_ring2_pass", 0.0)),
            raw_components=components,
            explanation=str(breakdown.get("explanation", "")),
        )

    def score_batch(self, samples: List[Dict[str, Any]], completions: List[str]) -> List[RewardResult]:
        if len(samples) != len(completions):
            raise ValueError("samples and completions length mismatch")
        return [self.score(s, c) for s, c in zip(samples, completions)]


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def self_test(dataset_path: str = "v2_mvp/rl/data/eval_set.jsonl", limit: int = 3) -> None:
    rows = _load_jsonl(dataset_path)[:limit]
    manager = RewardManager()

    print(f"self_test dataset={dataset_path} samples={len(rows)}")
    for i, sample in enumerate(rows, 1):
        # For reproducibility in this smoke test, use reference_output as completion.
        completion = str(sample.get("reference_output") or sample.get("reasoning") or "")
        result = manager.score(sample, completion)
        print(
            f"[{i}] id={sample.get('id')} outcome={sample.get('outcome')} "
            f"reward={result.total_reward:.4f} "
            f"fact={result.score_fact:.4f} law={result.score_law:.4f} process={result.score_process:.4f} "
            f"rings=({result.process_ring1_pass:.0f},{result.process_ring2_pass:.0f})"
        )


if __name__ == "__main__":
    self_test()
