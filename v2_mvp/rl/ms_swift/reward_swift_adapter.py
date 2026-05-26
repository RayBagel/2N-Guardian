"""
Reward adapter for TRL/GRPO-style training.

Purpose:
- expose a standard reward_fn(prompts, completions) -> List[float]
- reuse v2_mvp.rl.reward_manager.RewardManager as the single scoring backend
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v2_mvp.rl.reward_manager import RewardManager

_MANAGER = RewardManager()


def _extract_sample(meta: Any) -> Dict[str, Any]:
    """
    Normalize metadata payload to sample dict.
    Accepts dict or JSON string.
    """
    if isinstance(meta, dict):
        return meta
    if isinstance(meta, str):
        meta = meta.strip()
        if not meta:
            return {}
        try:
            obj = json.loads(meta)
            if isinstance(obj, dict):
                return obj
        except Exception:
            return {}
    return {}


def _extract_completion_text(obj: Any) -> str:
    """
    Normalize completion to text.
    Supports plain string or list/dict chat formats.
    """
    if isinstance(obj, str):
        return obj
    if isinstance(obj, list) and obj:
        last = obj[-1]
        if isinstance(last, dict):
            return str(last.get("content", ""))
    if isinstance(obj, dict):
        return str(obj.get("content", obj.get("text", "")))
    return str(obj or "")


def score_one(completion: str, sample: Dict[str, Any]) -> float:
    result = _MANAGER.score(sample, completion)
    return float(result.total_reward)


def score_batch(completions: List[str], samples: List[Dict[str, Any]]) -> List[float]:
    if len(completions) != len(samples):
        raise ValueError("completions and samples length mismatch")
    return [score_one(c, s) for c, s in zip(completions, samples)]


# ---- TRL-compatible entrypoint ----

def reward_fn(prompts: Sequence[str], completions: Sequence[Any], **kwargs: Any) -> List[float]:
    """
    TRL-style entrypoint: (prompts, completions, **kwargs) -> rewards
    """
    metas = kwargs.get("metas") or kwargs.get("samples") or []
    samples: List[Dict[str, Any]] = []
    for i, prompt in enumerate(prompts):
        base = {"facts": str(prompt or "")}
        if i < len(metas):
            base.update(_extract_sample(metas[i]))
        samples.append(base)
    texts = [_extract_completion_text(c) for c in completions]
    return score_batch(texts, samples)


# ---- Backward-compatible aliases (optional) ----

def get_reward(prompts: Sequence[str], completions: Sequence[Any], **kwargs: Any) -> List[float]:
    """Alias for frameworks expecting get_reward."""
    return reward_fn(prompts, completions, **kwargs)


def compute_rewards(prompts: Sequence[str], completions: Sequence[Any], **kwargs: Any) -> List[float]:
    """Alias for frameworks expecting compute_rewards."""
    return reward_fn(prompts, completions, **kwargs)


def reward_fn_legacy(completions: List[str], metas: List[Any]) -> List[float]:
    """
    Legacy entrypoint: (completions, metas) -> rewards
    """
    samples = [_extract_sample(m) for m in metas]
    return score_batch(completions, samples)


def self_test(dataset_path: str = "v2_mvp/rl/data/eval_set.jsonl", limit: int = 3) -> None:
    rows = [json.loads(l) for l in Path(dataset_path).read_text(encoding="utf-8").splitlines() if l.strip()][:limit]
    prompts = [str(r.get("facts") or r.get("input") or "") for r in rows]
    completions = [str(r.get("reference_output") or r.get("reasoning") or "") for r in rows]
    rewards = reward_fn(prompts, completions, metas=rows)
    print(f"self_test rows={len(rows)} rewards={[round(x,4) for x in rewards]}")


if __name__ == "__main__":
    self_test()
