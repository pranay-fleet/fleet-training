"""Reward computation logic for RLOO training."""

from typing import List, Optional, Tuple
import torch

from utils.tool_calls import parse_tool_call_str


def compute_rewards_from_generations(
    generations: List[str],
    correct_indices: torch.Tensor,
    retry_on_bad_json: int = 1,
) -> Tuple[List[float], List[Optional[int]]]:
    """
    Convert generated tool-call strings into rewards using environment's verifier.
    We lightly enforce JSON format by a small number of retries (caller can re-generate if desired).
    """
    preds = []
    rewards = []
    for s, correct_idx in zip(generations, correct_indices.tolist()):
        pred_i = parse_tool_call_str(s)
        reward = 1.0 if (pred_i is not None and int(pred_i) == int(correct_idx)) else 0.0
        preds.append(pred_i)
        rewards.append(reward)
    return rewards, preds

