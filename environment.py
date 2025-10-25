"""Environment for the pick image task with fixed correct index."""

import random
from typing import Any, Dict, List, Optional


class PickImageEnv:
    """
    Environment where each episode:
      - You pass a prompt like: "Pick the image that contains a red triangle."
      - There are N candidate images (N small, e.g., 4). Exactly one is correct.
      - The model must output a tool call: {"tool_name":"pick_image","arguments":{"i": <int>}}
      - Reward = 1.0 if i == correct_index else 0.0

    For this toy example, we always use a FIXED correct index (default: 0),
    but maintain the infrastructure for easy extension.
    """
    def __init__(
        self, 
        image_paths: List[str], 
        candidates_per_episode: int = 4,
        fixed_correct_index: int = 0
    ):
        assert len(image_paths) >= candidates_per_episode, "Need enough images to form candidate sets."
        assert 0 <= fixed_correct_index < candidates_per_episode, "Fixed correct index must be within candidate range."
        self.image_paths = image_paths
        self.k = candidates_per_episode
        self.fixed_correct_index = fixed_correct_index

    def sample_episode(self) -> Dict[str, Any]:
        """
        Sample K images, but always use the fixed correct index.
        This makes the task deterministic for toy testing while maintaining infrastructure.
        """
        # Randomly choose K images
        candidates = random.sample(self.image_paths, self.k)
        
        # Use fixed correct index
        correct_idx = self.fixed_correct_index
        correct_path = candidates[correct_idx]

        # Construct a prompt
        # In a real setup, you'd have structured metadata + a verifier.
        prompt = (
            "You are a vision agent. Exactly one of the following K images matches the instruction.\n"
            "Tool: pick_image(i) where i ∈ {0, ..., K-1}.\n"
            "Respond ONLY with the JSON tool-call.\n"
            "Instruction: Select the image that best matches the description.\n"
            f"K = {self.k}\n"
            f"The correct image is always at index {self.fixed_correct_index}.\n"
            "Images are numbered in the order given."
        )

        # Pack episode
        return {
            "prompt": prompt,
            "candidate_paths": candidates,
            "correct_index": correct_idx,
        }

    def reward(self, predicted_index: Optional[int], correct_index: int) -> float:
        """Compute reward: 1.0 if prediction matches correct index, 0.0 otherwise."""
        if predicted_index is None:
            return 0.0
        return 1.0 if int(predicted_index) == int(correct_index) else 0.0

