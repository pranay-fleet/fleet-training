"""Dataset and data collator for RLOO training."""

from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import PreTrainedTokenizerBase

from environment import PickImageEnv


class RepeatableEpisodeDataset(Dataset):
    """
    Provides 'episodes' that the trainer will roll out multiple times (RLOO).
    We re-sample an episode each __getitem__ to make it effectively infinite.
    """
    def __init__(self, env: PickImageEnv, processor: Any, max_length: int = 1024):
        self.env = env
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        # Endless-ish; trainer controls total steps. You can make this larger if you want epoch-like semantics.
        return 10**9

    def __getitem__(self, idx):
        ep = self.env.sample_episode()
        images = [Image.open(p).convert("RGB") for p in ep["candidate_paths"]]

        # Many VL processors accept lists of images + text
        # Qwen2.5-VL processors typically support something like:
        # processor(text=..., images=[...], return_tensors="pt")
        # We'll just stash PILs here; real processing happens in collator to keep batchable.
        return {
            "prompt": ep["prompt"],
            "images": images,
            "correct_index": ep["correct_index"],
        }


@dataclass
class VLMDataCollator:
    processor: Any
    tokenizer: PreTrainedTokenizerBase
    max_length: int = 1536

    def __call__(self, features: List[Dict]) -> Dict[str, Any]:
        # Batch prompts and images
        prompts = [f["prompt"] for f in features]
        batch_images = [f["images"] for f in features]  # list of list[PIL]
        correct_indices = torch.tensor([f["correct_index"] for f in features], dtype=torch.long)

        # Flatten images per sample into processor input
        # Qwen2.5-VL processors accept a list of images per sample. We pass them per-item via "images=batch_images".
        processed = self.processor(
            text=prompts,
            images=batch_images,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        # processor returns pixel_values and input_ids/attention_mask if it wraps a tokenizer; some Qwen processors return image inputs + tokenized text.
        # Ensure we have input_ids; if not, tokenize text separately:
        if "input_ids" not in processed:
            tok = self.tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            processed.update(tok)

        processed["labels"] = None  # RL setting: no supervised labels
        processed["correct_index"] = correct_indices
        return processed

