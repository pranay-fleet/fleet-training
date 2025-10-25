#!/usr/bin/env python
# coding: utf-8
"""
End-to-end TRL RLOO training for Qwen2.5-VL-32B-Instruct on a single node with 8x H100.
- Inputs: (image, text) via AutoProcessor
- Output: constrained tool-call JSON, e.g. {"tool_name":"pick_image","arguments":{"i": 2}}
- Trainer: Custom RLOO (subclassed for multi-turn generation)
- Parallelism: Accelerate + DeepSpeed ZeRO-3 (memory sharding across 8 GPUs)
- Optional: LoRA for rapid iteration

Launch (example):
  accelerate launch --config_file accelerate_config.yaml training.py \
    --model_id Qwen/Qwen2.5-VL-32B-Instruct \
    --data_root ./toy_images \
    --train_steps 200 \
    --per_device_batch_size 1 \
    --num_rollouts 4 \
    --use_lora false

Make sure you have:
  pip install "transformers>=4.43" "trl>=0.8.6" accelerate deepspeed peft datasets pillow torchvision
  pip install flash-attn --no-build-isolation   # for H100 speed (optional, but recommended)
"""

import os
import argparse
from typing import List, Dict, Optional, Any, Union

import torch
from trl import RLOOConfig, RLOOTrainer
from contextlib import nullcontext

from environment import PickImageEnv
from dataset import RepeatableEpisodeDataset, VLMDataCollator
from model_utils import load_qwen25_vl, generate_tool_call
from reward import compute_rewards_from_generations


class MultiTurnRLOOTrainer(RLOOTrainer):
    """
    Custom RLOO Trainer that overrides generation to support multi-turn dialogues
    or custom generation logic for vision-language models.
    """
    
    def __init__(self, *args, processor=None, max_new_tokens=64, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_processor = processor  # Store the VL processor separately
        self.max_new_tokens = max_new_tokens
    
    def _generate_single_turn(self, prompts: list[str], images: Optional[list]):
        """
        Override the default generation to use custom multi-turn or VL-specific generation.
        This method is called by the parent class's _generate() method.
        """
        device = self.accelerator.device
        
        # Prepare the batch for custom generation
        # Build batch dict similar to what your collator creates
        batch_dict = {
            "prompt": prompts,
            "images": images if images is not None else [None] * len(prompts),
        }
        
        # Use your custom generate_tool_call function
        # Note: We need to reconstruct a batch format that your function expects
        batch_for_generation = {}
        
        # Process with your custom processor to get the right format
        if self.custom_processor is not None and images is not None:
            # Flatten images if they're nested
            flat_images = []
            for img_list in images:
                if img_list is not None:
                    flat_images.append(img_list)
                else:
                    flat_images.append([])
            
            # Use custom processor to prepare inputs
            processed = self.custom_processor(
                text=prompts,
                images=flat_images,
                padding=True,
                truncation=True,
                max_length=self.max_prompt_length,
                return_tensors="pt",
            )
            batch_for_generation = processed
        else:
            # Fallback to tokenizer only
            processed = self.processing_class(
                text=prompts,
                padding=True,
                truncation=True,
                max_length=self.max_prompt_length,
                return_tensors="pt",
            )
            batch_for_generation = processed
        
        # Move to device
        for key in batch_for_generation:
            if torch.is_tensor(batch_for_generation[key]):
                batch_for_generation[key] = batch_for_generation[key].to(device)
        
        # Use custom generation function
        from trl.models import unwrap_model_for_generation
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        
        with (
            unwrap_model_for_generation(
                self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
            ) as unwrapped_model,
            torch.no_grad(),
            FSDP.summon_full_params(self.model_wrapped, recurse=False) if self.is_fsdp_enabled else nullcontext(),
        ):
            # Call your custom generation
            generated_texts = generate_tool_call(
                model=unwrapped_model,
                processor=self.custom_processor if self.custom_processor is not None else self.processing_class,
                tokenizer=self.processing_class,
                batch=batch_for_generation,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            )
        
        # Convert generated texts back to token IDs
        # We need to return prompt_ids and completion_ids separately
        prompt_ids = []
        completion_ids = []
        
        for i, (prompt_text, generated_text) in enumerate(zip(prompts, generated_texts)):
            # Get prompt tokens
            prompt_tokens = self.processing_class.encode(prompt_text, add_special_tokens=False)
            prompt_ids.append(prompt_tokens)
            
            # Extract completion (remove prompt from generated text if present)
            # Your generate_tool_call might return full text or just completion
            completion_text = generated_text
            if generated_text.startswith(prompt_text):
                completion_text = generated_text[len(prompt_text):]
            
            completion_tokens = self.processing_class.encode(completion_text, add_special_tokens=False)
            completion_ids.append(completion_tokens)
        
        # forward_kwargs should contain any additional model inputs (e.g., pixel_values)
        # For reward calculation, we don't need pixel values, so return empty dict
        forward_kwargs = {}
        
        return prompt_ids, completion_ids, forward_kwargs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--data_root", type=str, default="./toy_images", required=False, help="Folder of candidate images (png/jpg).")
    parser.add_argument("--candidates_per_episode", type=int, default=4)
    parser.add_argument("--fixed_correct_index", type=int, default=0, help="Fixed index that is always correct (for toy example).")
    parser.add_argument("--train_steps", type=int, default=200)
    parser.add_argument("--per_device_batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--use_lora", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--num_rollouts", type=int, default=4, help="K rollouts per prompt for RLOO.")
    parser.add_argument("--max_length", type=int, default=1536)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--save_dir", type=str, default="./checkpoints_rloo_qwen25_vl")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    dtype = torch.bfloat16 if args.bf16 else torch.float16

    # 1) Load data
    exts = (".png", ".jpg", ".jpeg", ".webp")
    image_paths = [os.path.join(args.data_root, f) for f in os.listdir(args.data_root) if f.lower().endswith(exts)]
    assert len(image_paths) >= args.candidates_per_episode, "Not enough images in data_root."

    env = PickImageEnv(
        image_paths, 
        candidates_per_episode=args.candidates_per_episode,
        fixed_correct_index=args.fixed_correct_index
    )

    # 2) Load model/processor/tokenizer
    model, processor, tokenizer = load_qwen25_vl(
        model_id=args.model_id,
        dtype=dtype,
        use_lora=args.use_lora,
    )

    # 3) Build dataset/collator
    dataset = RepeatableEpisodeDataset(env, processor, max_length=args.max_length)
    collator = VLMDataCollator(processor=processor, tokenizer=tokenizer, max_length=args.max_length)

    # 4) RLOO config
    rloo_cfg = RLOOConfig(
        num_generations=args.num_rollouts,        # K generations per prompt (formerly num_rollouts)
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        per_device_train_batch_size=args.per_device_batch_size,
        logging_steps=10,
        save_steps=200,
        max_steps=args.train_steps,             # Total number of training steps
        output_dir=args.save_dir,
        bf16=args.bf16,
        remove_unused_columns=False,          # Important for custom batch dicts
        max_completion_length=args.max_new_tokens,  # Maximum length of generated completion
        temperature=0.8,                        # Temperature for sampling
        top_p=0.9,                              # Top-p for sampling
        # deepspeed will be picked up from env/config if present (accelerate/DeepSpeed)
    )

    # 5) Define reward function for the trainer
    # The new RLOOTrainer API expects a reward function that takes completions and returns rewards
    def reward_func(completions: List[str], **kwargs):
        """
        Reward function that takes completions and additional batch data.
        Returns a list of floats representing rewards for each completion.
        """
        # Extract correct_index from kwargs (passed from batch)
        correct_indices = kwargs.get("correct_index", None)
        if correct_indices is None:
            raise ValueError("correct_index not found in batch")
        
        rewards, _ = compute_rewards_from_generations(completions, correct_indices)
        return rewards

    # 6) Trainer - using custom subclass with multi-turn generation support
    trainer = MultiTurnRLOOTrainer(
        model=model,
        args=rloo_cfg,
        reward_funcs=reward_func,              # Single reward function
        train_dataset=dataset,
        processing_class=tokenizer,             # Use tokenizer as processing class
        processor=processor,                    # Pass the VL processor for custom generation
        max_new_tokens=args.max_new_tokens,     # Pass max_new_tokens for generation
    )

    # 7) Train
    trainer.train()

    # 8) Save
    trainer.save_model(args.save_dir)
    if hasattr(processor, "save_pretrained"):
        processor.save_pretrained(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)
    print(f"Training done. Saved to {args.save_dir}")


if __name__ == "__main__":
    main()
