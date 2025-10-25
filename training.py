#!/usr/bin/env python
# coding: utf-8
"""
End-to-end TRL RLOO training for Qwen2.5-VL-32B-Instruct on a single node with 8x H100.
- Inputs: (image, text) via AutoProcessor
- Output: constrained tool-call JSON, e.g. {"tool_name":"pick_image","arguments":{"i": 2}}
- Trainer: TRL RLOO (no value head)
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
from typing import List, Dict

import torch
from trl import RLOOConfig, RLOOTrainer

from environment import PickImageEnv
from dataset import RepeatableEpisodeDataset, VLMDataCollator
from model_utils import load_qwen25_vl, generate_tool_call
from reward import compute_rewards_from_generations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-VL-32B-Instruct")
    parser.add_argument("--data_root", type=str, required=True, help="Folder of candidate images (png/jpg).")
    parser.add_argument("--candidates_per_episode", type=int, default=4)
    parser.add_argument("--fixed_correct_index", type=int, default=0, help="Fixed index that is always correct (for toy example).")
    parser.add_argument("--train_steps", type=int, default=1000)
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
        num_rollouts=args.num_rollouts,        # K rollouts per prompt
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        per_device_train_batch_size=args.per_device_batch_size,
        logging_steps=10,
        save_steps=200,
        output_dir=args.save_dir,
        bf16=args.bf16,
        remove_unused_columns=False,          # Important for custom batch dicts
        # deepspeed will be picked up from env/config if present (accelerate/DeepSpeed)
    )

    # 5) Define generation & reward hooks for the trainer
    def generate_fn(model, batch):
        gens = generate_tool_call(
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            batch=batch,
            max_new_tokens=args.max_new_tokens,
            temperature=0.8,
            top_p=0.9,
        )
        return gens

    def reward_fn(samples: List[str], batch: Dict[str, torch.Tensor]) -> List[float]:
        rewards, _ = compute_rewards_from_generations(samples, batch["correct_index"])
        return rewards

    # 6) Trainer
    trainer = RLOOTrainer(
        config=rloo_cfg,
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        data_collator=collator,
        generate_fn=generate_fn,   # maps batch -> List[str] generations
        reward_fn=reward_fn,       # maps (generations, batch) -> List[float]
    )

    # 7) Train
    trainer.train(max_steps=args.train_steps)

    # 8) Save
    trainer.save_model(args.save_dir)
    if hasattr(processor, "save_pretrained"):
        processor.save_pretrained(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)
    print(f"Training done. Saved to {args.save_dir}")


if __name__ == "__main__":
    main()
