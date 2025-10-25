# Fleet - RLOO Training for Vision-Language Models

This repository contains a modular implementation of RLOO (Reinforcement Learning from Online Optimization) training for Qwen2.5-VL models.

## Project Structure

The codebase has been organized into focused modules:

```
fleet/
├── training.py              # Main training script and entry point
├── environment.py           # PickImageEnv - task environment with fixed correct index
├── dataset.py              # RepeatableEpisodeDataset and VLMDataCollator
├── model_utils.py          # Model loading and generation utilities
├── reward.py               # Reward computation logic
├── utils/
│   ├── __init__.py
│   └── tool_calls.py       # Tool-call formatting and parsing
├── accelerate_config.yaml  # Accelerate/DeepSpeed configuration
├── requirements.txt        # Python dependencies
└── toy_images/            # Sample images for training
```

## Key Features

- **Modular Architecture**: Clean separation of concerns with dedicated modules for environment, dataset, model utilities, and reward computation
- **Fixed Correct Index**: For toy examples, the environment now uses a fixed correct index (default: 0) instead of random selection, while maintaining good infrastructure for easy extension
- **RLOO Training**: Implements TRL's RLOO trainer for reinforcement learning
- **Vision-Language Support**: Full support for Qwen2.5-VL models with image+text inputs
- **DeepSpeed ZeRO-3**: Memory-efficient training across multiple GPUs
- **Optional LoRA**: Fast iteration with low-rank adaptation

## Usage

### Basic Training

```bash
accelerate launch --config_file accelerate_config.yaml training.py \
  --model_id Qwen/Qwen2.5-VL-32B-Instruct \
  --data_root ./toy_images \
  --train_steps 200 \
  --per_device_batch_size 1 \
  --num_rollouts 4 \
  --use_lora false \
  --fixed_correct_index 0
```

### Key Arguments

- `--model_id`: HuggingFace model identifier
- `--data_root`: Directory containing candidate images
- `--candidates_per_episode`: Number of images per episode (default: 4)
- `--fixed_correct_index`: The index that is always correct (default: 0)
- `--train_steps`: Total training steps
- `--num_rollouts`: Number of rollouts per prompt for RLOO
- `--use_lora`: Enable LoRA for efficient fine-tuning

## Module Details

### `training.py`
Main entry point that orchestrates the training process. Handles argument parsing, initialization, and trainer setup.

### `environment.py`
Contains `PickImageEnv` which defines the task environment. Now uses a **fixed correct index** for deterministic toy examples while maintaining infrastructure flexibility.

### `dataset.py`
- `RepeatableEpisodeDataset`: Provides infinite episodes for RLOO training
- `VLMDataCollator`: Batches vision-language data for the model

### `model_utils.py`
- `load_qwen25_vl()`: Loads Qwen2.5-VL model with optional LoRA
- `generate_tool_call()`: Generates tool-call outputs from the model

### `reward.py`
`compute_rewards_from_generations()`: Converts model outputs to rewards based on correctness

### `utils/tool_calls.py`
- `make_tool_call_str()`: Creates canonical tool-call JSON
- `parse_tool_call_str()`: Parses generated text into tool-call format

## Requirements

```bash
pip install "transformers>=4.43" "trl>=0.8.6" accelerate deepspeed peft datasets pillow torchvision
pip install flash-attn --no-build-isolation  # Optional but recommended for H100
```

## Changes from Original

1. **Modular Structure**: Split monolithic script into focused modules
2. **Fixed Correct Index**: Environment now uses a configurable fixed correct index instead of random selection
3. **Better Documentation**: Each module is well-documented with clear responsibilities
4. **Extensibility**: Easy to extend for more complex tasks while maintaining clean architecture

