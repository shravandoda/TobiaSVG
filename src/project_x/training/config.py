import os
from dataclasses import dataclass


@dataclass
class TobiasTrainingConfig:
    # Gradient accumulation config
    GRADIENT_ACCUMULATION_STEPS: int = 3
    MICRO_BATCH_SIZE: int = 1

    # Core training config
    LR: float = 2e-4  # Standard for LoRA
    WEIGHT_DECAY: float = 0.01
    WARMUP_STEPS: int = 10
    MAX_TRAIN_STEPS: int = 1
    LOG_EVERY_STEPS: int = 10
    VALIDATE_EVERY_STEPS: int = 500
    MAX_VALIDATION_BATCHES: int = 50
    PREPROCESSING_WORKERS: int = int(os.environ.get("PREPROCESSING_WORKERS", "1"))
    SEED: int = 42

    # LORA
    LORA_RANK: int = 64
    LORA_ALPHA: int = 128
    LORA_DROPOUT: float = 0.1
    LORA_TARGET_MODULES: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )

    # Checkpointing
    PROJECT_DIR: str = os.environ.get(
        "PROJECT_DIR",
        "./artifacts/training/tobias-svg-qwen3.5-4b-lora",
    )
    SAVE_EVERY_STEPS: int = 500
    KEEP_LAST_CHECKPOINTS: int = 3


training_config = TobiasTrainingConfig()
