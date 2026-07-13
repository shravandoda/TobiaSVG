import os
from dataclasses import dataclass


@dataclass
class TobiasTrainingConfig:
    # Gradient accumulation config
    GRADIENT_ACCUMULATION_STEPS: int = 3
    MICRO_BATCH_SIZE: int = 1

    # Core training config
    LR: float = float(os.environ.get("LR", "2e-4"))
    WEIGHT_DECAY: float = 0.01
    WARMUP_STEPS: int = int(os.environ.get("WARMUP_STEPS", "1000"))
    MAX_TRAIN_STEPS: int = int(os.environ.get("MAX_TRAIN_STEPS", "36000"))
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
    RESUME_CHECKPOINT: str | None = os.environ.get("RESUME_CHECKPOINT")
    RESET_SCHEDULER_ON_RESUME: bool = (
        os.environ.get("RESET_SCHEDULER_ON_RESUME", "false").lower() == "true"
    )
    SAVE_EVERY_STEPS: int = 500
    KEEP_LAST_CHECKPOINTS: int = 3

    # Hugging Face Hub backup
    HUB_REPO_ID: str | None = os.environ.get("HF_MODEL_REPO_ID")
    PUSH_TO_HUB_EVERY_STEPS: int = 5_000


training_config = TobiasTrainingConfig()
