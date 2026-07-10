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


training_config = TobiasTrainingConfig()
