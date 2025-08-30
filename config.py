from dataclasses import dataclass

@dataclass
class TrainConfig:
    env_id: str = "Sokoban-v0"
    n_envs: int = 1
    n_stack: int = 4
    total_timesteps: int = 200_000
    seed: int = 42
    log_dir: str = "./ppo_sokoban_tensorboard"
    ckpt_dir: str = "./checkpoints"
    eval_freq: int = 50_000  # steps
    ckpt_freq: int = 50_000  # steps
    eval_episodes: int = 3
    device: str = "auto"  # "cuda" or "cpu" or "auto"

# Curriculum settings (optional / simple)

@dataclass
class CurriculumStage:
    kwargs: dict
    min_steps: int

# Example curriculum list (small rooms / fewer boxes → larger)
CURRICULUM = [
    CurriculumStage(kwargs={"dim_room": (7,7), "num_boxes": 2}, min_steps=50_000),
    CurriculumStage(kwargs={"dim_room": (8,8), "num_boxes": 3}, min_steps=100_000),
    CurriculumStage(kwargs={"dim_room": (10,10), "num_boxes": 4}, min_steps=150_000),
]
