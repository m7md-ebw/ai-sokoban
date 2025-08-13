import os
import argparse
import torch
import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import VecFrameStack, VecMonitor, VecTransposeImage, SubprocVecEnv
from config import TrainConfig
from utils_env import ShapingWrapper

# === Try fast env, fallback to normal ===
try:
    from gym_sokoban.envs.sokoban_env_fast import SokobanEnvFast as SokobanEnv
except ImportError:
    from gym_sokoban.envs.sokoban_env import SokobanEnv

# === Pad observation to 160x160 ===
class PadObsTo160(gym.ObservationWrapper):
    def __init__(self, env, target_hw=(160, 160)):
        super().__init__(env)
        self.target_h, self.target_w = target_hw
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.target_h, self.target_w, 3), dtype=np.uint8
        )

    def observation(self, obs):
        h, w, c = obs.shape
        out = np.zeros((self.target_h, self.target_w, c), dtype=obs.dtype)
        y = (self.target_h - h) // 2
        x = (self.target_w - w) // 2
        out[y:y+h, x:x+w, :] = obs
        return out

# === Build single env with shaping ===
def make_sokoban_env(dim=7, boxes=2, max_steps=120, use_shaping=False):
    def _thunk():
        env = SokobanEnv(dim_room=(dim, dim), num_boxes=boxes, max_steps=max_steps)
        env = PadObsTo160(env, target_hw=(160, 160))
        if use_shaping:
            env = ShapingWrapper(env)
        return env
    return _thunk

# === Build vectorized env ===
def build_vec_env(dim, boxes, max_steps, n_envs, n_stack, seed, use_shaping=False):
    env = SubprocVecEnv([make_sokoban_env(dim, boxes, max_steps, use_shaping) for _ in range(n_envs)])
    env = VecMonitor(env)
    if n_stack and n_stack > 1:
        env = VecFrameStack(env, n_stack=n_stack)
    env = VecTransposeImage(env)
    print(f"[ENV CONFIG] dim_room=({dim},{dim}), num_boxes={boxes}, max_steps={max_steps}, n_envs={n_envs}")
    return env

# === Main training ===
def main():
    cfg = TrainConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_timesteps", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=cfg.seed)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--use_shaping", type=int, default=1)
    args = parser.parse_args()

    device = "cuda" if (args.device in ["auto", "cuda"] and torch.cuda.is_available()) else "cpu"
    print("Using device:", device)

    os.makedirs(cfg.log_dir, exist_ok=True)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    # Training env - EASY MODE
    env = build_vec_env(7, 2, 120, n_envs=6, n_stack=cfg.n_stack, seed=args.seed, use_shaping=bool(args.use_shaping))

    # PPO
    model = PPO(
        policy="CnnPolicy",
        env=env,
        verbose=1,
        tensorboard_log=cfg.log_dir,
        device=device,
        seed=args.seed,
        n_steps=512,
        batch_size=512,
        learning_rate=1e-4,
        clip_range=0.1,
        gamma=0.995,
        gae_lambda=0.95,
        ent_coef=0.02,
        vf_coef=1.0
    )

    # Eval env (same config, 1 worker)
    eval_env = build_vec_env(7, 2, 120, n_envs=1, n_stack=cfg.n_stack, seed=args.seed + 1, use_shaping=bool(args.use_shaping))

    # Callbacks
    ckpt_cb = CheckpointCallback(save_freq=50_000, save_path=cfg.ckpt_dir, name_prefix="ppo_sokoban_step")
    eval_cb = EvalCallback(eval_env, best_model_save_path=cfg.ckpt_dir, n_eval_episodes=3,
                           eval_freq=50_000, deterministic=True, render=False, verbose=1)

    # Train
    model.learn(total_timesteps=args.total_timesteps, callback=[ckpt_cb, eval_cb])
    model.save("ppo_sokoban_fresh")
    env.close()
    eval_env.close()
    print("Training complete. Model saved to ppo_sokoban_fresh.zip")

if __name__ == "__main__":
    main()
