import os
import argparse
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

from config import TrainConfig
from utils_env import make_training_env, make_eval_env
from callbacks import LiveRenderCallback

def main():
    cfg = TrainConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_timesteps", type=int, default=cfg.total_timesteps)
    parser.add_argument("--seed", type=int, default=cfg.seed)
    parser.add_argument("--device", type=str, default=cfg.device)
    parser.add_argument("--live_render_every", type=int, default=1000)
    parser.add_argument("--use_curriculum", type=int, default=0)
    args = parser.parse_args()

    device = "cuda" if (args.device in ["auto","cuda"] and torch.cuda.is_available()) else "cpu"
    print("Using device:", device)

    os.makedirs(cfg.log_dir, exist_ok=True)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    # Optionally: curriculum (start small, increase difficulty)
    env_kwargs = {}
    if args.use_curriculum:
        # Start with easier levels; you can re-run later with harder kwargs
        env_kwargs = {"dim_room": (7,7), "num_boxes": 2}

    env = make_training_env(cfg.env_id, cfg.n_envs, cfg.n_stack, args.seed, env_kwargs=env_kwargs)

    model = PPO(
        policy="CnnPolicy",
        env=env,
        verbose=1,
        tensorboard_log=cfg.log_dir,
        device=device,
        seed=args.seed,
    )

    # Evaluation environment (same kwargs for fairness if curriculum off)
    eval_env = make_eval_env(cfg.env_id, seed=args.seed+1, env_kwargs=env_kwargs if args.use_curriculum else None)

    ckpt_cb = CheckpointCallback(save_freq=cfg.ckpt_freq, save_path=cfg.ckpt_dir, name_prefix="ppo_sokoban_step")
    eval_cb = EvalCallback(eval_env, best_model_save_path=cfg.ckpt_dir, n_eval_episodes=cfg.eval_episodes,
                           eval_freq=cfg.eval_freq, deterministic=True, render=False, verbose=1)
    live_cb = LiveRenderCallback(render_every=args.live_render_every, verbose=1)

    model.learn(total_timesteps=args.total_timesteps, callback=[ckpt_cb, eval_cb, live_cb])
    model.save("ppo_sokoban")
    env.close()
    eval_env.close()
    print("Training complete. Model saved to ppo_sokoban.zip")

if __name__ == "__main__":
    main()
