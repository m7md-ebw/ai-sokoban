# pure_ppo.py
import os
import gym
import gym_sokoban
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecTransposeImage, VecNormalize
from stable_baselines3.common.env_util import make_vec_env

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def make_env():
    # Create vanilla Sokoban env with 1 box
    return gym_sokoban.envs.SokobanEnv(
        dim_room=(7, 7),
        num_boxes=1,
        max_steps=120
    )

def build_env(n_envs=8, seed=None):
    env = make_vec_env(make_env, n_envs=n_envs, seed=seed, vec_env_cls=SubprocVecEnv)
    env = VecTransposeImage(env)        # HWC → CHW
    env = VecFrameStack(env, n_stack=2) # frame stacking
    env = VecNormalize(env, norm_obs=False, norm_reward=True)
    return env

def main():
    os.makedirs("checkpoints_pure", exist_ok=True)
    tb_logdir = "runs/pure_ppo_tb"

    env = build_env(n_envs=8)

    # PPO with default CNN policy
    model = PPO(
        "CnnPolicy",
        env,
        device=DEVICE,
        verbose=1,
        tensorboard_log=tb_logdir,
        n_steps=256,
        batch_size=2048,
        n_epochs=4,
        learning_rate=2.5e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=42
    )

    model.learn(total_timesteps=500_000)
    model.save("checkpoints_pure/ppo_sokoban_pure")
    env.save("checkpoints_pure/vecnormalize.pkl")

    print("[done] Pure PPO model saved.")

if __name__ == "__main__":
    main()
