import os
import gym
import gym_sokoban
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack
import torch

# Paths
TENSORBOARD_DIR = "./ppo_sokoban_tensorboard/"

print("SB3 will use:", "cuda" if torch.cuda.is_available() else "cpu")
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create env (vectorized), then add frame stacking (gives tiny temporal context)
env = make_vec_env("Sokoban-v0", n_envs=1)
env = VecFrameStack(env, n_stack=4)

# PPO with CNN policy (better for image obs)
model = PPO(
    policy="CnnPolicy",
    env=env,
    verbose=1,
    tensorboard_log=TENSORBOARD_DIR,
    device=device
)

# Train (increase as needed)
model.learn(total_timesteps=200_000)  # start with 200k; plan to go higher later

# Save
model.save("ppo_sokoban")
print("Model saved!")

env.close()
