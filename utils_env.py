# utils_env.py
import gym
import gym_sokoban
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack, VecMonitor
from stable_baselines3.common.vec_env import VecTransposeImage  # <-- add this

def make_training_env(env_id="Sokoban-v0", n_envs=1, n_stack=4, seed=42, env_kwargs=None):
    env_kwargs = env_kwargs or {}
    venv = make_vec_env(env_id, n_envs=n_envs, seed=seed, env_kwargs=env_kwargs)
    venv = VecMonitor(venv)
    if n_stack and n_stack > 1:
        venv = VecFrameStack(venv, n_stack=n_stack)
    # SB3 usually auto-wraps this in training, but being explicit is fine:
    venv = VecTransposeImage(venv)
    return venv

def make_eval_env(env_id="Sokoban-v0", seed=0, n_stack=4, env_kwargs=None):
    env_kwargs = env_kwargs or {}
    venv = make_vec_env(env_id, n_envs=1, seed=seed, env_kwargs=env_kwargs)
    venv = VecMonitor(venv)
    if n_stack and n_stack > 1:
        venv = VecFrameStack(venv, n_stack=n_stack)
    venv = VecTransposeImage(venv)   # <-- ensure same as training
    return venv
