import os
import argparse
import gym
import gym_sokoban
import numpy as np
import imageio

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    DummyVecEnv, VecTransposeImage, VecFrameStack, VecNormalize
)

# must match training wrappers
from train_sb3_ppo_sokoban import (
    SokobanCtorWrapper, OneHotObsWrapper, TinyGridCNN,
    DEFAULT_MODEL_PATH, DEFAULT_NORM_PATH
)

def make_base_env(dim_room=(7,7), num_boxes=1, max_steps=120):
    e = gym.make("Sokoban-v0")
    e = SokobanCtorWrapper(e, dim_room=dim_room, num_boxes=num_boxes, max_steps=max_steps)
    e = OneHotObsWrapper(e)
    return e

def build_eval_env(norm_path, seed=123):
    venv = DummyVecEnv([lambda: make_base_env()])
    venv = VecTransposeImage(venv)
    venv = VecFrameStack(venv, n_stack=4)
    if os.path.exists(norm_path):
        venv = VecNormalize.load(norm_path, venv)
    else:
        venv = VecNormalize(venv, norm_obs=False, norm_reward=True)
    venv.training = False
    venv.norm_reward = True
    return venv

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH)
    p.add_argument("--norm", type=str, default=DEFAULT_NORM_PATH)
    p.add_argument("--save-video", action="store_true")
    p.add_argument("--video-path", type=str, default="videos/eval_play.mp4")
    args = p.parse_args()

    env = build_eval_env(args.norm)
    model = PPO.load(args.model, env=env)
    print("[eval] loaded:", args.model)

    # get underlying raw gym env for RGB render
    raw = env.envs[0]
    base = raw.env
    while hasattr(base, "env"):
        base = base.env

    total = 0.0
    frames = []
    for ep in range(args.episodes):
        obs = env.reset()
        done = False
        ep_ret = 0.0
        steps = 0
        while not done and steps < 1000:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, done, info = env.step(action)
            ep_ret += float(reward)
            steps += 1
            rgb = base.render(mode="rgb_array")
            frames.append(rgb)
        print(f"episode {ep+1}: return={ep_ret:.2f}, steps={steps}")
        total += ep_ret

    print(f"average return over {args.episodes} eps: {total/args.episodes:.2f}")

    if args.save_video:
        os.makedirs(os.path.dirname(args.video_path), exist_ok=True)
        with imageio.get_writer(args.video_path, fps=8) as w:
            for f in frames:
                w.append_data(f)
        print("[video] saved to", args.video_path)

if __name__ == "__main__":
    main()
