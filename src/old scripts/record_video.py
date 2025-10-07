import argparse, os
import gym
import gym_sokoban
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="ppo_sokoban.zip")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--deterministic", type=int, default=1)
    parser.add_argument("--video_len", type=int, default=500)
    args = parser.parse_args()

    os.makedirs("./videos", exist_ok=True)
    def make_env():
        return gym.make("Sokoban-v0")
    env = DummyVecEnv([make_env])
    env = VecVideoRecorder(env, "./videos", record_video_trigger=lambda x: x==0, video_length=args.video_len, name_prefix="ppo_sokoban_eval")

    model = PPO.load(args.model, device="cuda")
    obs = env.reset()
    for _ in range(args.steps):
        action, _ = model.predict(obs, deterministic=bool(args.deterministic))
        obs, reward, done, info = env.step(action)
        env.render(mode="rgb_array")
        if done.any():
            obs = env.reset()
    env.close()
    print("Video saved to ./videos")

if __name__ == "__main__":
    main()
