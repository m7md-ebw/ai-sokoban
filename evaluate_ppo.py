import argparse, time, numpy as np
import gym, gym_sokoban
from stable_baselines3 import PPO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="ppo_sokoban.zip")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--deterministic", type=int, default=0)
    args = parser.parse_args()

    env = gym.make("Sokoban-v0")
    model = PPO.load(args.model, device="cuda")

    for ep in range(args.episodes):
        obs = env.reset()
        done = False
        steps = 0
        total_reward = 0.0
        while not done and steps < args.max_steps:
            env.render(mode="human")
            action, _ = model.predict(obs, deterministic=bool(args.deterministic))
            action = int(action) if not isinstance(action, np.ndarray) else int(action.item())
            obs, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            time.sleep(0.02)
        print(f"Episode {ep+1}: steps={steps}, total_reward={total_reward}")
    env.close()

if __name__ == "__main__":
    main()
