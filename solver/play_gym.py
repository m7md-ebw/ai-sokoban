import os, argparse
import numpy as np
import matplotlib.pyplot as plt

try:
    import gym
except ImportError:
    import gymnasium as gym as gym  

def get_action_map(env):
    lookup = None
    if hasattr(env, "unwrapped") and hasattr(env.unwrapped, "get_action_lookup"):
        try:
            lookup = env.unwrapped.get_action_lookup()
        except Exception:
            lookup = None
    if isinstance(lookup, dict):
        inv = {}
        for k, v in lookup.items():
            if isinstance(v, (list, tuple)) and v:
                v = v[0]
            if isinstance(v, bytes):
                v = v.decode("utf-8", errors="ignore")
            if isinstance(v, str):
                inv[v.lower()] = int(k)
        if all(k in inv for k in ("up","down","left","right")):
            return {"u":inv["up"], "d":inv["down"], "l":inv["left"], "r":inv["right"]}
    n = int(env.action_space.n)
    cand = {"l":0, "r":1, "u":2, "d":3}
    if all(0 <= cand[k] < n for k in ("u","d","l","r")):
        return cand
    return {"l":0, "r":1, "u":2, "d":3}

def to_dirs(solution: str):
    return [ch.lower() for ch in solution if ch.lower() in ("u","d","l","r")]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--env_id", default="Sokoban-v0")
    ap.add_argument("--moves", default="")
    ap.add_argument("--delay", type=float, default=0.08)
    ap.add_argument("--render_mode", default="rgb_array", choices=["human","rgb_array"])
    args = ap.parse_args()

    try:
        env = gym.make(args.env_id, render_mode=args.render_mode)
    except TypeError:
        env = gym.make(args.env_id)

    try:
        obs, info = env.reset(seed=None)
    except TypeError:
        obs = env.reset()

    amap = get_action_map(env)
    print("Action map:", amap)

    if args.render_mode == "rgb_array":
        plt.figure()
        frame = env.render()
        if isinstance(frame, (list, tuple)): frame = frame[0]
        plt.imshow(frame); plt.axis("off"); plt.pause(0.001)

    total_reward = 0.0
    terminated = truncated = False
    for d in to_dirs(args.moves):
        a = amap[d]
        step_out = env.step(a)
        if len(step_out) == 5:
            obs, reward, terminated, truncated, info = step_out
        else:
            obs, reward, done, info = step_out
            terminated, truncated = bool(done), False
        total_reward += float(reward)
        if args.render_mode == "rgb_array":
            frame = env.render()
            if isinstance(frame, (list, tuple)): frame = frame[0]
            plt.imshow(frame); plt.axis("off"); plt.pause(args.delay)
        if terminated or truncated:
            break
    print(f"Done. total_reward={total_reward:.3f}, terminated={terminated}, truncated={truncated}")
    if args.render_mode == "rgb_array":
        plt.show()
    env.close()
