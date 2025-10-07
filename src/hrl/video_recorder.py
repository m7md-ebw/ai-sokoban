from typing import Callable
import imageio
import numpy as np


def record_episode(env, select_action: Callable, max_steps: int, out_path: str, fps: int = 15):
    """Run one episode, save an .mp4 using imageio-ffmpeg. select_action(obs)->int"""
    frames = []
    try:
        obs, info = env.reset()
    except Exception:
        obs = env.reset()
        info = {}
    done = False
    steps = 0
    while not done and steps < max_steps:
        frame = None
        try:
            frame = env.render()
        except Exception:
            pass
        if frame is None:
            try:
                frame = env.env.render(mode="rgb_array")
            except Exception:
                pass
        if isinstance(frame, np.ndarray):
            frames.append(frame)
        action = select_action(obs)
        try:
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
        except Exception:
            obs, reward, done, info = env.step(int(action))
        steps += 1
    if frames:
        imageio.mimsave(out_path, frames, fps=fps)
    return steps