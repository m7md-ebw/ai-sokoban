import os
from typing import Optional
import gym
import numpy as np
from PIL import Image

try:
    import gym_sokoban  
except Exception:
    pass
try:
    import boxoban  
except Exception:
    pass

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

class SokobanHRLWrapper(gym.Wrapper):
    """
    Wrapper tuned for small Sokoban: downscale obs, small step penalty, RGB frames for video.
    Backward-compatible with old/new Gym APIs.
    """

    def __init__(self, env: gym.Env, obs_size: int = 64, step_penalty: float = -0.005):
        super().__init__(env)
        self.obs_size = obs_size
        self.step_penalty = step_penalty
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(3, obs_size, obs_size), dtype=np.uint8
        )

    def _process_obs(self, obs: np.ndarray) -> np.ndarray:
        im = Image.fromarray(obs)
        im = im.resize((self.obs_size, self.obs_size), Image.NEAREST)
        arr = np.asarray(im, dtype=np.uint8)
        arr = np.transpose(arr, (2, 0, 1))
        return arr.copy()

    def reset(self, **kwargs):
        try:
            obs, info = self.env.reset(**kwargs)
        except Exception:
            obs = self.env.reset()
            info = {}
        return self._process_obs(obs), info

    def step(self, action: int):
        try:
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
        except Exception:
            obs, reward, done, info = self.env.step(action)
            truncated = False
        shaped = float(reward) + self.step_penalty
        return self._process_obs(obs), shaped, done, truncated, info


def make_sokoban_env(seed: Optional[int] = None, obs_size: int = 64, env_id: Optional[str] = "auto") -> gym.Env:
    if env_id == "auto":
        preferred = [
            "Sokoban-small-v0",
            "Sokoban-v0",
        ]
        available = [spec.id for spec in gym.envs.registry.all()]
        chosen = None
        for pid in preferred:
            if pid in available:
                chosen = pid
                break
        if chosen is None:
            for spec_id in available:
                if ("Sokoban" in spec_id) or ("Boxoban" in spec_id):
                    chosen = spec_id
                    break
        if chosen is None:
            raise RuntimeError("No Sokoban/Boxoban env registered. Install/import gym-sokoban or boxoban.")
        env_id = chosen
        print(f"[INFO] Using env id: {env_id}")

    cfg = dict(room_size=7, num_boxes=2, max_steps=120, render_mode="rgb_array")
    try:
        env = gym.make(env_id, **cfg)
    except TypeError:
        try:
            env = gym.make(env_id, render_mode="rgb_array")
        except Exception:
            env = gym.make(env_id)

    if seed is not None:
        try:
            env.reset(seed=seed)
        except TypeError:
            try:
                env.seed(seed)
            except Exception:
                pass
        try:
            env.action_space.seed(seed)
        except Exception:
            pass
        try:
            env.observation_space.seed(seed)
        except Exception:
            pass

    return SokobanHRLWrapper(env, obs_size=obs_size)
