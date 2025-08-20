import os
import argparse
from typing import Any, Optional, Tuple, List

import gym
import gym_sokoban
import numpy as np
import torch
import torch.nn as nn
from gym import ObservationWrapper, RewardWrapper, spaces

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import (
    VecFrameStack, VecTransposeImage, VecNormalize,
    DummyVecEnv, SubprocVecEnv
)
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure


# -------------------
# config / paths
# -------------------
DEFAULT_MODEL_PATH = "checkpoints/ppo_sokoban.zip"
DEFAULT_NORM_PATH  = "checkpoints/vecnormalize.pkl"
DEFAULT_TB_DIR     = "runs/ppo_sokoban_tb"
DEFAULT_TRAIN_VID_DIR = "videos_train"   # training videos here
DEFAULT_EVAL_VID_DIR  = "videos_eval"    # evaluation videos here

DIM_ROOM = (7, 7)
NUM_BOXES = 1
MAX_STEPS = 120

# small “time pressure” to avoid dithering
STEP_PENALTY = 0.008

# penalties / filters you can tune (can also be overridden via CLI)
NOOP_PENALTY = 0.03          # bumping wall / no movement
DEADLOCK_PENALTY = 0.2     # pushing box into corner/3-wall pocket (not on goal)
MIN_BOX_GOAL_DIST = 3        # reject trivial rooms where |box-goal| (Manhattan) < this

# distance shaping (dense hints)
DIST_SHAPING_W = 0.01        # per-step *delta* weight; 0 disables
DIST_SHAPING_CLIP = 3.0      # clip abs(delta reward) to avoid spikes
USE_HUNGARIAN = False       # try SciPy Hungarian if available (multi-box); else greedy

# stagnation (no box moved for N steps)
STAGNATION_STEPS = 60        # 0 disables
STAGNATION_PENALTY = 0.1     # applied on early termination due to stagnation
EARLY_TERM_ON_DEADLOCK = True  # end episode immediately on new deadlock (strong signal)

print("SB3 device:", "cuda" if torch.cuda.is_available() else "cpu")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------
# helpers (GRID-BASED)
# -------------------
def get_grid(env) -> Optional[np.ndarray]:
    """Return HxW int grid from gym_sokoban (room_state)."""
    try:
        return env.unwrapped.room_state  # 0..6 codes
    except Exception:
        return None

# tile codes in gym_sokoban:
# 0 floor, 1 wall, 2 goal, 3 player, 4 box, 5 box_on_goal, 6 player_on_goal

def player_pos_from_grid(grid: np.ndarray) -> Optional[Tuple[int, int]]:
    ys, xs = np.where((grid == 3) | (grid == 6))
    return (int(ys[0]), int(xs[0])) if len(ys) else None

def walls_mask_from_grid(grid: np.ndarray) -> np.ndarray:
    return (grid == 1)

def goals_mask_from_grid(grid: np.ndarray) -> np.ndarray:
    return (grid == 2)

def boxes_free_mask_from_grid(grid: np.ndarray) -> np.ndarray:
    return (grid == 4)

def boxes_any_mask_from_grid(grid: np.ndarray) -> np.ndarray:
    return (grid == 4) | (grid == 5)

def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def _greedy_assign(boxes: List[Tuple[int,int]], goals: List[Tuple[int,int]]) -> int:
    """Sum of Manhattan distances via greedy matching (fast, no SciPy)."""
    remaining_goals = goals[:]
    total = 0
    for b in boxes:
        dists = [(_manhattan(b, g), i) for i, g in enumerate(remaining_goals)]
        dists.sort(key=lambda x: x[0])
        d, idx = dists[0]
        total += d
        remaining_goals.pop(idx)
    return total

def _hungarian_assign(boxes: List[Tuple[int,int]], goals: List[Tuple[int,int]]) -> int:
    """Optional SciPy Hungarian assignment if available. Falls back to greedy."""
    try:
        from scipy.optimize import linear_sum_assignment
        import numpy as np
        n = len(boxes)
        C = np.zeros((n, n), dtype=np.int32)
        for i, b in enumerate(boxes):
            for j, g in enumerate(goals):
                C[i, j] = _manhattan(b, g)
        row, col = linear_sum_assignment(C)
        return int(C[row, col].sum())
    except Exception:
        return _greedy_assign(boxes, goals)

# -------------------
# wrappers
# -------------------
class SokobanCtorWrapper(gym.Wrapper):
    """Constructs a fresh SokobanEnv with desired params."""
    def __init__(self, env: gym.Env, dim_room=(7, 7), num_boxes=1, max_steps=120):
        new_env = gym_sokoban.envs.sokoban_env.SokobanEnv(
            dim_room=dim_room, num_boxes=num_boxes, max_steps=max_steps
        )
        super().__init__(new_env)

class FlexibleObsWrapper(ObservationWrapper):
    """RGB passthrough; we are not using one_hot."""
    def __init__(self, env: gym.Env):
        super().__init__(env)
        obs_space = self.env.observation_space
        assert isinstance(obs_space, spaces.Box), "Expected Box observation space"
        h, w, c = obs_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(h, w, c), dtype=np.uint8)
        print("[ObsWrapper] using native RGB (uint8[0,255])")

    def observation(self, obs: Any) -> np.ndarray:
        return obs

class StepPenaltyWrapper(RewardWrapper):
    """Gentle per-step time pressure."""
    def __init__(self, env: gym.Env, step_penalty: float = 0.01):
        super().__init__(env)
        self.step_penalty = float(step_penalty)
    def reward(self, reward: float) -> float:
        return reward - self.step_penalty if self.step_penalty > 0.0 else reward

class NoOpMovePenaltyWrapper(gym.Wrapper):
    """Penalize actions that do not change player position (wall-bumps / blocked pushes)."""
    def __init__(self, env, penalty=0.02):
        super().__init__(env)
        self.penalty = float(penalty)

    def _player_pos(self) -> Optional[Tuple[int,int]]:
        grid = get_grid(self.env)
        return player_pos_from_grid(grid) if grid is not None else None

    def step(self, action):
        prev = self._player_pos()
        obs, reward, done, info = self.env.step(action)
        if prev is not None:
            new = self._player_pos()
            if new == prev:
                reward -= self.penalty
                info = dict(info); info["noop_penalty"] = self.penalty
        return obs, reward, done, info

class BoxDeadlockPenaltyWrapper(gym.Wrapper):
    """
    Penalize when a push newly puts a non-goal box into a deadlock (corner or 3-wall pocket).
    Uses room_state grid. Optionally terminate immediately on new deadlock.
    """
    def __init__(self, env, penalty=0.5, early_term=True):
        super().__init__(env)
        self.penalty = float(penalty)
        self.early_term = bool(early_term)
        self._prev_deadlocked = None

    @staticmethod
    def _is_dead(y, x, walls_mask: np.ndarray) -> bool:
        H, W = walls_mask.shape
        def wall_at(yy, xx):
            if yy < 0 or yy >= H or xx < 0 or xx >= W:
                return True
            return bool(walls_mask[yy, xx])
        U = wall_at(y-1, x); D = wall_at(y+1, x)
        L = wall_at(y, x-1); R = wall_at(y, x+1)
        corner = (U and L) or (U and R) or (D and L) or (D and R)
        three  = (U + D + L + R) >= 3
        return corner or three

    @classmethod
    def _deadboxed(cls, grid: np.ndarray) -> set:
        walls = walls_mask_from_grid(grid)
        boxes_free = boxes_free_mask_from_grid(grid)  # exclude boxes_on_goal
        ys, xs = np.where(boxes_free)
        dead = set()
        for y, x in zip(ys, xs):
            if cls._is_dead(int(y), int(x), walls):
                dead.add((int(y), int(x)))
        return dead

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        grid = get_grid(self.env)
        self._prev_deadlocked = self._deadboxed(grid) if grid is not None else None
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if self._prev_deadlocked is None:
            return obs, reward, done, info
        grid = get_grid(self.env)
        if grid is None:
            return obs, reward, done, info
        now = self._deadboxed(grid)
        new_dead = now.difference(self._prev_deadlocked)
        if new_dead:
            penalty = self.penalty * len(new_dead)
            reward -= penalty
            info = dict(info)
            info["deadlock_penalty"] = penalty
            info["deadlocked_boxes"] = list(sorted(new_dead))
            if self.early_term:
                done = True
                info["terminated_by_wrapper"] = "deadlock"
        self._prev_deadlocked = now
        return obs, reward, done, info

class ResetConstraintWrapper(gym.Wrapper):
    """
    Reject trivial rooms at reset (for 1-box): require Manhattan(box, goal) >= min_box_goal_dist.
    Uses room_state grid.
    """
    def __init__(self, env, min_box_goal_dist=3, max_tries=50):
        super().__init__(env)
        self.min_d = int(min_box_goal_dist)
        self.max_tries = int(max_tries)

    def _ok(self) -> bool:
        grid = get_grid(self.env)
        if grid is None:
            return True
        goals = np.argwhere(goals_mask_from_grid(grid))
        boxes = np.argwhere(boxes_any_mask_from_grid(grid))
        if len(boxes) != 1 or len(goals) != 1:
            return True
        by, bx = boxes[0]; gy, gx = goals[0]
        manhattan = abs(int(by) - int(gy)) + abs(int(bx) - int(gx))
        return manhattan >= self.min_d

    def reset(self, **kwargs):
        for _ in range(self.max_tries):
            obs = self.env.reset(**kwargs)
            if self._ok():
                return obs
        return obs

class DistanceShapingWrapper(gym.Wrapper):
    """
    Dense shaping: reward += w * clamp((prev_dist_sum - dist_sum), [-clip, clip])
      dist_sum = player->nearest_box + assignment(boxes->goals)
    Uses room_state grid.
    """
    def __init__(self, env, weight=0.01, clip=3.0, use_hungarian=False):
        super().__init__(env)
        self.w = float(weight)
        self.clip = float(clip)
        self.use_hungarian = bool(use_hungarian)
        self.prev_metric = None

    def _metric(self) -> Optional[int]:
        grid = get_grid(self.env)
        if grid is None:
            return None

        p = player_pos_from_grid(grid)
        goals = np.argwhere(goals_mask_from_grid(grid))
        boxes_free = np.argwhere(grid == 4)
        boxes_goal = np.argwhere(grid == 5)
        if p is None or len(goals) == 0:
            return None

        # player -> nearest free box (fallback to box_on_goal)
        box_pool = boxes_free if len(boxes_free) > 0 else boxes_goal
        if len(box_pool) == 0:
            return None
        b_list = [tuple(map(int, b)) for b in box_pool]
        g_list = [tuple(map(int, g)) for g in goals]

        d_p = min(_manhattan(p, b) for b in b_list)
        if len(b_list) <= len(g_list):
            assign_cost = (_hungarian_assign if self.use_hungarian else _greedy_assign)(b_list, g_list)
        else:
            # more boxes than goals (unlikely in standard Sokoban, but safe)
            all_pairs = sorted(_manhattan(b, g) for b in b_list for g in g_list)
            assign_cost = sum(all_pairs[:len(g_list)])
        return int(d_p + assign_cost)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.prev_metric = self._metric()
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if self.w <= 0.0:
            return obs, reward, done, info
        cur = self._metric()
        if self.prev_metric is not None and cur is not None:
            delta = float(self.prev_metric - cur)  # improvement → positive
            delta = float(np.clip(delta, -self.clip, self.clip))
            shaped = self.w * delta
            reward += shaped
            info = dict(info); info["dist_shaping"] = shaped
        self.prev_metric = cur
        return obs, reward, done, info

class StagnationResetWrapper(gym.Wrapper):
    """
    If no box movement for N steps, early terminate + small penalty.
    Uses room_state grid.
    """
    def __init__(self, env, patience=30, penalty=0.1):
        super().__init__(env)
        self.patience = int(patience)
        self.penalty = float(penalty)
        self._prev_boxes = None
        self._still = 0

    def _boxes_set(self) -> Optional[set]:
        grid = get_grid(self.env)
        if grid is None:
            return None
        ys, xs = np.where(boxes_any_mask_from_grid(grid))
        return set(zip(map(int, ys), map(int, xs)))

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self._prev_boxes = self._boxes_set()
        self._still = 0
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        cur = self._boxes_set()
        if self.patience > 0 and cur is not None and self._prev_boxes is not None:
            moved = (cur != self._prev_boxes)
            self._still = 0 if moved else (self._still + 1)
            self._prev_boxes = cur
            if self._still >= self.patience:
                reward -= self.penalty
                done = True
                info = dict(info)
                info["terminated_by_wrapper"] = "stagnation"
                info["stagnation_penalty"] = self.penalty
                self._still = 0
        return obs, reward, done, info

# -------------------
# tiny cnn
# -------------------
class BalancedGridCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        c, h, w = observation_space.shape
        self.cnn = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, features_dim), nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(obs))

# -------------------
# env builders
# -------------------
def make_base_env():
    e = gym.make("Sokoban-v0")
    e = SokobanCtorWrapper(e, dim_room=DIM_ROOM, num_boxes=NUM_BOXES, max_steps=MAX_STEPS)

    if NUM_BOXES == 1 and MIN_BOX_GOAL_DIST > 0:
        e = ResetConstraintWrapper(e, min_box_goal_dist=MIN_BOX_GOAL_DIST)

    e = FlexibleObsWrapper(e)

    if STEP_PENALTY > 0.0:
        e = StepPenaltyWrapper(e, step_penalty=STEP_PENALTY)
    if NOOP_PENALTY > 0.0:
        e = NoOpMovePenaltyWrapper(e, penalty=NOOP_PENALTY)
    if DIST_SHAPING_W > 0.0:
        e = DistanceShapingWrapper(e, weight=DIST_SHAPING_W, clip=DIST_SHAPING_CLIP, use_hungarian=USE_HUNGARIAN)
    if DEADLOCK_PENALTY > 0.0:
        e = BoxDeadlockPenaltyWrapper(e, penalty=DEADLOCK_PENALTY, early_term=EARLY_TERM_ON_DEADLOCK)
    if STAGNATION_STEPS > 0:
        e = StagnationResetWrapper(e, patience=STAGNATION_STEPS, penalty=STAGNATION_PENALTY)

    return e

def build_train_env(n_envs: int, seed: Optional[int], vec_env_cls, load_norm_from: Optional[str] = None, n_stack: int = 2):
    venv = make_vec_env(make_base_env, n_envs=n_envs, seed=seed, vec_env_cls=vec_env_cls)
    venv = VecTransposeImage(venv)          # HWC -> CHW
    venv = VecFrameStack(venv, n_stack=n_stack)
    if load_norm_from and os.path.exists(load_norm_from):
        venv = VecNormalize.load(load_norm_from, venv)
        venv.training = True
        venv.norm_reward = True
    else:
        venv = VecNormalize(venv, norm_obs=False, norm_reward=True)
    return venv

def build_eval_env(load_norm_from: Optional[str], seed: Optional[int], n_stack: int = 2):
    """Separate 1-env Dummy pipeline for rendering evaluation clips."""
    venv = make_vec_env(make_base_env, n_envs=1, seed=seed, vec_env_cls=DummyVecEnv)
    venv = VecTransposeImage(venv)
    venv = VecFrameStack(venv, n_stack=n_stack)
    if load_norm_from and os.path.exists(load_norm_from):
        venv = VecNormalize.load(load_norm_from, venv)
    else:
        venv = VecNormalize(venv, norm_obs=False, norm_reward=True)
    venv.training = False
    venv.norm_reward = True
    return venv

# -------------------
# callbacks
# -------------------
class SaveEveryNSteps(BaseCallback):
    def __init__(self, save_every: int, model_path: str, norm_path: str, verbose=0):
        super().__init__(verbose)
        self.save_every = int(save_every)
        self.model_path = model_path
        self.norm_path  = norm_path
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.norm_path), exist_ok=True)
        self._last_save = 0
    def _on_step(self) -> bool:
        # schedule by true timesteps, not callback calls
        if (self.model.num_timesteps - self._last_save) >= self.save_every:
            self._last_save = self.model.num_timesteps
            self.model.save(self.model_path)
            try:
                self.training_env.save(self.norm_path)
            except Exception:
                pass
            if self.verbose:
                print(f"[checkpoint] saved {self.model_path} + {self.norm_path} at ts={self._last_save}")
        return True

class TrainingEpisodeRecorderCallback(BaseCallback):
    """
    Record full training episodes from the training env (not eval).
    Requires DummyVecEnv with n_envs=1.
    """
    def __init__(self, video_dir: str, video_len_cap: int = 0, record_k: int = 1, show_live: bool = False, verbose=0):
        super().__init__(verbose)
        self.video_dir = video_dir
        self.video_len_cap = int(video_len_cap)  # 0 = no cap
        self.record_k = max(1, int(record_k))
        self.show_live = show_live
        os.makedirs(self.video_dir, exist_ok=True)
        self.frames = []
        self.episode_idx = 0
        self._cv2 = None
        if self.show_live:
            try:
                import cv2
                self._cv2 = cv2
            except Exception:
                self._cv2 = None
                if self.verbose:
                    print("cv2 not available, live preview disabled.")
        self._base = None  # unwrapped base gym env
        self._ep_return = 0.0

    def _unwrap_base_env(self):
        env = self.model.get_env()
        raw = env.envs[0] if hasattr(env, "envs") else env
        base = raw
        while hasattr(base, "env"):
            base = base.env
        return base

    def _on_training_start(self) -> None:
        self._base = self._unwrap_base_env()
        self._ep_return = 0.0
        try:
            self.frames.append(self._base.render(mode="rgb_array"))
        except Exception:
            pass

    def _save_current_episode(self, ep_reward: float):
        if not self.frames:
            return
        stem = f"train_ep{self.episode_idx}_R{ep_reward:.1f}"
        mp4 = os.path.join(self.video_dir, f"{stem}.mp4")
        try:
            import imageio
            with imageio.get_writer(mp4, fps=8) as w:
                for f in self.frames:
                    w.append_data(f)
            if self.verbose:
                print(f"[video] saved {mp4}")
        except Exception:
            gif = os.path.join(self.video_dir, f"{stem}.gif")
            try:
                import imageio
                imageio.mimsave(gif, self.frames, fps=8)
                if self.verbose:
                    print(f"[video] saved GIF fallback {gif}")
            except Exception as e2:
                if self.verbose:
                    print(f"[video] failed to save GIF: {e2}")

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        rewards = self.locals.get("rewards")
        # accumulate return for env 0
        if rewards is not None:
            r = float(rewards[0]) if isinstance(rewards, (list, np.ndarray)) else float(rewards)
            self._ep_return += r
        try:
            rgb = self._base.render(mode="rgb_array")
            self.frames.append(rgb)
            if self._cv2 is not None:
                self._cv2.imshow("Sokoban (training)", rgb[:, :, ::-1])
                self._cv2.waitKey(1)
        except Exception:
            pass

        if self.video_len_cap > 0 and len(self.frames) >= self.video_len_cap:
            self._save_current_episode(ep_reward=self._ep_return)
            self.frames = []

        if dones is not None and len(dones) > 0 and bool(dones[0]):
            self.episode_idx += 1
            if (self.episode_idx % self.record_k) == 0:
                self._save_current_episode(ep_reward=self._ep_return)
            self._ep_return = 0.0
            self.frames = []
            try:
                self.frames.append(self._base.render(mode="rgb_array"))
            except Exception:
                pass
        return True

class PeriodicEvalAndVideoCallback(BaseCallback):
    """
    Every N true timesteps:
      - runs eval on a separate DummyVecEnv (renderable)
      - saves MP4; falls back to GIF if few/no frames or ffmpeg issues
      - optional: keep going across episode ends to reach fixed video_len
    """
    def __init__(self, eval_env, every_steps=0, video_dir=DEFAULT_EVAL_VID_DIR,
                 video_len=300, deterministic=True, show_live=False,
                 fixed_len=True, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.every_steps = int(every_steps)
        self.video_dir = video_dir
        self.video_len = int(video_len)
        self.deterministic = deterministic
        self.show_live = show_live
        self.fixed_len = fixed_len
        os.makedirs(self.video_dir, exist_ok=True)
        self._cv2 = None
        self._last_eval = 0
        if self.show_live:
            try:
                import cv2
                self._cv2 = cv2
            except Exception:
                self._cv2 = None
                if self.verbose:
                    print("cv2 not available, will only save videos.")

    @staticmethod
    def _safe_render(base):
        try:
            rgb = base.render(mode="rgb_array")
            # ensure proper dtype/contiguous
            if rgb is not None:
                rgb = np.asarray(rgb, dtype=np.uint8)
            return rgb
        except Exception as e:
            return None

    def _on_step(self) -> bool:
        # schedule by true timesteps
        if (self.model.num_timesteps - self._last_eval) < self.every_steps:
            return True
        self._last_eval = self.model.num_timesteps

        obs = self.eval_env.reset()
        # unwrap to base for rgb render
        raw = self.eval_env.envs[0]  # DummyVecEnv
        base = raw
        while hasattr(base, "env"):
            base = base.env

        frames = []
        ep_reward = 0.0
        steps = 0

        # capture an initial frame right after reset
        f0 = self._safe_render(base)
        if f0 is not None:
            frames.append(f0)
            if self._cv2 is not None:
                self._cv2.imshow("Sokoban (eval)", f0[:, :, ::-1]); self._cv2.waitKey(1)
        else:
            if self.verbose:
                print("[eval video] initial render failed; will try after first step.")

        done = False
        while steps < self.video_len:
            action, _ = self.model.predict(obs, deterministic=self.deterministic)
            obs, reward, done_arr, info = self.eval_env.step(action)
            # VecEnv returns arrays
            r = float(reward[0]) if isinstance(reward, (list, np.ndarray)) else float(reward)
            done = bool(done_arr[0]) if isinstance(done_arr, (list, np.ndarray)) else bool(done_arr)
            ep_reward += r
            steps += 1

            rgb = self._safe_render(base)
            if rgb is not None:
                frames.append(rgb)
                if self._cv2 is not None:
                    self._cv2.imshow("Sokoban (eval)", rgb[:, :, ::-1]); self._cv2.waitKey(1)
            else:
                if self.verbose:
                    print(f"[eval video] render failed at step {steps}")

            if done and not self.fixed_len:
                break
            if done and self.fixed_len:
                # keep video going: reset episode and continue recording
                obs = self.eval_env.reset()

        stem = f"eval_ts_{self._last_eval}_R{ep_reward:.1f}"
        mp4 = os.path.join(self.video_dir, f"{stem}.mp4")

        def save_gif(frames, stem):
            gif = os.path.join(self.video_dir, f"{stem}.gif")
            try:
                import imageio
                # even a single frame gif works; but add duplicate if only one frame for a visible length
                seq = frames if len(frames) > 1 else (frames * 2)
                imageio.mimsave(gif, seq, fps=8)
                if self.verbose:
                    print(f"[eval video] saved GIF fallback {gif} (frames={len(frames)})")
            except Exception as e2:
                if self.verbose:
                    print(f"[eval video] failed to save GIF: {e2}")

        if len(frames) < 2:
            # avoid 0s MP4s – use GIF fallback
            if self.verbose:
                print(f"[eval video] too few frames ({len(frames)}); writing GIF instead.")
            save_gif(frames, stem)
            return True

        try:
            import imageio
            with imageio.get_writer(mp4, fps=8) as w:
                for f in frames:
                    w.append_data(f)
            if self.verbose:
                print(f"[eval video] saved {mp4} (frames={len(frames)})")
        except Exception as e:
            if self.verbose:
                print(f"[eval video] MP4 save failed: {e}; falling back to GIF.")
            save_gif(frames, stem)

        if self._cv2 is not None:
            self._cv2.waitKey(1)
        return True

# -------------------
# train / resume
# -------------------
def main():
    # 🔧 declare globals before any usage in this function
    global DIST_SHAPING_W, DIST_SHAPING_CLIP, USE_HUNGARIAN
    global NOOP_PENALTY, DEADLOCK_PENALTY, EARLY_TERM_ON_DEADLOCK
    global STAGNATION_STEPS, STAGNATION_PENALTY, MIN_BOX_GOAL_DIST

    p = argparse.ArgumentParser()
    p.add_argument("--total-steps", type=int, default=500_000)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--n-envs", type=int, default=4)  # used when not recording
    p.add_argument("--save-every", type=int, default=50_000)
    p.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    p.add_argument("--norm-path",  type=str, default=DEFAULT_NORM_PATH)
    p.add_argument("--tb-dir", type=str, default=DEFAULT_TB_DIR)

    # training video options
    p.add_argument("--record-training", action="store_true",
                   help="Record training episodes (full). Forces n_envs=1 & DummyVecEnv.")
    p.add_argument("--record-k", type=int, default=1,
                   help="Record every k-th episode (1=every episode, 2=every 2nd, ...).")
    p.add_argument("--video-dir", type=str, default=DEFAULT_TRAIN_VID_DIR)
    p.add_argument("--video-cap", type=int, default=0,
                   help="Soft cap on frames per episode video (0=off).")
    p.add_argument("--show-live", action="store_true",
                   help="Show live OpenCV window while recording training.")

    # evaluation video options
    p.add_argument("--eval-every", type=int, default=50000,
                   help="Save an evaluation video every N training steps (set <=0 to disable).")
    p.add_argument("--eval-frames", type=int, default=300,
                   help="Max frames per eval video.")
    p.add_argument("--eval-deterministic", action="store_true",
                   help="Use deterministic actions for eval video.")
    p.add_argument("--eval-video-dir", type=str, default=DEFAULT_EVAL_VID_DIR)
    p.add_argument("--eval-show-live", action="store_true",
                   help="Show live OpenCV window for eval videos.")

    # override shaping/penalties from CLI (optional)
    p.add_argument("--dist-w", type=float, default=DIST_SHAPING_W)
    p.add_argument("--dist-clip", type=float, default=DIST_SHAPING_CLIP)
    p.add_argument("--hungarian", action="store_true", default=USE_HUNGARIAN)
    p.add_argument("--noop-pen", type=float, default=NOOP_PENALTY)
    p.add_argument("--deadlock-pen", type=float, default=DEADLOCK_PENALTY)
    p.add_argument("--deadlock-term", action="store_true", default=EARLY_TERM_ON_DEADLOCK)
    p.add_argument("--stagnation", type=int, default=STAGNATION_STEPS)
    p.add_argument("--stagnation-pen", type=float, default=STAGNATION_PENALTY)
    p.add_argument("--min-boxgoal", type=int, default=MIN_BOX_GOAL_DIST)

    args = p.parse_args()

    # apply CLI overrides (module-level globals power make_base_env)
    DIST_SHAPING_W = args.dist_w
    DIST_SHAPING_CLIP = args.dist_clip
    USE_HUNGARIAN = args.hungarian
    NOOP_PENALTY = args.noop_pen
    DEADLOCK_PENALTY = args.deadlock_pen
    EARLY_TERM_ON_DEADLOCK = args.deadlock_term
    STAGNATION_STEPS = args.stagnation
    STAGNATION_PENALTY = args.stagnation_pen
    MIN_BOX_GOAL_DIST = args.min_boxgoal

    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.norm_path), exist_ok=True)
    os.makedirs(args.video_dir, exist_ok=True)
    os.makedirs(args.eval_video_dir, exist_ok=True)

    # force random each run (no fixed seed)
    seed = None

    # choose vec env class
    if args.record_training:
        vec_cls = DummyVecEnv; n_envs = 1; n_stack = 2
        print("[recording] ON → DummyVecEnv, n_envs=1 (renderable)")
    else:
        vec_cls = SubprocVecEnv; n_envs = args.n_envs; n_stack = 2
        print(f"[recording] OFF → {vec_cls.__name__}, n_envs={n_envs} (faster)")

    # build env + model
    if args.resume and os.path.exists(args.model_path):
        train_env = build_train_env(n_envs, seed, vec_cls, load_norm_from=args.norm_path, n_stack=n_stack)
        model = PPO.load(args.model_path, env=train_env, device=DEVICE)
        reset_flag = False
        print(f"[resume] loaded {args.model_path}")
    else:
        train_env = build_train_env(n_envs, seed, vec_cls, load_norm_from=None, n_stack=n_stack)
        policy_kwargs = dict(
            features_extractor_class=BalancedGridCNN,
            features_extractor_kwargs=dict(features_dim=128),
        )
        model = PPO(
            "CnnPolicy", train_env, device=DEVICE, verbose=1, seed=seed,
            tensorboard_log=args.tb_dir,
            policy_kwargs=policy_kwargs,
            n_steps=256, batch_size=1024, learning_rate=3e-4,
            n_epochs=4, gamma=0.99, gae_lambda=0.95,
            ent_coef=0.02, clip_range=0.2, vf_coef=0.5, max_grad_norm=0.5,
        )
        reset_flag = True
        print("[train] starting from scratch")

    callbacks = [SaveEveryNSteps(args.save_every, args.model_path, args.norm_path, verbose=1)]
    if args.record_training:
        callbacks.append(
            TrainingEpisodeRecorderCallback(
                video_dir=args.video_dir,
                video_len_cap=args.video_cap,
                record_k=args.record_k,
                show_live=args.show_live,
                verbose=1
            )
        )

    # periodic evaluation video (works with SubprocVecEnv training)
    if args.eval_every and args.eval_every > 0:
        eval_env = build_eval_env(args.norm_path if args.resume else None, seed, n_stack=n_stack)
        callbacks.append(
            PeriodicEvalAndVideoCallback(
                eval_env=eval_env,
                every_steps=args.eval_every,
                video_dir=args.eval_video_dir,
                video_len=args.eval_frames,
                deterministic=args.eval_deterministic,
                show_live=args.eval_show_live,
                verbose=1
            )
        )

    # force TensorBoard to log to a specific folder 
    log_dir = os.path.join(args.tb_dir, "PPO_15")
    os.makedirs(log_dir, exist_ok=True)
    new_logger = configure(log_dir, ["stdout", "tensorboard", "csv"])
    model.set_logger(new_logger)


    model.learn(total_timesteps=args.total_steps, callback=callbacks, reset_num_timesteps=reset_flag)

    # save model + normalization
    model.save(args.model_path)
    try:
        train_env.save(args.norm_path)
    except Exception:
        pass
    print("[done] saved model + normalization.")

if __name__ == "__main__":
    main()
