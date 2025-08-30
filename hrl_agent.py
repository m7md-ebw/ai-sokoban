# hrl_sokoban.py
import os
import numpy as np
import torch
import gym
import gym_sokoban
from collections import deque

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.logger import configure

# ---------------- Environment Wrappers ----------------
class NoOpPenaltyWrapper(gym.Wrapper):
    def __init__(self, env, penalty=0.02):
        super().__init__(env)
        self.penalty = penalty
    def step(self, a):
        pos0 = self._get_pos()
        obs, r, done, info = self.env.step(a)
        pos1 = self._get_pos()
        if pos0 == pos1:
            r -= self.penalty
            info["noop_penalty"] = self.penalty
        return obs, r, done, info
    def _get_pos(self):
        try:
            g = self.env.unwrapped.room_state
            ys, xs = np.where((g==3)|(g==6))
            return (int(ys[0]), int(xs[0])) if len(ys) else None
        except Exception:
            return None

class BackForthPenaltyWrapper(gym.Wrapper):
    def __init__(self, env, penalty=0.03, max_history=8):
        super().__init__(env)
        self.penalty = penalty
        self.hist = deque(maxlen=max_history)
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.hist.clear()
        pos = self._get_pos()
        if pos is not None: self.hist.append(pos)
        return obs
    def step(self, a):
        obs, r, done, info = self.env.step(a)
        pos = self._get_pos()
        if pos is not None:
            self.hist.append(pos)
            if len(self.hist) >= 4:
                a,b,c,d = self.hist[-4:]
                if a==c and b==d and a!=b:
                    r -= self.penalty
                    info["backforth_penalty"] = self.penalty
        return obs, r, done, info
    def _get_pos(self):
        try:
            g = self.env.unwrapped.room_state
            ys, xs = np.where((g==3)|(g==6))
            return (int(ys[0]), int(xs[0])) if len(ys) else None
        except Exception: return None

# ---------------- Environment Constructor ----------------
def make_sokoban_env(dim_room=(8,8), num_boxes=2, max_steps=160):
    env = gym.make("Sokoban-v0")
    env = NoOpPenaltyWrapper(env, penalty=0.05)
    env = BackForthPenaltyWrapper(env, penalty=0.03)
    return env

def build_vec_env(n_envs, n_stack, dim_room, num_boxes):
    venv = DummyVecEnv([lambda: make_sokoban_env(dim_room, num_boxes) for _ in range(n_envs)])
    venv = VecTransposeImage(venv)
    venv = VecFrameStack(venv, n_stack)
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False)
    return venv

# ---------------- Callbacks ----------------
class EvalVideoCallback(BaseCallback):
    def __init__(self, eval_env, out_dir="videos", every_steps=50_000, video_len=300, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.out_dir = out_dir
        self.every_steps = every_steps
        self.video_len = video_len
        os.makedirs(out_dir, exist_ok=True)
        self._last = 0
    def _on_step(self):
        if (self.model.num_timesteps - self._last) < self.every_steps:
            return True
        self._last = self.model.num_timesteps
        obs = self.eval_env.reset()
        frames = []
        done = False
        step_count = 0
        while step_count < self.video_len:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done_arr, info = self.eval_env.step(action)
            step_count += 1
            frame = self.eval_env.render(mode="rgb_array")
            if frame is not None:
                frames.append(frame)
            if done_arr[0]:
                obs = self.eval_env.reset()
        import imageio
        video_path = os.path.join(self.out_dir, f"eval_{self.model.num_timesteps}.mp4")
        try:
            with imageio.get_writer(video_path, fps=8) as w:
                for f in frames:
                    w.append_data(f)
            if self.verbose: print(f"[video] saved {video_path}")
        except Exception as e:
            if self.verbose: print(f"[video] failed: {e}")
        return True

# ---------------- HRL Agent ----------------
class HRLAgent:
    def __init__(self, manager_env, worker_env, manager_timesteps=2000, worker_timesteps=128):
        self.manager_env = manager_env
        self.worker_env = worker_env
        self.manager_timesteps = manager_timesteps
        self.worker_timesteps = worker_timesteps
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._build_models()
    def _build_models(self):
        # Manager chooses sub-goals (MLP)
        self.manager_model = PPO(
            "MlpPolicy", self.manager_env,
            verbose=1, device=self.device,
            n_steps=self.manager_timesteps,
            tensorboard_log="runs/hrl/manager"
        )
        # Worker executes primitive actions (CNN)
        self.worker_model = PPO(
            "CnnPolicy", self.worker_env,
            verbose=1, device=self.device,
            n_steps=self.worker_timesteps,
            tensorboard_log="runs/hrl/worker"
        )
    def train(self, total_timesteps=500_000):
        for i in range(total_timesteps // self.manager_timesteps):
            # Manager chooses sub-goals
            obs_m = self.manager_env.reset()
            subgoal, _ = self.manager_model.predict(obs_m, deterministic=True)
            # Worker executes until subgoal done or worker_timesteps exhausted
            obs_w = self.worker_env.reset()
            for _ in range(self.worker_timesteps):
                action, _ = self.worker_model.predict(obs_w, deterministic=False)
                obs_w, reward, done, info = self.worker_env.step(action)
                if done[0]: break
            # Optionally train manager on cumulative reward
            self.manager_model.learn(total_timesteps=self.manager_timesteps, reset_num_timesteps=False)
            self.worker_model.learn(total_timesteps=self.worker_timesteps, reset_num_timesteps=False)
        self.manager_model.save("hrl_manager.zip")
        self.worker_model.save("hrl_worker.zip")
        print("[HRL] Training complete. Models saved.")

# ---------------- Main ----------------
def main():
    # Adjustable room & boxes
    room_dim = (8,8)
    num_boxes = 2

    # Vector envs
    manager_env = build_vec_env(n_envs=4, n_stack=1, dim_room=room_dim, num_boxes=num_boxes)
    worker_env  = build_vec_env(n_envs=4, n_stack=2, dim_room=room_dim, num_boxes=num_boxes)

    # Train HRL
    agent = HRLAgent(manager_env, worker_env)
    agent.train(total_timesteps=500_000)

if __name__ == "__main__":
    main()
