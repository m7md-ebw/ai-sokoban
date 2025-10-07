import argparse
import os
import gymnasium as gym
import numpy as np
from collections import deque
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from gymnasium.spaces import Discrete, Box
import imageio

try:
    from sb3_contrib import RecurrentPPO
except ImportError:
    RecurrentPPO = None

# Sokoban params
SOKOBAN_PARAMS = {
    "Sokoban-small-v0": {"dim_room": (7,7), "num_boxes": 1, "max_steps": 50},
    "Sokoban-small-v1": {"dim_room": (7,7), "num_boxes": 3, "max_steps": 50},
    "Sokoban-v0": {"dim_room": (10,10), "num_boxes": 3, "max_steps": 100},
}

# Gym wrapper
class GymnasiumSokobanWrapper(gym.Env):
    def __init__(self, env_id):
        from gym_sokoban.envs import SokobanEnv
        params = SOKOBAN_PARAMS[env_id]
        self.env = SokobanEnv(dim_room=params["dim_room"],
                              num_boxes=params["num_boxes"],
                              max_steps=params["max_steps"])
        self.observation_space = Box(low=0, high=255,
                                     shape=self.env.observation_space.shape,
                                     dtype=np.uint8)
        self.action_space = Discrete(self.env.action_space.n)

    def reset(self, **kwargs):
        obs = self.env.reset()
        return obs, {}

    def step(self, action):
        res = self.env.step(action)
        if len(res) == 4:
            obs, reward, done, info = res
            truncated = False
        elif len(res) == 5:
            obs, reward, terminated, truncated, info = res
            done = terminated or truncated
        else:
            raise RuntimeError("Unexpected step return from Sokoban env")
        return obs, reward, done, truncated, info

    def render(self, mode="human"):
        return self.env.render(mode=mode)

    def close(self):
        self.env.close()

# Env factory
def make_env(env_id, seed=0):
    def _init():
        if env_id.startswith("Sokoban"):
            env = GymnasiumSokobanWrapper(env_id)
        else:
            env = gym.make(env_id)
        env = Monitor(env)
        return env
    return _init

def choose_policy(env, use_recurrent=False):
    if isinstance(env.observation_space, Box) and len(env.observation_space.shape) == 3:
        return "CnnLstmPolicy" if use_recurrent else "CnnPolicy"
    else:
        return "MlpLstmPolicy" if use_recurrent else "MlpPolicy"

# Video callback
class PeriodicEvalVideoCallback(BaseCallback):
    def __init__(self, eval_env, out_dir="videos", every_steps=50_000, video_len=300, deterministic=True, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.out_dir = out_dir
        self.every_steps = int(every_steps)
        self.video_len = int(video_len)
        self.deterministic = bool(deterministic)
        self._last = 0
        os.makedirs(self.out_dir, exist_ok=True)

    def _unwrap_base(self, env):
        raw = env.envs[0] if hasattr(env, "envs") else env
        base = raw
        while hasattr(base, "env"):
            base = base.env
        return base

    def _safe_render(self, base):
        try:
            rgb = base.render(mode="rgb_array")
            return np.asarray(rgb, dtype=np.uint8) if rgb is not None else None
        except Exception:
            return None

    def _on_step(self) -> bool:
        if (self.model.num_timesteps - self._last) < self.every_steps:
            return True
        self._last = self.model.num_timesteps

        obs = self.eval_env.reset()
        base = self._unwrap_base(self.eval_env)

        frames, ep_reward, steps = [], 0.0, 0
        f0 = self._safe_render(base)
        if f0 is not None:
            frames.append(f0)

        done = False
        while steps < self.video_len:
            action, _ = self.model.predict(obs, deterministic=self.deterministic)
            obs, reward, done_arr, info = self.eval_env.step(action)
            ep_reward += float(reward[0]) if isinstance(reward, (np.ndarray, list)) else float(reward)
            steps += 1

            rgb = self._safe_render(base)
            if rgb is not None:
                frames.append(rgb)

            done = bool(done_arr[0]) if isinstance(done_arr, (np.ndarray, list)) else bool(done_arr)
            if done:
                obs = self.eval_env.reset()

        stem = f"eval_ts_{self._last}_R{ep_reward:.1f}"
        mp4 = os.path.join(self.out_dir, f"{stem}.mp4")

        if len(frames) < 2:
            gif = os.path.join(self.out_dir, f"{stem}.gif")
            try:
                imageio.mimsave(gif, frames if frames else [np.zeros((112,112,3),dtype=np.uint8)], fps=8)
                if self.verbose:
                    print(f"[video] saved {gif}")
            except Exception as e:
                if self.verbose:
                    print(f"[video] failed to save GIF: {e}")
            return True

        try:
            with imageio.get_writer(mp4, fps=8) as w:
                for f in frames:
                    w.append_data(f)
            if self.verbose:
                print(f"[video] saved {mp4} ({len(frames)} frames, return={ep_reward:.1f})")
        except Exception as e:
            gif = os.path.join(self.out_dir, f"{stem}.gif")
            imageio.mimsave(gif, frames, fps=8)
            if self.verbose:
                print(f"[video] MP4 failed ({e}), saved GIF {gif}")
        return True

# Training
def train(env_id="Sokoban-small-v0",
          algo="ppo",
          total_timesteps=int(2e6),
          logdir="logs",
          seed=0,
          n_envs=8,
          continue_from=None,
          use_subproc=True,
          save_video=False):
    os.makedirs(logdir, exist_ok=True)

    env_fns = [make_env(env_id, seed+i) for i in range(n_envs)]
    if use_subproc:
        vec_env = SubprocVecEnv(env_fns)
    else:
        vec_env = DummyVecEnv(env_fns)

    vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True)
    vec_env = VecMonitor(vec_env, filename=os.path.join(logdir, "monitor.csv"))

    sample_env = make_env(env_id)()
    use_recurrent = algo.lower() in ("rppo", "ppo_lstm", "recurrentppo")
    policy = choose_policy(sample_env, use_recurrent)

    if continue_from and os.path.isfile(continue_from):
        if algo.lower() == "ppo":
            model = PPO.load(continue_from, env=vec_env, device="auto", tensorboard_log="logs/PPO_59")
        else:
            if RecurrentPPO is None:
                raise RuntimeError("sb3_contrib.RecurrentPPO not installed.")
            model = RecurrentPPO.load(continue_from, env=vec_env, device="auto")
    else:
        if algo.lower() == "ppo":
            model = PPO(policy, vec_env, verbose=1, tensorboard_log=logdir)
        else:
            if RecurrentPPO is None:
                raise RuntimeError("sb3_contrib.RecurrentPPO not installed.")
            model = RecurrentPPO(policy, vec_env, verbose=1, tensorboard_log=logdir)

    callbacks = []
    checkpoint_callback = CheckpointCallback(save_freq=250_000 // n_envs,
                                             save_path=logdir,
                                             name_prefix=f"{env_id}_{algo}_ckpt")
    callbacks.append(checkpoint_callback)

    if save_video:
        eval_env = DummyVecEnv([make_env(env_id, seed+1000)])
        eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=False)
        video_callback = PeriodicEvalVideoCallback(
            eval_env,
            out_dir=os.path.join(logdir,"videos"),
            every_steps=50_000
        )
        callbacks.append(video_callback)

    from stable_baselines3.common.callbacks import CallbackList
    callbacks = CallbackList(callbacks)

    try:
        model.learn(total_timesteps=total_timesteps, callback=callbacks)
    except KeyboardInterrupt:
        pass
    finally:
        final_path = os.path.join(logdir, f"{env_id}_{algo}_final.zip")
        model.save(final_path)
        vec_env.close()
        if save_video:
            eval_env.close()

    return model

# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="Sokoban-small-v0")
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo","rppo"])
    parser.add_argument("--timesteps", type=int, default=int(1e6))
    parser.add_argument("--logdir", type=str, default="logs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_envs", type=int, default=8)
    parser.add_argument("--continue_from", type=str, default=None)
    parser.add_argument("--use_subproc", action="store_true")
    parser.add_argument("--save_video", action="store_true")
    args = parser.parse_args()

    train(env_id=args.env,
          algo=args.algo,
          total_timesteps=args.timesteps,
          logdir=args.logdir,
          seed=args.seed,
          n_envs=args.n_envs,
          continue_from=args.continue_from,
          use_subproc=args.use_subproc,
          save_video=args.save_video)