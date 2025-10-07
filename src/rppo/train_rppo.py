import argparse
import os
import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize, VecMonitor, VecTransposeImage
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack
from gymnasium.spaces import Discrete, Box
import imageio

try:
    from sb3_contrib import RecurrentPPO
except ImportError:
    raise RuntimeError("sb3_contrib.RecurrentPPO not installed")

# Sokoban params
SOKOBAN_PARAMS = {
    "Sokoban-small-v0": {"dim_room": (7, 7), "num_boxes": 2, "max_steps": 50},
    "Sokoban-small-v1": {"dim_room": (7, 7), "num_boxes": 3, "max_steps": 50},
    "Sokoban-v0": {"dim_room": (10, 10), "num_boxes": 3, "max_steps": 100},
}

# Utils
def linear_schedule(initial_value: float):
    def func(progress_remaining: float) -> float:
        return float(progress_remaining) * initial_value
    return func

def is_image_observation(env: gym.Env) -> bool:
    space = env.observation_space
    return isinstance(space, Box) and len(space.shape) == 3 and space.dtype == np.uint8

def choose_policy(env: gym.Env) -> str:
    return "CnnLstmPolicy" if is_image_observation(env) else "MlpLstmPolicy"

# Gym wrapper
class GymnasiumSokobanWrapper(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}
    def __init__(self, env_id: str, render_mode: str = None):
        from gym_sokoban.envs import SokobanEnv
        params = SOKOBAN_PARAMS[env_id]
        self.env = SokobanEnv(
            dim_room=params["dim_room"],
            num_boxes=params["num_boxes"],
            max_steps=params["max_steps"],
        )
        self.render_mode = render_mode
        self.observation_space = Box(
            low=0, high=255, shape=self.env.observation_space.shape, dtype=np.uint8
        )
        self.action_space = Discrete(self.env.action_space.n)

    def reset(self, *, seed: int = None, options: dict = None):
        if seed is not None:
            try:
                self.env.seed(seed)
            except Exception:
                pass
            np.random.seed(seed)
        obs = self.env.reset()
        return obs, {}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        terminated = bool(done)
        truncated = bool(info.get("TimeLimit.truncated", False))
        return obs, reward, terminated, truncated, info

    def render(self):
        return self.env.render(mode=self.render_mode or "human")

    def close(self):
        self.env.close()

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
        lstm_states = None
        episode_starts = np.ones((1,), dtype=np.bool_)

        frames, ep_reward, steps = [], 0.0, 0
        f0 = self._safe_render(base)
        if f0 is not None:
            frames.append(f0)

        done = False
        while steps < self.video_len:
            action, lstm_states = self.model.predict(
                obs, state=lstm_states, episode_start=episode_starts, deterministic=self.deterministic
            )
            obs, reward, done_arr, info = self.eval_env.step(action)
            ep_reward += float(reward[0]) if isinstance(reward, (np.ndarray, list)) else float(reward)
            steps += 1
            episode_starts = done_arr

            rgb = self._safe_render(base)
            if rgb is not None:
                frames.append(rgb)

            done = bool(done_arr[0]) if isinstance(done_arr, (np.ndarray, list)) else bool(done_arr)
            if done:
                obs = self.eval_env.reset()
                lstm_states = None
                episode_starts = np.ones((1,), dtype=np.bool_)

        stem = f"eval_ts_{self._last}_R{ep_reward:.1f}"
        mp4 = os.path.join(self.out_dir, f"{stem}.mp4")

        if len(frames) < 2:
            gif = os.path.join(self.out_dir, f"{stem}.gif")
            try:
                imageio.mimsave(gif, frames if frames else [np.zeros((112,112,3), dtype=np.uint8)], fps=8)
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

# Save VecNormalize callback
class SaveVecNormCallback(BaseCallback):
    def __init__(self, save_path: str, vecnorm: VecNormalize, verbose: int = 0):
        super().__init__(verbose)
        self.save_path = save_path
        self.vecnorm = vecnorm
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if self.vecnorm is not None:
            path = os.path.join(self.save_path, "vecnormalize.pkl")
            self.vecnorm.save(path)
            if self.verbose:
                print(f"[VecNormalize] saved to {path}")

# Env factory
def make_env(env_id: str, seed: int = 0):
    def _init():
        if env_id.startswith("Sokoban"):
            env = GymnasiumSokobanWrapper(env_id)
        else:
            env = gym.make(env_id)
        try:
            env.reset(seed=seed)
        except TypeError:
            pass
        return env
    return _init

# Training
def train(
    env_id: str = "Sokoban-small-v0",
    total_timesteps: int = int(3e6),
    logdir: str = "logs",
    seed: int = 0,
    eval_env_id: str = None,
    eval_freq: int = 50_000,
    save_freq: int = 250_000,
    n_envs: int = 16,
    frame_stack: int = 4,
    subproc: bool = True,
    device: str = "auto",
):
    os.makedirs(logdir, exist_ok=True)

    # Train env
    env_fns = [make_env(env_id, seed + i) for i in range(n_envs)]
    vec = SubprocVecEnv(env_fns) if subproc and n_envs > 1 else DummyVecEnv(env_fns)
    sample_env = make_env(env_id, seed)()
    if is_image_observation(sample_env):
        vec = VecTransposeImage(vec)
        vec = VecFrameStack(vec, n_stack=frame_stack, channels_order="first")
    vec = VecNormalize(vec, norm_obs=False, norm_reward=True, clip_reward=10.0)
    vec = VecMonitor(vec, filename=os.path.join(logdir, "monitor.csv"))

    # Model
    policy = choose_policy(sample_env)
    model = RecurrentPPO(
        policy,
        vec,
        verbose=1,
        n_steps=128,
        batch_size=256,
        n_epochs=4,
        learning_rate=linear_schedule(3e-4),
        ent_coef=0.005,
        gae_lambda=0.95,
        vf_coef=0.5,
        clip_range=linear_schedule(0.2),
        target_kl=0.02,
        max_grad_norm=0.5,
        seed=seed,
        tensorboard_log=logdir,
        device=device,
        policy_kwargs=dict(
            ortho_init=True,
            net_arch=dict(pi=[512, 512], vf=[512, 512]),
            lstm_hidden_size=512,
        ),
    )

    # Callbacks
    callbacks = []
    ckpt = CheckpointCallback(
        save_freq=max(1, save_freq // max(1, n_envs)),
        save_path=logdir,
        name_prefix=f"{env_id}_rppo_ckpt",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    callbacks.append(ckpt)
    callbacks.append(SaveVecNormCallback(save_path=logdir, vecnorm=vec, verbose=1))

    if eval_env_id is not None:
        eval_env_fns = [make_env(eval_env_id, seed + 10_000 + i) for i in range(1)]
        eval_vec = DummyVecEnv(eval_env_fns)
        if is_image_observation(sample_env):
            eval_vec = VecTransposeImage(eval_vec)
            eval_vec = VecFrameStack(eval_vec, n_stack=frame_stack, channels_order="first")
        eval_vec = VecNormalize(eval_vec, norm_obs=False, norm_reward=False)
        eval_vec = VecMonitor(eval_vec, filename=os.path.join(logdir, "eval_monitor.csv"))
        eval_callback = EvalCallback(
            eval_env=eval_vec,
            best_model_save_path=os.path.join(logdir, "best_model"),
            log_path=os.path.join(logdir, "eval_logs"),
            eval_freq=eval_freq // max(1, n_envs),
            deterministic=True,
            render=False,
            n_eval_episodes=10,
        )
        callbacks.append(eval_callback)

    callback_list = CallbackList(callbacks)

    # Training loop
    try:
        model.learn(total_timesteps=total_timesteps, callback=callback_list, progress_bar=True)
    except KeyboardInterrupt:
        pass
    finally:
        final_model_path = os.path.join(logdir, f"{env_id}_rppo_final")
        model.save(final_model_path)
        vec.save(os.path.join(logdir, "vecnormalize.pkl"))
        vec.close()
        if eval_env_id is not None:
            eval_vec.close()

    return model

# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="Sokoban-small-v0")
    parser.add_argument("--timesteps", type=int, default=int(3e6))
    parser.add_argument("--logdir", type=str, default="logs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval_env", type=str, default=None)
    parser.add_argument("--eval_freq", type=int, default=50_000)
    parser.add_argument("--save_freq", type=int, default=250_000)
    parser.add_argument("--n_envs", type=int, default=16)
    parser.add_argument("--frame_stack", type=int, default=4)
    parser.add_argument("--no_subproc", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    train(
        env_id=args.env,
        total_timesteps=args.timesteps,
        logdir=args.logdir,
        seed=args.seed,
        eval_env_id=args.eval_env,
        eval_freq=args.eval_freq,
        save_freq=args.save_freq,
        n_envs=args.n_envs,
        frame_stack=args.frame_stack,
        subproc=not args.no_subproc,
        device=args.device,
    )
