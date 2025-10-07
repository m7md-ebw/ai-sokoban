import os
import argparse
import numpy as np
import torch
import gym
import gym_sokoban
import imageio
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    DummyVecEnv, SubprocVecEnv, VecTransposeImage, VecFrameStack, VecNormalize
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.logger import configure

# Wrappers
class PrintObsShapeOnce(gym.Wrapper):
    def __init__(self, env, tag="env"):
        super().__init__(env)
        self._printed = False
        self._tag = tag

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        if not self._printed:
            print(f"[{self._tag}] obs_space={self.observation_space.shape}")
            self._printed = True
        return obs

    def step(self, action):
        return self.env.step(action)

# Env constructor
def _construct_sokoban(dim_room, num_boxes, max_steps):
    SokobanEnv = None
    try:
        from gym_sokoban.envs import SokobanEnv as _SokobanEnv
        SokobanEnv = _SokobanEnv
    except Exception:
        SokobanEnv = None

    if SokobanEnv is not None:
        for kwargs in (
            dict(dim_room=dim_room, num_boxes=num_boxes, max_steps=max_steps),
            dict(dim_room=dim_room, num_boxes=num_boxes, max_steps_per_episode=max_steps),
            dict(dim_room=dim_room, num_boxes=num_boxes),
            dict(),
        ):
            try:
                return SokobanEnv(**kwargs)
            except TypeError:
                continue
            except Exception:
                continue

    for env_id in ("Sokoban-v0", "Sokoban-v1", "Sokoban-v2"):
        try:
            return gym.make(env_id)
        except Exception:
            pass

    for env_id in ("SokobanPush-v0", "SokobanSmall-v0"):
        try:
            return gym.make(env_id)
        except Exception:
            pass

    raise RuntimeError("Could not construct a Sokoban env.")

# Env factory
def make_base_env(dim_room=(7, 7), num_boxes=1, max_steps=120, step_penalty=0.0, print_tag=None):
    env = _construct_sokoban(dim_room=dim_room, num_boxes=num_boxes, max_steps=max_steps)

    if step_penalty > 0:
        from gym import RewardWrapper
        class StepPenalty(RewardWrapper):
            def __init__(self, env, p): super().__init__(env); self.p = float(p)
            def reward(self, r): return r - self.p
        env = StepPenalty(env, step_penalty)

    if print_tag is not None:
        env = PrintObsShapeOnce(env, tag=print_tag)
    return env

def build_vec_env(n_envs, n_stack, vec_cls, seed=None, norm_reward=False,
                  dim_room=(7,7), num_boxes=1, max_steps=120, step_penalty=0.0, tag="train"):
    venv = make_vec_env(
        lambda: make_base_env(dim_room, num_boxes, max_steps, step_penalty, print_tag="base"),
        n_envs=n_envs, seed=seed, vec_env_cls=vec_cls,
    )
    venv = VecTransposeImage(venv)
    venv = VecFrameStack(venv, n_stack=n_stack)
    venv = VecNormalize(venv, norm_obs=False, norm_reward=norm_reward)
    venv.training = (tag == "train")
    return venv

# Utilities
def assert_obs_compat(model, env, where):
    ms = tuple(model.observation_space.shape)
    es = tuple(env.observation_space.shape)
    if ms != es:
        raise RuntimeError(
            f"[{where}] Observation shape mismatch: model expects {ms} but env provides {es}."
        )

# Callbacks
class PeriodicEvalVideoCallback(BaseCallback):
    def __init__(self, eval_env, out_dir="videos", every_steps=50_000, video_len=300, deterministic=True, verbose=0):
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
            ep_reward += float(reward[0])
            steps += 1

            rgb = self._safe_render(base)
            if rgb is not None:
                frames.append(rgb)

            done = bool(done_arr[0])
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
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--total-timesteps", type=int, default=300_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--n-steps", type=int, default=1024)
    p.add_argument("--n-stack", type=int, default=2)
    p.add_argument("--ent-coef", type=float, default=0.005)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--save-every", type=int, default=50_000)
    p.add_argument("--eval-every", type=int, default=50_000)
    p.add_argument("--eval-episodes", type=int, default=5)
    p.add_argument("--tb-dir", type=str, default="runs/ppo_tb")
    p.add_argument("--model-path", type=str, default="checkpoints/ppo.zip")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--room-h", type=int, default=7)
    p.add_argument("--room-w", type=int, default=7)
    p.add_argument("--num-boxes", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=120)
    p.add_argument("--step-penalty", type=float, default=0.0)
    p.add_argument("--record-now", dest="record_now", action="store_true")
    p.add_argument("--record-frames", type=int, default=300)
    args = p.parse_args()

    dim_room = (args.room_h, args.room_w)
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Quick record mode
    if args.record_now:
        eval_env = build_vec_env(
            n_envs=1,
            n_stack=args.n_stack,
            vec_cls=DummyVecEnv,
            seed=None,
            norm_reward=False,
            dim_room=dim_room,
            num_boxes=args.num_boxes,
            max_steps=args.max_steps,
            step_penalty=args.step_penalty,
            tag="eval",
        )
        if os.path.exists(args.model_path):
            model = PPO.load(args.model_path, env=eval_env, device=device)
        else:
            model = PPO("CnnPolicy", eval_env, device=device, n_steps=8, batch_size=8)
        video_cb = PeriodicEvalVideoCallback(
            eval_env=eval_env,
            out_dir="videos",
            every_steps=1,
            video_len=args.record_frames,
            deterministic=True,
            verbose=1,
        )
        video_cb.model = model
        video_cb._on_step()
        return

    # Train env
    train_env = build_vec_env(
        n_envs=args.n_envs,
        n_stack=args.n_stack,
        vec_cls=SubprocVecEnv,
        seed=None,
        norm_reward=False,
        dim_room=dim_room,
        num_boxes=args.num_boxes,
        max_steps=args.max_steps,
        step_penalty=args.step_penalty,
        tag="train",
    )

    # Model
    if args.resume and os.path.exists(args.model_path):
        model = PPO.load(args.model_path, env=train_env, device=device)
        reset_flag = False
    else:
        model = PPO(
            "CnnPolicy",
            train_env,
            device=device,
            verbose=1,
            n_steps=args.n_steps,
            batch_size=256,
            learning_rate=args.lr,
            n_epochs=4,
            gamma=0.995,
            gae_lambda=0.95,
            ent_coef=args.ent_coef,
            clip_range=0.2,
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log=args.tb_dir,
        )
        reset_flag = True

    assert_obs_compat(model, train_env, where="train-start")

    # Eval env
    eval_env = build_vec_env(
        n_envs=1,
        n_stack=args.n_stack,
        vec_cls=DummyVecEnv,
        seed=None,
        norm_reward=False,
        dim_room=dim_room,
        num_boxes=args.num_boxes,
        max_steps=args.max_steps,
        step_penalty=args.step_penalty,
        tag="eval",
    )
    assert_obs_compat(model, eval_env, where="eval-build")

    # Callbacks
    video_cb = PeriodicEvalVideoCallback(
        eval_env=eval_env,
        out_dir="videos",
        every_steps=20_000,
        video_len=300,
        deterministic=True,
        verbose=1,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=max(1, args.save_every // max(args.n_envs, 1)),
        save_path=os.path.dirname(args.model_path),
        name_prefix=os.path.splitext(os.path.basename(args.model_path))[0],
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.dirname(args.model_path),
        log_path=os.path.join(args.tb_dir, "eval"),
        eval_freq=max(1, args.eval_every // max(args.n_envs, 1)),
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
    )

    # Logging
    run_dir = os.path.join(args.tb_dir, "PPO_run")
    os.makedirs(run_dir, exist_ok=True)
    model.set_logger(configure(run_dir, ["stdout", "tensorboard", "csv"]))

    # Training
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[ckpt_cb, eval_cb, video_cb],
        reset_num_timesteps=reset_flag,
        log_interval=1,
    )

    model.save(args.model_path)

if __name__ == "__main__":
    main()