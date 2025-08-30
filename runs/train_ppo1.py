# train_ppo.py
import os
import argparse
import numpy as np
import torch
import gym
import gym_sokoban  # pip install gym-sokoban

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    DummyVecEnv, SubprocVecEnv, VecTransposeImage, VecFrameStack, VecNormalize
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.logger import configure


# -------------------
# Print base env obs shape once (safe only on single env)
# -------------------
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


# -------------------
# Robust Sokoban constructor (handles different gym_sokoban versions)
# -------------------
def _construct_sokoban(dim_room, num_boxes, max_steps):
    """
    Try several ways to build a Sokoban env so it works across versions/forks.
    """
    # Try importing the class directly and calling with different signatures
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
            dict(),  # defaults only
        ):
            try:
                return SokobanEnv(**kwargs)
            except TypeError:
                continue
            except Exception:
                continue

    # Fall back to registry without kwargs (some builds ignore them)
    for env_id in ("Sokoban-v0", "Sokoban-v1", "Sokoban-v2"):
        try:
            return gym.make(env_id)
        except Exception:
            pass

    # Some forks register different IDs
    for env_id in ("SokobanPush-v0", "SokobanSmall-v0"):
        try:
            return gym.make(env_id)
        except Exception:
            pass

    raise RuntimeError(
        "Could not construct a Sokoban env. Your gym_sokoban build exposes "
        "unexpected constructors/IDs. Run `pip show gym-sokoban` and share the version."
    )


# -------------------
# Base env + tiny optional step penalty
# -------------------
def make_base_env(dim_room=(7, 7), num_boxes=1, max_steps=120, step_penalty=0.0, print_tag=None):
    env = _construct_sokoban(dim_room=dim_room, num_boxes=num_boxes, max_steps=max_steps)

    if step_penalty > 0:
        from gym import RewardWrapper
        class StepPenalty(RewardWrapper):
            def __init__(self, env, p): super().__init__(env); self.p = float(p)
            def reward(self, r): return r - self.p
        env = StepPenalty(env, step_penalty)

    # SAFE: print wrapper only on the single base env (before vectorization)
    if print_tag is not None:
        env = PrintObsShapeOnce(env, tag=print_tag)
    return env


# -------------------
# Build vector env (IDENTICAL pipeline for train & eval)
# -------------------
def build_vec_env(n_envs, n_stack, vec_cls, seed=None, norm_reward=False,
                  dim_room=(7,7), num_boxes=1, max_steps=120, step_penalty=0.0, tag="train"):
    venv = make_vec_env(
        lambda: make_base_env(dim_room, num_boxes, max_steps, step_penalty, print_tag="base"),
        n_envs=n_envs, seed=seed, vec_env_cls=vec_cls,
    )
    # HWC -> CHW
    venv = VecTransposeImage(venv)
    # IMPORTANT: same n_stack for BOTH train & eval
    venv = VecFrameStack(venv, n_stack=n_stack)
    # Keep rewards unnormalized while debugging shape issues
    venv = VecNormalize(venv, norm_obs=False, norm_reward=norm_reward)
    venv.training = (tag == "train")
    # DO NOT wrap VecEnv with a Gym.Wrapper here (that caused your crash)
    return venv


def assert_obs_compat(model, env, where):
    ms = tuple(model.observation_space.shape)
    es = tuple(env.observation_space.shape)
    if ms != es:
        raise RuntimeError(
            f"[{where}] Observation shape mismatch: model expects {ms} but env provides {es}. "
            f"Use the SAME --n-stack and identical preprocessing order for train and eval."
        )


def main():
    p = argparse.ArgumentParser()
    # timesteps & vec config
    p.add_argument("--total-timesteps", type=int, default=300_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--n-steps", type=int, default=1024)
    p.add_argument("--n-stack", type=int, default=2)           # <— must match train & eval
    # PPO hyperparams
    p.add_argument("--ent-coef", type=float, default=0.005)
    p.add_argument("--lr", type=float, default=3e-4)
    # save/eval
    p.add_argument("--save-every", type=int, default=50_000)
    p.add_argument("--eval-every", type=int, default=50_000)
    p.add_argument("--eval-episodes", type=int, default=5)
    p.add_argument("--tb-dir", type=str, default="runs/ppo_tb")
    p.add_argument("--model-path", type=str, default="checkpoints/ppo.zip")
    p.add_argument("--resume", action="store_true")
    # env size/options (keep identical for train & eval)
    p.add_argument("--room-h", type=int, default=7)
    p.add_argument("--room-w", type=int, default=7)
    p.add_argument("--num-boxes", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=120)
    p.add_argument("--step-penalty", type=float, default=0.0)
    args = p.parse_args()

    dim_room = (args.room_h, args.room_w)
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)

    # ---------------- build TRAIN env ----------------
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

    # ---------------- create / load model ----------------
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.resume and os.path.exists(args.model_path):
        model = PPO.load(args.model_path, env=train_env, device=device)
        print(f"[resume] loaded {args.model_path}")
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

    # Fail fast if shapes diverge
    assert_obs_compat(model, train_env, where="train-start")

    # ---------------- build EVAL env (Dummy) with SAME n_stack ----------------
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

    # ---------------- callbacks ----------------
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

    # tensorboard logger folder
    run_dir = os.path.join(args.tb_dir, "PPO_run")
    os.makedirs(run_dir, exist_ok=True)
    model.set_logger(configure(run_dir, ["stdout", "tensorboard", "csv"]))

    # ---------------- learn ----------------
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[ckpt_cb, eval_cb],
        reset_num_timesteps=reset_flag,
        log_interval=1,
        # progress_bar=True,  # uncomment if your SB3 version supports it
    )

    model.save(args.model_path)
    print("[done] saved", args.model_path)


if __name__ == "__main__":
    main()
