# train.py
import os
import argparse
import numpy as np
import torch
import gym
import gym_sokoban
import imageio
from collections import deque

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecTransposeImage, VecFrameStack, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.env_util import make_vec_env

#  Wrappers 

class BackForthPenaltyWrapper(gym.Wrapper):
    def __init__(self, env, penalty=0.0, max_history=6):
        super().__init__(env)
        self.penalty = float(penalty)
        self.hist = deque(maxlen=int(max_history))

    def _player_pos(self):
        g = getattr(self.env.unwrapped, "room_state", None)
        if g is None:
            return None
        ys, xs = np.where((g == 3) | (g == 6))
        if len(ys) == 0:
            return None
        return (int(ys[0]), int(xs[0]))

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.hist.clear()
        p = self._player_pos()
        if p is not None:
            self.hist.append(p)
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        p = self._player_pos()
        if p is not None:
            self.hist.append(p)
            if len(self.hist) >= 4:
                a, b, c, d = list(self.hist)[-4:]
                pingpong2 = (a==c) and (b==d) and (a!=b)
                bounce3 = False
                if len(self.hist)>=5:
                    a2, b2, c2, d2, e2 = list(self.hist)[-5:]
                    bounce3 = (a2==e2) and (b2==d2) and (a2!=b2!=c2)
                if pingpong2 or bounce3:
                    reward -= self.penalty
                    info = dict(info)
                    info["backforth_penalty"] = info.get("backforth_penalty", 0.0) + self.penalty
        return obs, reward, done, info

class NoOpMovePenaltyWrapper(gym.Wrapper):
    def __init__(self, env, penalty=0.0):
        super().__init__(env)
        self.penalty = float(penalty)
    def _pos(self):
        g = getattr(self.env.unwrapped, "room_state", None)
        if g is None:
            return None
        ys,xs = np.where((g==3)|(g==6))
        if len(ys)==0:
            return None
        return (int(ys[0]), int(xs[0]))
    def step(self, a):
        p0 = self._pos()
        obs, r, done, info = self.env.step(a)
        p1 = self._pos()
        if p0 is not None and p1==p0:
            r -= self.penalty
            info = dict(info)
            info["noop_penalty"] = self.penalty
        return obs, r, done, info

class UndoMovePenaltyWrapper(gym.Wrapper):
    INV = {4:5, 5:4, 6:7, 7:6, 0:1, 1:0, 2:3, 3:2}
    def __init__(self, env, penalty=0.0):
        super().__init__(env)
        self.penalty = float(penalty)
        self.prev = None
    def reset(self, **kwargs):
        self.prev = None
        return self.env.reset(**kwargs)
    def step(self, a):
        obs, r, done, info = self.env.step(a)
        if self.prev is not None and a == self.INV.get(self.prev, None):
            r -= self.penalty
            info = dict(info)
            info["undo_penalty"] = self.penalty
        self.prev = a
        return obs, r, done, info

#  Video callback 

class PeriodicEvalVideoCallback(BaseCallback):
    def __init__(self, eval_env, out_dir="videos", every_steps=50_000, video_len=300, deterministic=True, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.out_dir = out_dir
        self.every_steps = every_steps
        self.video_len = video_len
        self.deterministic = deterministic
        self._last = 0
        os.makedirs(out_dir, exist_ok=True)

    def _unwrap_base(self, env):
        base = env
        while hasattr(base, "env"):
            base = base.env
        return base

    def _safe_render(self, base):
        try:
            rgb = base.render(mode="rgb_array")
            return np.asarray(rgb, dtype=np.uint8) if rgb is not None else None
        except Exception:
            return None

    def _on_step(self):
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
        try:
            with imageio.get_writer(mp4, fps=8) as w:
                for f in frames:
                    w.append_data(f)
            if self.verbose:
                print(f"[video] saved {mp4}")
        except Exception as e:
            gif = os.path.join(self.out_dir, f"{stem}.gif")
            imageio.mimsave(gif, frames, fps=8)
            if self.verbose:
                print(f"[video] MP4 failed ({e}), saved GIF {gif}")
        return True

#  Env builder 

def make_base_env(dim_room=(7,7), num_boxes=1, max_steps=160, step_penalty=0.0):
    env = None
    try:
        from gym_sokoban.envs import SokobanEnv
        env = SokobanEnv(dim_room=dim_room, num_boxes=num_boxes, max_steps=max_steps)
    except Exception:
        env = gym.make("Sokoban-v0")

    if step_penalty > 0:
        from gym import RewardWrapper
        class StepPenalty(RewardWrapper):
            def __init__(self, env, p): super().__init__(env); self.p = float(p)
            def reward(self, r): return r - self.p
        env = StepPenalty(env, step_penalty)

    # start with zero penalties, will increase automatically in curriculum
    env = NoOpMovePenaltyWrapper(env, penalty=step_penalty*1.0)
    env = UndoMovePenaltyWrapper(env, penalty=step_penalty*0.6)
    env = BackForthPenaltyWrapper(env, penalty=step_penalty*0.6)
    return env

def build_vec_env(n_envs, n_stack, vec_cls, dim_room=(8,8), num_boxes=1, max_steps=160, step_penalty=0.0):
    venv = make_vec_env(
        lambda: make_base_env(dim_room=dim_room, num_boxes=num_boxes, max_steps=max_steps, step_penalty=step_penalty),
        n_envs=n_envs,
        vec_env_cls=vec_cls,
    )
    venv = VecTransposeImage(venv)
    venv = VecFrameStack(venv, n_stack=n_stack)
    venv = VecNormalize(venv, norm_obs=False, norm_reward=True)
    return venv

#  Curriculum training 

def curriculum_train(model, device, args):
    # Curriculum: start with 1 box → 2 boxes → 3 boxes etc...
    stages = [
        {"num_boxes": 1, "timesteps": args.total_timesteps//3, "step_penalty": 0.0},
        {"num_boxes": 2, "timesteps": args.total_timesteps//3, "step_penalty": 0.003},
        {"num_boxes": 2, "timesteps": args.total_timesteps//3, "step_penalty": 0.005},
    ]
    for stage in stages:
        print(f"[curriculum] Training with {stage['num_boxes']} boxes, step penalty {stage['step_penalty']}")
        train_env = build_vec_env(
            n_envs=args.n_envs,
            n_stack=args.n_stack,
            vec_cls=SubprocVecEnv,
            dim_room=(args.room_h,args.room_w),
            num_boxes=stage["num_boxes"],
            max_steps=args.max_steps,
            step_penalty=stage["step_penalty"],
        )
        eval_env = build_vec_env(
            n_envs=1,
            n_stack=args.n_stack,
            vec_cls=DummyVecEnv,
            dim_room=(args.room_h,args.room_w),
            num_boxes=stage["num_boxes"],
            max_steps=args.max_steps,
            step_penalty=stage["step_penalty"],
        )
        video_cb = PeriodicEvalVideoCallback(eval_env, out_dir="videos", every_steps=50_000, video_len=300, deterministic=True, verbose=1)
        ckpt_cb = CheckpointCallback(save_freq=50_000//args.n_envs, save_path="checkpoints", name_prefix=f"ppo_sokoban_{stage['num_boxes']}box")
        eval_cb = EvalCallback(eval_env, best_model_save_path="checkpoints", log_path=os.path.join(args.tb_dir,"eval"), eval_freq=50_000//args.n_envs, n_eval_episodes=5)

        model.set_env(train_env)
        model.learn(total_timesteps=stage["timesteps"], callback=[ckpt_cb, eval_cb, video_cb])

#  Main 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=3_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--n-stack", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--ent-coef", type=float, default=0.02)
    parser.add_argument("--room-h", type=int, default=8)
    parser.add_argument("--room-w", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=160)
    parser.add_argument("--model-path", type=str, default="ppo_sokoban.zip")
    parser.add_argument("--tb-dir", type=str, default="runs/ppo_tb")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Start with 1 box for curriculum
    train_env = build_vec_env(
        n_envs=args.n_envs,
        n_stack=args.n_stack,
        vec_cls=SubprocVecEnv,
        dim_room=(args.room_h,args.room_w),
        num_boxes=1,
        max_steps=args.max_steps,
        step_penalty=0.0,
    )

    model = PPO(
        "CnnPolicy",
        train_env,
        device=device,
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
        verbose=1,
    )

    # Curriculum training
    curriculum_train(model, device, args)

    model.save(args.model_path)
    print("[done] saved", args.model_path)

if __name__=="__main__":
    main()
