import os
import argparse
import numpy as np
import torch
import gym
import gym_sokoban  # pip install gym-sokoban
import imageio

from collections import deque

# stable-baselines imports
from stable_baselines3 import PPO
try:
    # optional: RecurrentPPO lives in sb3_contrib
    from sb3_contrib import RecurrentPPO
except Exception:
    RecurrentPPO = None

from stable_baselines3.common.vec_env import (
    DummyVecEnv, SubprocVecEnv, VecTransposeImage, VecFrameStack, VecNormalize
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.logger import configure

# -------------------- custom wrappers (kept from your script) --------------------
class BackForthPenaltyWrapper(gym.Wrapper):
    def __init__(self, env, penalty=0.02, max_history=6):
        super().__init__(env)
        self.penalty = float(penalty)
        self.hist = deque(maxlen=int(max_history))

    def _player_pos(self):
        try:
            g = self.env.unwrapped.room_state
        except Exception:
            return None
        if g is None:
            return None
        ys, xs = np.where((g == 3) | (g == 6))
        return (int(ys[0]), int(xs[0])) if len(ys) else None

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
                a, b, c, d = self.hist[-4], self.hist[-3], self.hist[-2], self.hist[-1]
                pingpong2 = (a == c) and (b == d) and (a != b)
                bounce3 = False
                if len(self.hist) >= 5:
                    a2, b2, c2, d2, e2 = self.hist[-5], self.hist[-4], self.hist[-3], self.hist[-2], self.hist[-1]
                    bounce3 = (a2 == e2) and (b2 == d2) and (a2 != b2 != c2)
                if pingpong2 or bounce3:
                    reward -= self.penalty
                    info = dict(info)
                    info["backforth_penalty"] = info.get("backforth_penalty", 0.0) + self.penalty
        return obs, reward, done, info


class NoOpMovePenaltyWrapper(gym.Wrapper):
    def __init__(self, env, penalty=0.02):
        super().__init__(env)
        self.penalty = float(penalty)

    def _pos(self):
        try:
            g = self.env.unwrapped.room_state
            ys, xs = np.where((g == 3) | (g == 6))
            return (int(ys[0]), int(xs[0])) if len(ys) else None
        except Exception:
            return None

    def step(self, a):
        p0 = self._pos()
        obs, r, done, info = self.env.step(a)
        p1 = self._pos()
        if p0 is not None and p1 == p0:
            r -= self.penalty
            info = dict(info)
            info["noop_penalty"] = self.penalty
        return obs, r, done, info


class UndoMovePenaltyWrapper(gym.Wrapper):
    INV = {4:5, 5:4, 6:7, 7:6, 0:1, 1:0, 2:3, 3:2}
    def __init__(self, env, penalty=0.01):
        super().__init__(env)
        self.penalty = float(penalty)
        self.prev = None
    def reset(self, **kw):
        self.prev = None
        return self.env.reset(**kw)
    def step(self, a):
        obs, r, done, info = self.env.step(a)
        if self.prev is not None and a == self.INV.get(self.prev, None):
            r -= self.penalty
            info = dict(info); info["undo_penalty"] = self.penalty
        self.prev = a
        return obs, r, done, info

# -------------------- video callback (keeps your approach) --------------------
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

    @staticmethod
    def _unwrap_base(env):
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
            ep_reward += float(np.asarray(reward).sum() if isinstance(reward, (list, tuple, np.ndarray)) else float(reward))
            steps += 1

            rgb = self._safe_render(base)
            if rgb is not None:
                frames.append(rgb)

            done = bool(done_arr[0]) if isinstance(done_arr, (list, tuple, np.ndarray)) else bool(done_arr)
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

# -------------------- Print wrapper --------------------
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

# -------------------- robust Sokoban constructor --------------------
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

    raise RuntimeError("Could not construct a Sokoban env. Your gym_sokoban build exposes unexpected constructors/IDs.")

# -------------------- new: Grid -> fixed-size RGB converter --------------------
class GridToFixedImage(gym.ObservationWrapper):
    """
    Convert env.unwrapped.room_state (grid ints) -> fixed sized HxWx3 uint8 image.
    This makes the network input invariant to the underlying grid size.
    If room_state is not available, fallback to env.render().
    """
    def __init__(self, env, out_size=64):
        super().__init__(env)
        self.out_size = int(out_size)
        # output is HWC uint8, so VecTransposeImage will convert to CHW for the policy
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.out_size, self.out_size, 3), dtype=np.uint8)

    def _grid_to_rgb(self, grid):
        # grid codes vary across forks; we color common codes simply
        # 0 empty, 1 wall, 2 goal, 3 player, 5 box, 6 box_on_goal, 7 player_on_goal
        h, w = grid.shape
        img = np.zeros((h, w, 3), dtype=np.uint8)
        # simple palette (tweak if you like)
        img[(grid == 0)] = [0, 0, 0]           # empty -> black
        img[(grid == 1)] = [200, 200, 200]     # wall -> light gray
        img[(grid == 2)] = [0, 160, 0]         # goal -> green
        img[(grid == 3)] = [0, 0, 255]         # player -> blue
        img[(grid == 5)] = [160, 80, 0]        # box -> brown
        img[(grid == 6)] = [255, 180, 0]       # box_on_goal -> orange
        img[(grid == 7)] = [0, 255, 255]       # player on goal -> cyan
        return img

    def observation(self, obs):
        # prefer the internal grid if available
        grid = None
        try:
            grid = getattr(self.env.unwrapped, "room_state", None)
        except Exception:
            grid = None

        if grid is not None:
            grid = np.array(grid)
            # convert to rgb using palette and resize to out_size
            img = self._grid_to_rgb(grid)
        else:
            # fallback: try to use render output (HWC uint8)
            try:
                img = self.env.render(mode="rgb_array")
                img = np.asarray(img, dtype=np.uint8)
            except Exception:
                # ultimate fallback: empty image
                img = np.zeros((self.out_size, self.out_size, 3), dtype=np.uint8)

        # resize to fixed size (use nearest to preserve blocks)
        import cv2
        if img is None or img.size == 0:
            img = np.zeros((self.out_size, self.out_size, 3), dtype=np.uint8)
        else:
            try:
                img = cv2.resize(img, (self.out_size, self.out_size), interpolation=cv2.INTER_NEAREST)
            except Exception:
                img = np.zeros((self.out_size, self.out_size, 3), dtype=np.uint8)

        return img.astype(np.uint8)

# -------------------- base env constructor --------------------
def make_base_env(dim_room=(7,7), num_boxes=1, max_steps=120, step_penalty=0.0, print_tag=None, img_size=64):
    env = _construct_sokoban(dim_room=dim_room, num_boxes=num_boxes, max_steps=max_steps)

    # optional tiny step penalty wrapper (kept)
    if step_penalty > 0:
        from gym import RewardWrapper
        class StepPenalty(RewardWrapper):
            def __init__(self, env, p): super().__init__(env); self.p = float(p)
            def reward(self, r): return r - self.p
        env = StepPenalty(env, step_penalty)

    # your movement penalties
    env = NoOpMovePenaltyWrapper(env, penalty=0.01)
    env = UndoMovePenaltyWrapper(env, penalty=0.01)
    env = BackForthPenaltyWrapper(env, penalty=0.02, max_history=8)

    # convert grid -> fixed image (HWC uint8)
    env = GridToFixedImage(env, out_size=img_size)

    if print_tag is not None:
        env = PrintObsShapeOnce(env, tag=print_tag)
    return env

# -------------------- vector builder (identical pipeline for train & eval) --------------------
def build_vec_env(n_envs, n_stack, vec_cls, seed=None, norm_reward=False,
                  dim_room=(7,7), num_boxes=1, max_steps=120, step_penalty=0.0, tag="train", img_size=64):
    venv = make_vec_env(
        lambda: make_base_env(dim_room=dim_room, num_boxes=num_boxes, max_steps=max_steps, step_penalty=step_penalty, print_tag="base", img_size=img_size),
        n_envs=n_envs, seed=seed, vec_env_cls=vec_cls,
    )
    # VecTransposeImage expects HWC uint8 images and will convert to CHW
    venv = VecTransposeImage(venv)
    # frame stack (same value for train & eval!)
    venv = VecFrameStack(venv, n_stack=n_stack)
    venv = VecNormalize(venv, norm_obs=False, norm_reward=norm_reward)
    venv.training = (tag == "train")
    return venv

def assert_obs_compat(model, env, where):
    ms = tuple(model.observation_space.shape)
    es = tuple(env.observation_space.shape)
    if ms != es:
        raise RuntimeError(f"[{where}] Observation shape mismatch: model expects {ms} but env provides {es}. "
                           f"Use the SAME --n-stack and identical preprocessing order for train and eval.")

# -------------------- main --------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--total-timesteps", type=int, default=1_000_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--n-steps", type=int, default=1024)
    p.add_argument("--n-stack", type=int, default=2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--save-every", type=int, default=50_000)
    p.add_argument("--eval-every", type=int, default=50_000)
    p.add_argument("--eval-episodes", type=int, default=5)
    p.add_argument("--tb-dir", type=str, default="runs/ppo_tb")
    p.add_argument("--model-path", type=str, default="checkpoints/ppo.zip")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--room-h", type=int, default=8)
    p.add_argument("--room-w", type=int, default=8)
    p.add_argument("--num-boxes", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=160)
    p.add_argument("--step-penalty", type=float, default=0.0)
    p.add_argument("--img-size", type=int, default=64, help="fixed network input image size (px).")
    p.add_argument("--use-lstm", action="store_true", help="use RecurrentPPO (sb3_contrib) if available")
    p.add_argument("--record-now", dest="record_now", action="store_true")
    p.add_argument("--record-frames", type=int, default=250)
    args = p.parse_args()

    dim_room = (args.room_h, args.room_w)
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # quick record mode
    if args.record_now:
        eval_env = build_vec_env(n_envs=1, n_stack=args.n_stack, vec_cls=DummyVecEnv,
                                 seed=None, norm_reward=False, dim_room=dim_room, num_boxes=args.num_boxes, max_steps=args.max_steps, step_penalty=args.step_penalty, tag="eval", img_size=args.img_size)
        if os.path.exists(args.model_path):
            # loading with env helps recurrent policies to set spaces
            if args.use_lstm and RecurrentPPO is not None:
                model = RecurrentPPO.load(args.model_path, env=eval_env, device=device)
            else:
                model = PPO.load(args.model_path, env=eval_env, device=device)
            print(f"[record-now] loaded {args.model_path}")
        else:
            model = PPO("CnnPolicy", eval_env, device=device, n_steps=8, batch_size=8)
            print("[record-now] no checkpoint found; using fresh model for a sample video")

        video_cb = PeriodicEvalVideoCallback(eval_env=eval_env, out_dir="videos", every_steps=1, video_len=args.record_frames, deterministic=True, verbose=1)
        video_cb.model = model
        video_cb._on_step()
        return

    # build train env
    train_env = build_vec_env(n_envs=args.n_envs, n_stack=args.n_stack, vec_cls=SubprocVecEnv, seed=None,
                              norm_reward=False, dim_room=dim_room, num_boxes=args.num_boxes, max_steps=args.max_steps, step_penalty=args.step_penalty, tag="train", img_size=args.img_size)

    # create / load model
    use_lstm = args.use_lstm and (RecurrentPPO is not None)
    if args.resume and os.path.exists(args.model_path):
        if use_lstm:
            model = RecurrentPPO.load(args.model_path, env=train_env, device=device)
        else:
            model = PPO.load(args.model_path, env=train_env, device=device)
        print(f"[resume] loaded {args.model_path}")
        reset_flag = False
    else:
        policy = "CnnLstmPolicy" if use_lstm else "CnnPolicy"
        if use_lstm:
            # RecurrentPPO constructor will be used
            model = RecurrentPPO(policy, train_env, device=device, verbose=1,
                                 n_steps=args.n_steps, batch_size=256, learning_rate=args.lr,
                                 n_epochs=4, gamma=0.995, gae_lambda=0.95, ent_coef=args.ent_coef,
                                 clip_range=0.2, vf_coef=0.5, max_grad_norm=0.5, tensorboard_log=args.tb_dir)
        else:
            model = PPO("CnnPolicy", train_env, device=device, verbose=1,
                        n_steps=args.n_steps, batch_size=256, learning_rate=args.lr,
                        n_epochs=4, gamma=0.995, gae_lambda=0.95, ent_coef=args.ent_coef,
                        clip_range=0.2, vf_coef=0.5, max_grad_norm=0.5, tensorboard_log=args.tb_dir)
        reset_flag = True

    # sanity check obs shapes
    assert_obs_compat(model, train_env, where="train-start")

    # build eval env (same pipeline)
    eval_env = build_vec_env(n_envs=1, n_stack=args.n_stack, vec_cls=DummyVecEnv, seed=None,
                             norm_reward=False, dim_room=dim_room, num_boxes=args.num_boxes, max_steps=args.max_steps, step_penalty=args.step_penalty, tag="eval", img_size=args.img_size)
    assert_obs_compat(model, eval_env, where="eval-build")

    # callbacks
    video_cb = PeriodicEvalVideoCallback(eval_env=eval_env, out_dir="videos", every_steps=args.eval_every, video_len=300, deterministic=True, verbose=1)

    ckpt_cb = CheckpointCallback(save_freq=max(1, args.save_every // max(args.n_envs, 1)),
                                 save_path=os.path.dirname(args.model_path),
                                 name_prefix=os.path.splitext(os.path.basename(args.model_path))[0],
                                 save_replay_buffer=False, save_vecnormalize=False)

    eval_cb = EvalCallback(eval_env, best_model_save_path=os.path.dirname(args.model_path),
                           log_path=os.path.join(args.tb_dir, "eval"),
                           eval_freq=max(1, args.eval_every // max(args.n_envs, 1)),
                           n_eval_episodes=args.eval_episodes, deterministic=True, render=False)

    run_dir = os.path.join(args.tb_dir, "PPO_run")
    os.makedirs(run_dir, exist_ok=True)
    model.set_logger(configure(run_dir, ["stdout", "tensorboard", "csv"]))

    # learn
    model.learn(total_timesteps=args.total_timesteps, callback=[ckpt_cb, eval_cb, video_cb], reset_num_timesteps=reset_flag, log_interval=1)

    model.save(args.model_path)
    print("[done] saved", args.model_path)

if __name__ == "__main__":
    main()
