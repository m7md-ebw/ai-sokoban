import os, math, random, argparse, logging, warnings
from collections import deque
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import gym_sokoban
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API", category=UserWarning, module="gym_sokoban")

try:
    import gym
    from gym.vector import AsyncVectorEnv
except Exception as e:
    raise SystemExit("This script requires classic Gym (e.g., gym==0.21.0).\n" + str(e))

ENV_ID = "Sokoban-v0"
IMG_SIZE = 160

# Logging

def setup_logger(logfile: str):
    logger = logging.getLogger("sokoban")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(logfile, mode="a", encoding="utf-8"); fh.setFormatter(fmt)
    sh = logging.StreamHandler(); sh.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(sh)
    return logger

# Env helpers (Sokoban-v0)

def make_single_env(seed, max_steps=None):
    def thunk():
        env = gym.make(ENV_ID)
        if max_steps is not None:
            env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps)
        try:
            env.reset(seed=seed)
        except TypeError:
            env.seed(seed)
        return env
    return thunk

def safe_vec_reset(venv):
    out = venv.reset()
    return out[0] if isinstance(out, tuple) else out

def safe_vec_step(venv, actions):
    out = venv.step(actions)
    if len(out) == 4:
        return out
    if len(out) == 5:
        obs, rew, done, trunc, info = out
        return obs, rew, np.logical_or(done, trunc), info
    raise RuntimeError("Unsupported vector step format")

def safe_reset_single(env):
    out = env.reset()
    return out[0] if isinstance(out, tuple) else out

def safe_step_single(env, action):
    out = env.step(action)
    if len(out) == 4:
        obs, rew, done, info = out
        trunc = False
    else:
        obs, rew, done, trunc, info = out
    return obs, rew, done, trunc, info

# Preprocessing (gray, 160x160, UINT8) + framestack

def to_gray_batch(obs_rgb_batch):
    n = obs_rgb_batch.shape[0]
    out = np.empty((n, IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    for i in range(n):
        g = cv2.cvtColor(obs_rgb_batch[i], cv2.COLOR_RGB2GRAY)
        out[i] = cv2.resize(g, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return out

def to_gray(obs_rgb):
    g = cv2.cvtColor(obs_rgb, cv2.COLOR_RGB2GRAY)
    return cv2.resize(g, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA).astype(np.uint8)

class FrameStackVec:
    def __init__(self, n_envs, k):
        self.n, self.k = n_envs, k; self.frames=None
    def reset(self, first_gray):
        self.frames = np.repeat(first_gray[:, None, :, :], self.k, axis=1)
        return self.frames.copy()
    def step(self, next_gray):
        self.frames = np.concatenate([self.frames[:, 1:, :, :], next_gray[:, None, :, :]], axis=1)
        return self.frames.copy()

class FrameStackSingle:
    def __init__(self, k):
        self.k = k; self.frames=None
    def reset(self, first_gray):
        self.frames = np.repeat(first_gray[None, :, :], self.k, axis=0)
        return self.frames.copy()
    def step(self, next_gray):
        self.frames = np.concatenate([self.frames[1:, :, :], next_gray[None, :, :]], axis=0)
        return self.frames.copy()

# Replay buffer

class ReplayBuffer:
    def __init__(self, cap: int, k: int, h: int, w: int):
        self.cap = int(cap)
        self.k, self.h, self.w = int(k), int(h), int(w)
        self.s  = np.empty((self.cap, self.k, self.h, w), dtype=np.uint8)
        self.ns = np.empty((self.cap, self.k, self.h, w), dtype=np.uint8)
        self.a  = np.empty((self.cap,), dtype=np.int64)
        self.r  = np.empty((self.cap,), dtype=np.float32)
        self.d  = np.empty((self.cap,), dtype=np.uint8)
        self.idx = 0
        self.size = 0
    def push_batch(self, states, actions, rewards, next_states, dones):
        n = states.shape[0]
        for i in range(n):
            self.s[self.idx]  = states[i]
            self.ns[self.idx] = next_states[i]
            self.a[self.idx]  = int(actions[i])
            self.r[self.idx]  = float(rewards[i])
            self.d[self.idx]  = int(dones[i])
            self.idx = (self.idx + 1) % self.cap
            if self.size < self.cap: self.size += 1
    def sample(self, bs):
        idxs = np.random.randint(0, self.size, size=bs)
        s  = torch.from_numpy(self.s[idxs]).float().div_(255.0)
        ns = torch.from_numpy(self.ns[idxs]).float().div_(255.0)
        a  = torch.from_numpy(self.a[idxs]).long()
        r  = torch.from_numpy(self.r[idxs]).float()
        d  = torch.from_numpy(self.d[idxs]).float()
        return s, a, r, ns, d
    def __len__(self):
        return self.size

# Dueling DQN

class DuelingDQN(nn.Module):
    def __init__(self, in_ch, n_actions, img_size=IMG_SIZE):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=8, stride=4), nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),    nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),    nn.ReLU(True),
        )
        with torch.no_grad():
            n_flat = self.feat(torch.zeros(1, in_ch, img_size, img_size)).view(1, -1).shape[1]
        self.val = nn.Sequential(nn.Linear(n_flat, 512), nn.ReLU(True), nn.Linear(512, 1))
        self.adv = nn.Sequential(nn.Linear(n_flat, 512), nn.ReLU(True), nn.Linear(512, n_actions))
    def forward(self, x):
        x = self.feat(x)
        x = torch.flatten(x, 1)
        v = self.val(x); a = self.adv(x)
        return v + (a - a.mean(dim=1, keepdim=True))

# Epsilon schedule

class EpsilonScheduler:
    def __init__(self, start=1.0, end=0.1, decay_steps=200_000):
        self.s, self.e, self.decay, self.t = start, end, decay_steps, 0
    def step(self, n=1): self.t += n
    def eps(self): return self.e + (self.s - self.e) * math.exp(-self.t / self.decay)

# Info helpers

def extract_boxes_and_solved(infos, n_env, dones):
    """Return (nb, all_on_now, solved_at_done) from vector infos across Gym/Gymnasium variants."""
    nb = np.zeros((n_env,), dtype=np.int32)
    all_on = np.zeros((n_env,), dtype=np.int32)
    solved_at_done = np.zeros((n_env,), dtype=np.int32)

    if isinstance(infos, (list, tuple)):
        for i, inf in enumerate(infos):
            if isinstance(inf, dict):
                nbi = int(inf.get("num_boxes_on_target", 0))
                nb[i] = nbi
                ao = 1 if inf.get("all_boxes_on_target", False) else 0
                all_on[i] = ao or (nbi >= 3)
                if dones[i] and (ao or nbi >= 3):
                    solved_at_done[i] = 1

    elif isinstance(infos, dict):
        if "num_boxes_on_target" in infos:
            try: nb = np.array(infos["num_boxes_on_target"]).astype(np.int32)
            except Exception: pass
        if "all_boxes_on_target" in infos:
            try:
                ao_arr = np.array(infos["all_boxes_on_target"]).astype(np.int32)
                all_on = np.maximum(all_on, ao_arr)
            except Exception:
                pass
        fi = infos.get("final_info", None)
        if fi is not None:
            for i, inf in enumerate(fi):
                if isinstance(inf, dict) and dones[i]:
                    nbi = int(inf.get("num_boxes_on_target", nb[i]))
                    ao = bool(inf.get("all_boxes_on_target", False))
                    solved_at_done[i] = 1 if (ao or nbi >= 3) else 0

    all_on = np.maximum(all_on, (nb >= 3).astype(np.int32))
    return nb, all_on, solved_at_done

# Training video buffer (saves training episodes to MP4)

class TrainingVideoBuffer:
    def __init__(self, n_envs, video_dir, fps=12, periodic_env_index=0):
        self.n = n_envs
        self.dir = video_dir
        os.makedirs(self.dir, exist_ok=True)
        self.fps = int(fps)
        self.periodic_env_index = int(periodic_env_index)
        self._frames = [[] for _ in range(self.n)]

    def reset_episode(self, first_obs_batch):
        self._frames = [[first_obs_batch[i]] for i in range(self.n)]

    def add_step(self, obs_batch):
        for i in range(self.n):
            self._frames[i].append(obs_batch[i])

    def _save_mp4(self, frames, path):
        if not frames:
            return False
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, self.fps, (w, h))
        for f in frames:
            bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
            writer.write(bgr)
        writer.release()
        return True

    def save_env_episode(self, env_idx, path):
        env_idx = int(env_idx)
        frames = self._frames[env_idx]
        return self._save_mp4(frames, path)

# Optimize

def optimize(policy, target, buf, opt, huber, bs, gamma, device):
    policy.train(); target.eval()
    s, a, r, ns, d = buf.sample(bs)
    s, a, r, ns, d = s.to(device), a.to(device), r.to(device), ns.to(device), d.to(device)

    q = policy(s).gather(1, a.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        na = torch.argmax(policy(ns), dim=1, keepdim=True)
        nq = target(ns).gather(1, na).squeeze(1)
        tgt = r + (1.0 - d) * gamma * nq

    loss = huber(q, tgt)
    opt.zero_grad(set_to_none=True); loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
    opt.step()
    return float(loss.item())

# Train

def train(args):
    logger = setup_logger(args.logfile)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    logger.info(f"[INFO] Using device: {device}")

    torch.backends.cudnn.benchmark = True

    env_fns = [make_single_env(seed=args.seed + i, max_steps=args.max_steps_per_episode) for i in range(args.num_envs)]
    venv = AsyncVectorEnv(env_fns)
    n_env = args.num_envs

    obs0 = safe_vec_reset(venv)
    n_actions = venv.single_action_space.n

    fs = FrameStackVec(n_env, args.k_frames)
    state_np = fs.reset(to_gray_batch(obs0))

    video_dir = os.path.join(args.logdir, args.train_video_dirname)
    tvb = TrainingVideoBuffer(n_envs=n_env, video_dir=video_dir, fps=args.train_video_fps, periodic_env_index=args.train_record_env)
    tvb.reset_episode(obs0)

    prev_boxes = np.zeros((n_env,), dtype=np.int32)

    policy = DuelingDQN(args.k_frames, n_actions, img_size=IMG_SIZE).to(device)
    target = DuelingDQN(args.k_frames, n_actions, img_size=IMG_SIZE).to(device)
    target.load_state_dict(policy.state_dict()); target.eval()

    opt = optim.Adam(policy.parameters(), lr=args.lr)
    huber = nn.SmoothL1Loss()
    buf = ReplayBuffer(args.replay_size, args.k_frames, IMG_SIZE, IMG_SIZE)
    writer = SummaryWriter(args.logdir)
    eps = EpsilonScheduler(args.eps_start, args.eps_end, args.eps_decay_steps)

    gstep = 0
    best_avg = -1e9
    recent_returns = deque(maxlen=100)
    ep_counter = 0

    ep_returns = np.zeros((n_env,), dtype=np.float32)
    ep_steps   = np.zeros((n_env,), dtype=np.int32)

    next_periodic = max(0, int(args.train_record_every))
    pending_periodic_records = 0

    while ep_counter < args.episodes:
        with torch.no_grad():
            inp = torch.from_numpy(state_np).to(device).float().div_(255.0)
            q = policy(inp)
            greedy = torch.argmax(q, dim=1).cpu().numpy()
        epsilon_now = eps.eps()
        actions = np.where(
            np.random.rand(n_env) < epsilon_now,
            np.array([venv.single_action_space.sample() for _ in range(n_env)], dtype=np.int64),
            greedy
        )

        next_obs, rewards_raw, dones, infos = safe_vec_step(venv, actions)
        nb, _, solved_at_done = extract_boxes_and_solved(infos, n_env, dones)

        delta = (nb - prev_boxes).astype(np.float32)
        shaped_rewards = rewards_raw.astype(np.float32) + args.box_bonus * delta

        if np.any(dones) and args.finish_bonus > 0.0:
            for i in range(n_env):
                if dones[i] and solved_at_done[i]:
                    shaped_rewards[i] += args.finish_bonus

        prev_boxes = nb

        next_gray = to_gray_batch(next_obs)
        next_state_np = fs.step(next_gray)

        buf.push_batch(state_np, actions, shaped_rewards, next_state_np, dones.astype(np.float32))

        ep_returns += shaped_rewards
        ep_steps   += 1
        gstep      += n_env
        eps.step(n_env)

        tvb.add_step(next_obs)

        if len(buf) >= args.batch_size and (gstep // n_env) % args.train_every == 0:
            loss = optimize(policy, target, buf, opt, huber, args.batch_size, args.gamma, device)
            writer.add_scalar("train/loss", loss, gstep)

        if (gstep // n_env) % args.target_update == 0:
            target.load_state_dict(policy.state_dict())

        if args.train_record_every > 0:
            while gstep >= next_periodic:
                pending_periodic_records += 1
                next_periodic += args.train_record_every

        if np.any(dones):
            for i in range(n_env):
                if dones[i]:
                    ep_counter += 1
                    avg100 = np.mean(recent_returns) if len(recent_returns) else ep_returns[i]
                    writer.add_scalar("episode/return", ep_returns[i], ep_counter)
                    writer.add_scalar("episode/avg_return_100", avg100, ep_counter)
                    writer.add_scalar("episode/steps", ep_steps[i], ep_counter)
                    writer.add_scalar("episode/epsilon", epsilon_now, ep_counter)
                    writer.add_scalar("episode/solved", float(solved_at_done[i]), ep_counter)
                    recent_returns.append(float(ep_returns[i]))

                    logger.info(
                        f"[EP {ep_counter:05d}] return={ep_returns[i]:.2f} avg100={avg100:.2f} "
                        f"steps={int(ep_steps[i])} eps~{epsilon_now:.3f} solved={int(solved_at_done[i])}"
                    )

                    if solved_at_done[i]:
                        fname = f"train_solved_step{gstep:09d}_ep{ep_counter:05d}_env{i}.mp4"
                        fpath = os.path.join(video_dir, fname)
                        ok = tvb.save_env_episode(i, fpath)
                        if ok:
                            logger.info(f"[TRAIN-VIDEO] Saved solved episode -> {fpath}")
                        else:
                            logger.warning(f"[TRAIN-VIDEO] Failed to save solved episode (no frames?) env={i}")

            if pending_periodic_records > 0:
                idx = int(args.train_record_env)
                fname = f"train_periodic_step{gstep:09d}_ep{ep_counter:05d}_env{idx}.mp4"
                fpath = os.path.join(video_dir, fname)
                ok = tvb.save_env_episode(idx, fpath)
                if ok:
                    logger.info(f"[TRAIN-VIDEO] Saved periodic episode -> {fpath}")
                else:
                    logger.warning(f"[TRAIN-VIDEO] Failed to save periodic episode (no frames?) env={idx}")
                pending_periodic_records -= 1

            cur_avg = np.mean(recent_returns)
            if cur_avg > best_avg:
                best_avg = cur_avg
                os.makedirs(args.checkpoint_dir, exist_ok=True)
                bestp = os.path.join(args.checkpoint_dir, "best.pt")
                torch.save({
                    "policy": policy.state_dict(),
                    "target": target.state_dict(),
                    "optim":  opt.state_dict(),
                    "global_step": gstep,
                    "args": vars(args)
                }, bestp)
                writer.add_scalar("episode/best_avg_return", best_avg, ep_counter)
                logger.info(f"[INFO] New best avg100={best_avg:.2f}. Saved {bestp}")

            if ep_counter % args.checkpoint_every == 0:
                os.makedirs(args.checkpoint_dir, exist_ok=True)
                path = os.path.join(args.checkpoint_dir, f"sokoban_dqn_ep{ep_counter}.pt")
                torch.save({
                    "policy": policy.state_dict(),
                    "target": target.state_dict(),
                    "optim":  opt.state_dict(),
                    "global_step": gstep,
                    "args": vars(args)
                }, path)
                logger.info(f"[INFO] Saved checkpoint: {path}")

            obs0 = safe_vec_reset(venv)
            state_np = fs.reset(to_gray_batch(obs0))
            tvb.reset_episode(obs0)
            prev_boxes.fill(0); ep_returns.fill(0); ep_steps.fill(0)
        else:
            state_np = next_state_np

    writer.close(); venv.close()
    logger.info("[DONE] Training finished.")

# CLI

def parse_args():
    p = argparse.ArgumentParser(
        "Vectorized DQN on Sokoban-v0 (dueling + k4 + signed shaping + finish bonus + training-video + logfile)"
    )
    p.add_argument("--episodes", type=int, default=4000)
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=5e-5)

    p.add_argument("--replay-size", type=int, default=40_000, help="capacity (transitions)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--train-every", type=int, default=2)
    p.add_argument("--target-update", type=int, default=2000)

    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.1)
    p.add_argument("--eps-decay-steps", type=int, default=200_000)

    p.add_argument("--k-frames", type=int, default=4)

    p.add_argument("--box-bonus", type=float, default=4.0, help="per-step reward for ± change in boxes-on-target")
    p.add_argument("--finish-bonus", type=float, default=30.0, help="extra bonus only when the level is actually solved (at terminal)")

    p.add_argument("--train-record-every", type=int, default=5_000, help="save one training episode every N global env-steps (0=off)")
    p.add_argument("--train-record-env", type=int, default=0, help="which env index to save for periodic recordings")
    p.add_argument("--train-video-fps", type=int, default=12, help="fps for saved MP4s")
    p.add_argument("--train-video-dirname", type=str, default="videos_train/dqn3boxes", help="subdir inside logdir to store MP4s")

    p.add_argument("--max-steps-per-episode", type=int, default=180, help="TimeLimit cap for 10x10 levels")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true")

    p.add_argument("--logdir", type=str, default="runs/sokoban_v0_dqn_vec")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints/dqn3boxes")
    p.add_argument("--checkpoint-every", type=int, default=200)
    p.add_argument("--logfile", type=str, default="train_log4.txt")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)