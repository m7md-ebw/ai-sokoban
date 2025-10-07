from dataclasses import dataclass
from typing import Optional, List
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .env_wrappers import make_sokoban_env
from .agents import HRLAgent, HRLConfig, HRLTrainer
from .replay_buffer import RolloutBuffers
from .utils import set_seed, make_dir, timestamp


@dataclass
class TrainConfig:
    levels: int = 4                 # number of fixed-seed variations to cycle
    level_seeds: List[int] = None   # will default to 4 seeds
    total_episodes: int = 200
    max_steps_per_ep: int = 200
    manager_interval: int = 6
    obs_size: int = 64
    seed: int = 7
    save_dir: str = "checkpoints"
    logdir: str = "runs"
    device: str = "cuda"          # default to CUDA (falls back inside code if unavailable)
    eval_interval_steps: int = 50000
    eval_episodes: int = 3
    record_dir: str = "videos"
    load_path: Optional[str] = None  # load to resume or warm-start
    env_id: str = "auto"


def _default_seeds(levels: int) -> List[int]:
    base = [101, 202, 303, 404]
    return base[:max(1, min(levels, 4))]


def train(cfg: TrainConfig):
    set_seed(cfg.seed)
    make_dir(cfg.save_dir)
    make_dir(cfg.record_dir)

    # Env to read action space
    env = make_sokoban_env(seed=cfg.seed, obs_size=cfg.obs_size, env_id=cfg.env_id)
    n_actions = env.action_space.n

    # Build agent/trainer
    hrl_cfg = HRLConfig(obs_size=cfg.obs_size, manager_interval=cfg.manager_interval)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    agent = HRLAgent(n_actions=n_actions, cfg=hrl_cfg).to(device)
    trainer = HRLTrainer(agent, device=device)

    # Resume or warm-start
    global_step = 0
    episodes_done = 0
    latest_ckpt = os.path.join(cfg.save_dir, "latest.pt")
    if cfg.load_path is not None and os.path.isfile(cfg.load_path):
        data = torch.load(cfg.load_path, map_location=device)
        agent.load_state_dict(data["agent"])  # resume weights
        if "opt" in data:
            trainer.opt.load_state_dict(data["opt"])  # resume optimizer if available
        global_step = int(data.get("global_step", 0))
        episodes_done = int(data.get("episodes_done", 0))
        print(f"[INFO] Loaded checkpoint {cfg.load_path} @ step {global_step}, ep {episodes_done}")

    # TensorBoard writer
    run_name = f"hrl_sokoban_{timestamp()}"
    writer = SummaryWriter(log_dir=os.path.join(cfg.logdir, run_name))
    writer.add_hparams({
        "obs_size": cfg.obs_size,
        "manager_interval": cfg.manager_interval,
        "feat_dim": hrl_cfg.feat_dim,
        "goal_dim": hrl_cfg.goal_dim,
        "intrinsic_coef": hrl_cfg.intrinsic_coef,
        "lr": hrl_cfg.lr,
    }, {})

    level_seeds = cfg.level_seeds or _default_seeds(cfg.levels)

    pbar = tqdm(total=cfg.total_episodes, desc="Train", leave=True)
    pbar.update(episodes_done)

    while episodes_done < cfg.total_episodes:
        # Cycle seeds as lightweight "levels"
        level_idx = episodes_done % len(level_seeds)
        level_seed = level_seeds[level_idx] + episodes_done  # vary slightly
        env = make_sokoban_env(seed=level_seed, obs_size=cfg.obs_size, env_id=cfg.env_id)

        # Reset (compat)
        try:
            obs, info = env.reset()
        except Exception:
            obs = env.reset()
            info = {}
        done = False
        t = 0
        buf = RolloutBuffers()

        # Initial manager goal
        with torch.no_grad():
            obs_t = torch.tensor(obs, device=device).unsqueeze(0)
            goal, m_value, m_feat = agent.act_manager(obs_t)

        episodic_env_r = 0.0
        episodic_int_r = 0.0

        while not done and t < cfg.max_steps_per_ep:
            obs_t = torch.tensor(obs, device=device).unsqueeze(0)
            with torch.no_grad():
                action, logp, value, feat = agent.act_worker(obs_t, goal)

            try:
                next_obs, reward, terminated, truncated, info = env.step(int(action.item()))
                done = terminated or truncated

            except Exception:
                next_obs, reward, done, info = env.step(int(action.item()))

            with torch.no_grad():
                next_obs_t = torch.tensor(next_obs, device=device).unsqueeze(0)
                next_feat = agent.encoder(next_obs_t)
                r_int = trainer.compute_intrinsic_reward(feat, goal, next_feat)

            total_r = float(reward) + agent.cfg.intrinsic_coef * float(r_int.item())
            episodic_env_r += float(reward)
            episodic_int_r += float(r_int.item())

            buf.add_worker(
                obs=obs_t.squeeze(0).to(device),
                goal=goal.squeeze(0).to(device),
                action=action.to(device),
                logp=logp.to(device),
                value=value.to(device),
                reward=torch.tensor(total_r, dtype=torch.float32, device=device),
                done=torch.tensor(1.0 if done else 0.0, dtype=torch.float32, device=device),
            )

            obs = next_obs
            t += 1
            global_step += 1

            if t % agent.cfg.manager_interval == 0 or done:
                m_reward = torch.tensor(episodic_int_r, dtype=torch.float32, device=device)
                buf.add_manager(feat=m_feat.to(device), goal=goal.squeeze(0).to(device), value=m_value.to(device), reward=m_reward)
                # next goal unless episode ended
                if not done:
                    with torch.no_grad():
                        goal, m_value, m_feat = agent.act_manager(next_obs_t)
                episodic_int_r = 0.0 

            # Periodic evaluation & video saving by timesteps
            if cfg.eval_interval_steps > 0 and global_step % cfg.eval_interval_steps == 0:
                ckpt_path = os.path.join(cfg.save_dir, f"step_{global_step}.pt")
                torch.save({
                    "agent": agent.state_dict(),
                    "opt": trainer.opt.state_dict(),
                    "cfg": hrl_cfg.__dict__,
                    "train_cfg": cfg.__dict__,
                    "global_step": global_step,
                    "episodes_done": episodes_done,
                }, ckpt_path)
                # Record one eval episode
                from .eval import evaluate, EvalConfig
                eval_cfg = EvalConfig(episodes=1, obs_size=cfg.obs_size, max_steps=cfg.max_steps_per_ep,
                                      record_dir=os.path.join(cfg.record_dir, f"step_{global_step}"), device=cfg.device, env_id=cfg.env_id)
                evaluate(ckpt_path, eval_cfg)

        stats = trainer.update_from_rollout(buf)
        buf.clear()

        episodes_done += 1

        # Save latest checkpoint
        latest_payload = {
            "agent": agent.state_dict(),
            "opt": trainer.opt.state_dict(),
            "cfg": hrl_cfg.__dict__,
            "train_cfg": cfg.__dict__,
            "global_step": global_step,
            "episodes_done": episodes_done,
        }
        torch.save(latest_payload, latest_ckpt)

        # TensorBoard
        writer.add_scalar("reward/env", episodic_env_r, episodes_done)
        writer.add_scalar("reward/intrinsic", episodic_int_r, episodes_done)
        writer.add_scalar("loss/worker_policy", stats["worker_policy_loss"], episodes_done)
        writer.add_scalar("loss/worker_value", stats["worker_value_loss"], episodes_done)
        writer.add_scalar("loss/manager_value", stats["manager_value_loss"], episodes_done)
        writer.add_scalar("misc/entropy", stats["entropy"], episodes_done)
        writer.add_scalar("misc/global_step", global_step, episodes_done)

        
        pbar.set_postfix_str(
            f"lvl {level_idx+1}/{len(level_seeds)} | ts {global_step} | R {episodic_env_r:+.1f} | len {t} | loss {stats['total_loss']:.3f}"
        )
        pbar.update(1)

    pbar.close()
    writer.close()
    print(f"Saved latest checkpoint to: {latest_ckpt}")
