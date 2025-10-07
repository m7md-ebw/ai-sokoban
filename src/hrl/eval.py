from dataclasses import dataclass
from typing import Optional
import os
import torch
from .env_wrappers import make_sokoban_env
from .agents import HRLAgent, HRLConfig
from .utils import make_dir
from .video_recorder import record_episode


@dataclass
class EvalConfig:
    episodes: int = 5
    obs_size: int = 64
    max_steps: int = 200
    record_dir: str = "videos"
    device: str = "cuda"
    env_id: str = "auto"


def load_agent_for_eval(ckpt_path: str, n_actions: int, device: torch.device) -> HRLAgent:
    data = torch.load(ckpt_path, map_location=device)
    cfg_dict = data.get("cfg", HRLConfig().__dict__)
    hrl_cfg = HRLConfig(**cfg_dict)
    agent = HRLAgent(n_actions=n_actions, cfg=hrl_cfg).to(device)
    agent.load_state_dict(data["agent"])
    agent.eval()
    return agent


def evaluate(ckpt_path: Optional[str], eval_cfg: EvalConfig):
    device = torch.device(eval_cfg.device if torch.cuda.is_available() else "cpu")
    env = make_sokoban_env(seed=None, obs_size=eval_cfg.obs_size, env_id=eval_cfg.env_id)
    n_actions = env.action_space.n

    if ckpt_path is None:
        agent = HRLAgent(n_actions=n_actions, cfg=HRLConfig(obs_size=eval_cfg.obs_size)).to(device)
    else:
        agent = load_agent_for_eval(ckpt_path, n_actions, device)

    make_dir(eval_cfg.record_dir)

    def act(obs_np):
        with torch.no_grad():
            obs_t = torch.tensor(obs_np, device=device).unsqueeze(0)
            goal, _, _ = agent.act_manager(obs_t)
            action, _, _, _ = agent.act_worker(obs_t, goal)
            return int(action.item())

    for i in range(eval_cfg.episodes):
        out_path = os.path.join(eval_cfg.record_dir, f"eval_ep{i+1}.mp4")
        record_episode(env, act, max_steps=eval_cfg.max_steps, out_path=out_path)
        print(f"Saved video: {out_path}")