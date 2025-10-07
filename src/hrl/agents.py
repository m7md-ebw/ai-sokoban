from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from .networks import CNNEncoder, HRLManager, HRLWorker
from .replay_buffer import RolloutBuffers


@dataclass
class HRLConfig:
    obs_size: int = 64
    feat_dim: int = 128
    goal_dim: int = 32
    manager_interval: int = 6  #shorter for 7x7 (each 6 steps gives a goal)
    gamma: float = 0.99
    lam: float = 0.95
    entropy_coef: float = 0.02
    value_coef: float = 0.5
    lr: float = 2e-4  # lower for stability
    max_grad_norm: float = 0.5
    intrinsic_coef: float = 0.1


class HRLAgent(nn.Module):
    def __init__(self, n_actions: int, cfg: HRLConfig):
        super().__init__()
        self.encoder = CNNEncoder(in_channels=3, feat_dim=cfg.feat_dim)
        self.manager = HRLManager(feat_dim=cfg.feat_dim, goal_dim=cfg.goal_dim)
        self.worker = HRLWorker(feat_dim=cfg.feat_dim, goal_dim=cfg.goal_dim, n_actions=n_actions)
        self.goal_to_feat = nn.Linear(cfg.goal_dim, cfg.feat_dim, bias=False)
        self.cfg = cfg

    def act_worker(self, obs_t, goal_t):
        feat = self.encoder(obs_t)
        logits, value = self.worker(feat, goal_t)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        logp = dist.log_prob(action)
        return action, logp, value, feat

    def act_manager(self, obs_t):
        feat = self.encoder(obs_t)
        goal, value = self.manager(feat)
        return goal.detach(), value, feat


class HRLTrainer:
    def __init__(self, agent: HRLAgent, device: torch.device = torch.device("cuda")):
        self.agent = agent.to(device)
        self.device = device
        self.opt = torch.optim.Adam(self.agent.parameters(), lr=self.agent.cfg.lr)

    def compute_intrinsic_reward(self, feat, goal, next_feat):
        eps = 1e-8
        goal_feat = self.agent.goal_to_feat(goal)
        before = F.cosine_similarity(feat, goal_feat, dim=-1, eps=eps)
        after = F.cosine_similarity(next_feat, goal_feat, dim=-1, eps=eps)
        return (after - before).detach()

    def update_from_rollout(self, buf: RolloutBuffers):
        cfg = self.agent.cfg
        # ===== Worker update =====
        w_returns, w_adv, m_returns, m_values = buf.compute_returns_advantages(
            gamma=cfg.gamma, lam=cfg.lam, device=self.device
        )
        obs_batch = torch.stack([s.obs for s in buf.worker]).to(self.device)
        goal_batch = torch.stack([s.goal for s in buf.worker]).to(self.device)
        act_batch = torch.stack([s.action for s in buf.worker]).to(self.device)

        feat = self.agent.encoder(obs_batch)
        logits, values = self.agent.worker(feat, goal_batch)
        dist = torch.distributions.Categorical(logits=logits)
        logp = dist.log_prob(act_batch)

        advantages = (w_returns - values.squeeze(-1)).detach()
        policy_loss = -(advantages * logp).mean()
        value_loss = F.mse_loss(values.squeeze(-1), w_returns)
        entropy = dist.entropy().mean()
        worker_loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy

        loss = worker_loss

        # Manager update 
        if len(buf.manager) > 0:
            m_feat = torch.stack([m.feat for m in buf.manager]).to(self.device)
            m_goal = torch.stack([m.goal for m in buf.manager]).to(self.device)
            _, m_values_now = self.agent.manager(m_feat)
            m_values_now = m_values_now.squeeze(-1)  
            m_returns = m_returns.view(-1)   
            m_value_loss = F.mse_loss(m_values_now, m_returns)
            loss = loss + cfg.value_coef * m_value_loss
        else:
            m_value_loss = torch.tensor(0.0, device=self.device)

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.parameters(), cfg.max_grad_norm)
        self.opt.step()

        stats = {
            "worker_policy_loss": float(policy_loss.detach().cpu()),
            "worker_value_loss": float(value_loss.detach().cpu()),
            "entropy": float(entropy.detach().cpu()),
            "manager_value_loss": float(m_value_loss.detach().cpu()),
            "total_loss": float(loss.detach().cpu()),
        }
        return stats