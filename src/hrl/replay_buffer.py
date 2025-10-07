from dataclasses import dataclass
from typing import List
import torch

@dataclass
class WorkerStep:
    obs: torch.Tensor
    goal: torch.Tensor
    action: torch.Tensor
    logp: torch.Tensor
    value: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor

@dataclass
class ManagerStep:
    feat: torch.Tensor
    goal: torch.Tensor
    value: torch.Tensor
    reward: torch.Tensor

class RolloutBuffers:
    def __init__(self):
        self.worker: List[WorkerStep] = []
        self.manager: List[ManagerStep] = []

    def add_worker(self, *args, **kwargs):
        self.worker.append(WorkerStep(*args, **kwargs))

    def add_manager(self, *args, **kwargs):
        self.manager.append(ManagerStep(*args, **kwargs))

    def clear(self):
        self.worker.clear()
        self.manager.clear()

    def compute_returns_advantages(self, gamma=0.99, lam=0.95, device="cpu"):
        R = torch.tensor(0.0, device=device)
        returns, advantages = [], []
        values = torch.stack([s.value for s in self.worker]).squeeze(-1)
        rewards = [s.reward for s in self.worker]
        dones = [s.done for s in self.worker]

        for r, d in zip(reversed(rewards), reversed(dones)):
            R = r.to(device) + gamma * R * (1.0 - d.to(device))
            returns.insert(0, R)
        returns_t = torch.stack(returns).detach()

        deltas = returns_t - values
        A = torch.zeros(1, device=device)
        for dlt, d in zip(reversed(deltas), reversed(dones)):
            A = dlt + gamma * lam * A * (1.0 - d.to(device))
            advantages.insert(0, A)
        adv_t = torch.stack(advantages).detach()

        if len(self.manager) > 0:
            m_values = torch.stack([m.value for m in self.manager]).squeeze(-1)
            m_rewards = torch.stack([m.reward for m in self.manager]).to(device)
            m_returns = m_rewards
        else:
            m_values = torch.tensor([]).to(device)
            m_returns = torch.tensor([]).to(device)

        return returns_t, adv_t, m_returns, m_values
