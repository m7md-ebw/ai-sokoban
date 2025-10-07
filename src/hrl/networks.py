import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNEncoder(nn.Module):
    def __init__(self, in_channels=3, feat_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, feat_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = x.float() / 255.0
        x = self.conv(x)
        x = self.head(x)
        return x


class PolicyHead(nn.Module):
    def __init__(self, in_dim, n_actions):
        super().__init__()
        self.pi = nn.Linear(in_dim, n_actions)

    def forward(self, z):
        return self.pi(z)


class ValueHead(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.v = nn.Linear(in_dim, 1)

    def forward(self, z):
        return self.v(z)


class HRLWorker(nn.Module):

    def __init__(self, feat_dim: int, goal_dim: int, n_actions: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(feat_dim + goal_dim, 256),
            nn.ReLU(inplace=True),
        )
        self.pi = PolicyHead(256, n_actions)
        self.v = ValueHead(256)

    def forward(self, feat, goal):
        h = torch.cat([feat, goal], dim=-1)
        h = self.fc(h)
        logits = self.pi(h)
        value = self.v(h)
        return logits, value


class HRLManager(nn.Module):

    def __init__(self, feat_dim: int, goal_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
        )
        self.goal_head = nn.Linear(256, goal_dim)
        self.v = ValueHead(256)

    def forward(self, feat):
        h = self.fc(feat)
        goal = self.goal_head(h)
        value = self.v(h)
        return goal, value
