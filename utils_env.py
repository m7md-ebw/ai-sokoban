import numpy as np
import gym
from stable_baselines3.common.vec_env import VecFrameStack, VecMonitor, VecTransposeImage

class ShapingWrapper(gym.Wrapper):
    def __init__(self, env, dist_coef=0.01, deadlock_pen=-0.02):
        super().__init__(env)
        self.dist_coef = dist_coef
        self.deadlock_pen = deadlock_pen
        self.last_dist = None

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.last_dist = self._total_box_goal_dist()
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        current_dist = self._total_box_goal_dist()
        if self.last_dist is not None:
            reward += self.dist_coef * (self.last_dist - current_dist)
        self.last_dist = current_dist
        if self._box_in_corner():
            reward += self.deadlock_pen
        return obs, reward, done, info

    def _total_box_goal_dist(self):
        try:
            boxes = self.env.unwrapped.boxes
            goals = self.env.unwrapped.targets
            total_dist = 0
            for box in boxes:
                min_dist = min(abs(box[0]-gx) + abs(box[1]-gy) for gx, gy in goals)
                total_dist += min_dist
            return total_dist
        except Exception:
            return 0

    def _box_in_corner(self):
        try:
            walls = self.env.unwrapped.room_fixed
            for bx, by in self.env.unwrapped.boxes:
                if (walls[by-1, bx] and walls[by, bx-1]) or \
                   (walls[by-1, bx] and walls[by, bx+1]) or \
                   (walls[by+1, bx] and walls[by, bx-1]) or \
                   (walls[by+1, bx] and walls[by, bx+1]):
                    if (bx, by) not in self.env.unwrapped.targets:
                        return True
            return False
        except Exception:
            return False
