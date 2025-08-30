import numpy as np
import gym

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
                min_dist = min(abs(box[0] - gx) + abs(box[1] - gy) for gx, gy in goals)
                total_dist += min_dist
            return total_dist
        except:
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
        except:
            return False

class PadObsToHW(gym.ObservationWrapper):
    def __init__(self, env, target_hw=(112, 112)):
        super().__init__(env)
        self.target_h, self.target_w = target_hw
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.target_h, self.target_w, 3), dtype=np.uint8
        )

    def observation(self, obs):
        # obs is H x W x C (uint8)
        h, w, c = obs.shape
        out = np.zeros((self.target_h, self.target_w, c), dtype=obs.dtype)

        # center-pad
        y = (self.target_h - h) // 2
        x = (self.target_w - w) // 2

        y0, y1 = max(0, y), max(0, y) + min(h, self.target_h)
        x0, x1 = max(0, x), max(0, x) + min(w, self.target_w)

        # crop if source is larger than target (rare, but safe)
        src_y0, src_y1 = max(0, -y), max(0, -y) + (y1 - y0)
        src_x0, src_x1 = max(0, -x), max(0, -x) + (x1 - x0)

        out[y0:y1, x0:x1, :] = obs[src_y0:src_y1, src_x0:src_x1, :]
        return out