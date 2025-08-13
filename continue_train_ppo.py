import argparse
import os
import torch
import numpy as np
import gym

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage, VecMonitor, SubprocVecEnv

# Try fast env, fallback to regular
try:
    from gym_sokoban.envs.sokoban_env_fast import SokobanEnvFast as SokobanEnv
except ImportError:
    from gym_sokoban.envs.sokoban_env import SokobanEnv

# --- Pad obs to 160x160 so it matches the model's expected input ---
class PadObsTo160(gym.ObservationWrapper):
    def __init__(self, env, target_hw=(160, 160)):
        super().__init__(env)
        self.target_h, self.target_w = target_hw
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.target_h, self.target_w, 3), dtype=np.uint8
        )

    def observation(self, obs):
        h, w, c = obs.shape
        out = np.zeros((self.target_h, self.target_w, c), dtype=obs.dtype)
        y = (self.target_h - h) // 2
        x = (self.target_w - w) // 2
        out[y:y+h, x:x+w, :] = obs
        return out

def make_sokoban_env(dim=7, boxes=2, max_steps=120):
    def _thunk():
        env = SokobanEnv(dim_room=(dim, dim), num_boxes=boxes, max_steps=max_steps)
        env = PadObsTo160(env, target_hw=(160, 160))
        return env
    return _thunk

def build_vec_env(dim, boxes, max_steps, n_envs, n_stack, seed):
    env = SubprocVecEnv([make_sokoban_env(dim, boxes, max_steps) for _ in range(n_envs)])
    env = VecMonitor(env)
    if n_stack and n_stack > 1:
        env = VecFrameStack(env, n_stack=n_stack)
    env = VecTransposeImage(env)
    print(f"[ENV] dim_room={dim}x{dim}, boxes={boxes}, max_steps={max_steps}, n_envs={n_envs}, n_stack={n_stack}")
    return env

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="ppo_sokoban.zip")  # match enhanced training save name
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--n_envs", type=int, default=4)
    parser.add_argument("--n_stack", type=int, default=4)
    parser.add_argument("--easy_dim", type=int, default=7)
    parser.add_argument("--easy_boxes", type=int, default=2)
    parser.add_argument("--easy_max_steps", type=int, default=120)
    parser.add_argument("--curriculum", type=int, default=0)  # 0 = stay easy, 1 = ramp difficulty
    args = parser.parse_args()

    device = "cuda" if (args.device in ["auto", "cuda"] and torch.cuda.is_available()) else "cpu"
    print("Using device:", device)

    assert os.path.exists(args.model), f"Model file not found: {args.model}"
    print("[MODEL] Loading:", args.model)

    env = build_vec_env(args.easy_dim, args.easy_boxes, args.easy_max_steps,
                        args.n_envs, args.n_stack, args.seed)
    model = PPO.load(args.model, env=env, device=device, print_system_info=False)

    ckpt_cb = CheckpointCallback(save_freq=50_000, save_path="./checkpoints", name_prefix="ppo_sokoban_step")

    if args.curriculum == 0:
        print(f"[TRAIN] Easy mode: {args.easy_dim}x{args.easy_dim}, boxes={args.easy_boxes}, "
              f"max_steps={args.easy_max_steps} for {args.timesteps} timesteps")
        model.learn(total_timesteps=args.timesteps, callback=ckpt_cb, reset_num_timesteps=False)
    else:
        t1 = int(args.timesteps * 0.3)
        t2 = int(args.timesteps * 0.3)
        t3 = args.timesteps - t1 - t2
        stages = [
            {"dim": 7,  "boxes": 2, "max_steps": 120, "ts": t1},
            {"dim": 8,  "boxes": 3, "max_steps": 160, "ts": t2},
            {"dim": 10, "boxes": 4, "max_steps": 200, "ts": t3},
        ]
        for i, s in enumerate(stages, 1):
            print(f"[TRAIN][Stage {i}] {s['dim']}x{s['dim']}, boxes={s['boxes']}, "
                  f"max_steps={s['max_steps']}, timesteps={s['ts']}")
            stage_env = build_vec_env(s["dim"], s["boxes"], s["max_steps"],
                                      args.n_envs, args.n_stack, args.seed + i)
            model.set_env(stage_env)
            model.learn(total_timesteps=s["ts"], callback=ckpt_cb, reset_num_timesteps=False)
            stage_env.close()

    model.save(args.model)
    env.close()
    print("Continue training complete. Model updated:", args.model)

if __name__ == "__main__":
    main()
