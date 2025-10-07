import argparse
import os
from .train import TrainConfig, train
from .eval import evaluate, EvalConfig


def main():
    parser = argparse.ArgumentParser(description="HRL Sokoban (for 7x7, 2 boxes)")
    parser.add_argument("--run", type=str, choices=["train", "eval"], default="train")
    parser.add_argument("--levels", type=int, default=4, help="Number of seed-based levels to cycle (1..4)")
    parser.add_argument("--total-episodes", type=int, default=200)
    parser.add_argument("--max-steps-per-ep", type=int, default=200)
    parser.add_argument("--obs-size", type=int, default=64)
    parser.add_argument("--manager-interval", type=int, default=6)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--logdir", type=str, default="runs/hrl")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu (defaults to cuda)")
    parser.add_argument("--eval-interval-steps", type=int, default=5000)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--record-dir", type=str, default="videos/hrl")
    parser.add_argument("--load", type=str, default=None, help="Path to checkpoint to load (resume or eval)")
    parser.add_argument("--env-id", type=str, default="auto", help="Environment id")

    args = parser.parse_args()

    if args.run == "train":
        cfg = TrainConfig(
            levels=args.levels,
            total_episodes=args.total_episodes,
            max_steps_per_ep=args.max_steps_per_ep,
            manager_interval=args.manager_interval,
            obs_size=args.obs_size,
            seed=args.seed,
            save_dir=args.save_dir,
            logdir=args.logdir,
            device=args.device,
            eval_interval_steps=args.eval_interval_steps,
            eval_episodes=args.eval_episodes,
            record_dir=args.record_dir,
            load_path=args.load,
            env_id=args.env_id,
        )
        train(cfg)
    else:  # eval
        eval_cfg = EvalConfig(
            episodes=args.eval_episodes,
            obs_size=args.obs_size,
            max_steps=args.max_steps_per_ep,
            record_dir=args.record_dir,
            device=args.device,
            env_id=args.env_id,
        )
        evaluate(args.load, eval_cfg)


if __name__ == "__main__":
    main()
