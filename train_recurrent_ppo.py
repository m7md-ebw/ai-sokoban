# Optional: PPO with LSTM via sb3-contrib (RecurrentPPO)
import argparse, torch
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback
from utils_env import make_training_env

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = "cuda" if (args.device in ["auto","cuda"] and torch.cuda.is_available()) else "cpu"
    env = make_training_env("Sokoban-v0", n_envs=1, n_stack=1, seed=args.seed)  # Recurrent policy usually doesn't need frame stacking

    model = RecurrentPPO(
        policy="MlpLstmPolicy",  # CNN+LSTM is more complex; start with MLP+LSTM using processed obs if you add preprocessing
        env=env,
        verbose=1,
        device=device,
        seed=args.seed,
    )

    ckpt_cb = CheckpointCallback(save_freq=50_000, save_path="./checkpoints", name_prefix="rppo_sokoban_step")
    model.learn(total_timesteps=args.timesteps, callback=ckpt_cb)
    model.save("rppo_sokoban")
    env.close()
    print("Recurrent PPO training complete.")

if __name__ == "__main__":
    main()
