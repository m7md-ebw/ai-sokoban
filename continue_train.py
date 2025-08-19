import subprocess, sys

# continue for another 200k (change as you like)
cmd = [sys.executable, "train_sb3_ppo_sokoban.py", "--resume", "--total-steps", "200000"]
raise SystemExit(subprocess.call(cmd))
