Quick Run Notes
---------------
1) Activate your venv (Windows PowerShell):
   .\venv\Scripts\activate

2) Install requirements:
   pip install -r requirements.txt

3) Train baseline PPO (with checkpoints + eval + TensorBoard):
   python train_ppo_enhanced.py

4) Continue training from latest checkpoint (optional):
   python continue_train_ppo.py

5) Evaluate a saved model:
   python evaluate_ppo.py --model ppo_sokoban.zip --episodes 5 --max_steps 500 --deterministic 1

6) Record a short video rollout (saved to ./videos):
   python record_video.py --model ppo_sokoban.zip --steps 500 --deterministic 1

7) TensorBoard:
   tensorboard --logdir=ppo_sokoban_tensorboard
