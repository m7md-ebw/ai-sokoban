# AI Sokoban Agent with RL

This project involves building a reinforcement learning agent that can solve Sokoban puzzles using Proximal Policy Optimization (PPO). The goal is to explore and evaluate improvements such as memory integration (e.g., LSTM) and hierarchical reinforcement learning (HRL) to help the agent handle sparse rewards and complex decision-making in puzzle-solving environments.

This work is part of a graduation project submitted to the Islamic University of Gaza.

## Project Goals
- Implement a PPO agent that learns to play Sokoban
- Enhance the agent's performance by adding memory-based modules
- Apply hierarchical decision-making to improve long-term planning
- Evaluate the models based on success rate, learning speed, and the agent's ability to handle new levels


## Environment Setup
### Requirements
Note: These libraries will be installed in a virtual environment (venv), which is excluded from the repository due to network constraints. Please install them manually using the provided requirements.

- Python 3.10 or higher
- PyTorch
- Stable-Baselines3
- Gym
- Gym-Sokoban

## Quick Run Notes
---------------

1) Clone the repository
   git clone https://github.com/m7md-ebw/ai-sokoban.git

2) Activate your venv (Terminal / Windows PowerShell):
   .\venv\Scripts\activate

3) Install requirements:
   pip install -r requirements.txt

4) Train baseline PPO :
   for example one of the training files : train_ppo.py

5) TensorBoard:
   tensorboard --logdir YourPath
