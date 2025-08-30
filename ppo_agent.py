import gym
import gym_sokoban
from stable_baselines3 import PPO

# Create environment
env = gym.make("Sokoban-v0")

# Create PPO agent
model = PPO("MlpPolicy", env, verbose=1)

# Train agent
model.learn(total_timesteps=10000)  # You can increase timesteps later

# Save the model
model.save("ppo_sokoban")

# Test the trained agent
obs = env.reset()
done = False
while not done:
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render()

env.close()
print("PPO agent test complete!")
