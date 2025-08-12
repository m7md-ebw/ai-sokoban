import gym
import gym_sokoban


num_episodes = 5
max_steps = 300  

env = gym.make("Sokoban-v0")

for episode in range(num_episodes):
    obs = env.reset()
    done = False
    total_reward = 0
    steps = 0

    while not done and steps < max_steps:
        env.render(mode="human")
        action = env.action_space.sample()  
        obs, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1

    print(f"Episode {episode+1} finished in {steps} steps with total reward {total_reward}")

env.close()
