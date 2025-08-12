import gym
import gym_sokoban

print("Creating environment...")
env = gym.make("Sokoban-v0")  
obs = env.reset()
print("Environment ready. Starting loop...")

for i in range(100):
    print(f"Step {i}")
    env.render(mode="human")
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    if done:
        print("Game Over")
        break

env.close()
print("Finished.")
