from drawing_env import DrawingEnv
import os
os.makedirs("./temp/test_env", exist_ok=True)

env = DrawingEnv(perceptual_weight=1.0, l2_weight=0.0, max_steps=100)
observation = env.reset()
done = False
total_reward = 0
step = 0

while not done:
    action = env.action_space.sample()  # Replace with your model's action
    observation, reward, done, truncated, info = env.step(action)
    total_reward += reward
    step += 1
    print(step, total_reward)
    if step > 10:
        break
env.render()  # Optionally render the environment

print(f"Total reward: {total_reward}")
env.save_observation_to_file("./temp/test_env/output.png")
