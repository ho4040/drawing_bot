from drawing_env import DrawingEnv
import os
os.makedirs("./temp/test_env", exist_ok=True)

env = DrawingEnv(perceptual_weight=1.0, l2_weight=0.0, max_steps=100)
total_reward = 0

for j in range(5):
    env.reset()
    DrawingEnv.inc_difficulty()
    step = 0
    done = False
    while not done:
        action = env.action_space.sample()  # Replace with your model's action
        observation, reward, done, truncated, info = env.step(action)
        total_reward += reward
        step += 1        
        print(f"{j}th episode Step: {step}")
        

    print(f"Total reward of {j}th episode: {total_reward}")
    env.save_observation_to_file(f"./temp/test_env/output_{j}.png")
