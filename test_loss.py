from drawing_env import BezierDrawingCanvas, DrawingEnv
from PIL import Image
import os

env = DrawingEnv(perceptual_weight=0.5, l2_weight=0.5, max_steps=100)

base_image = Image.open("target.png").convert('RGB').resize((BezierDrawingCanvas.VGG_SIZE, BezierDrawingCanvas.VGG_SIZE))
env = DrawingEnv(perceptual_weight=0.5, l2_weight=0.5, max_steps=100)


for j in range(3):
    DrawingEnv.DIFFICULTY = j 
    env.reset()
    # env.canvas.fill_image(base_image)
    # env.target.fill_image(base_image)
    # env.update_target_cache()
    last_loss = env.get_current_loss()
    print("last_loss", last_loss)
    os.makedirs(f"./temp/test_loss/{j}", exist_ok=True)
    env.save_canvas_to_file(f"./temp/test_loss/{j}/canvas.png")
    env.save_target_to_file(f"./temp/test_loss/{j}/target.png")
    env.save_observation_to_file(f"./temp/test_loss/{j}/obs_0.png")
    for i in range(10): # 노이즈가 증가하면서 loss가 증가하는지 확인
        env.canvas.draw_random_strokes(1)
        env.save_observation_to_file(f"./temp/test_loss/{j}/obs_{i+1}.png")
        loss = env.get_current_loss()
        reward = last_loss - loss
        print("loss", loss, "reward", reward)

