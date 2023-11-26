from drawing_env import BezierDrawing, DrawingEnv
from PIL import Image
import os
os.makedirs("./temp/test_loss", exist_ok=True)

perfact_image = Image.open("target.png").convert('RGB').resize((BezierDrawing.VGG_SIZE, BezierDrawing.VGG_SIZE))

env = DrawingEnv(perceptual_weight=0.5, l2_weight=0.5, max_steps=100)
env.drawing.fill_image(perfact_image)
last_loss = env.get_current_loss()

env.save_canvas_to_file("./temp/test_loss/canvas.png")
env.save_target_to_file("./temp/test_loss/env.png")
env.save_observation_to_file("./temp/test_loss/obs_0.png")

print("last_loss", last_loss)

for i in range(10): # 노이즈가 증가하면서 loss가 증가하는지 확인
    env.drawing.draw_random_strokes(1)
    env.save_observation_to_file(f"./temp/test_loss/obs_{i+1}.png")
    loss = env.get_current_loss()
    reward = last_loss - loss
    print("loss", loss, "reward", reward)

