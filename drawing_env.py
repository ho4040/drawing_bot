import cairo
import gymnasium as gym
from gymnasium import spaces
import torch
import torchvision.transforms as T
from torchvision.models import vgg19, VGG19_Weights
from PIL import Image
import numpy as np
import random

def random_color():
    """Generate a random color in RGBA format."""
    r = random.random()
    g = random.random()
    b = random.random()
    a = random.random() # Fully opaque
    return r, g, b, a

class BezierDrawingCanvas:
    VGG_SIZE = 224
    MAX_BRUSH_WIDTH = 80
    def __init__(self):
        self.width = BezierDrawingCanvas.VGG_SIZE
        self.height = BezierDrawingCanvas.VGG_SIZE
        self.surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, self.width, self.height )
        self.ctx = cairo.Context(self.surface)
        self.ctx.set_source_rgba(1, 1, 1, 1)
        self.ctx.paint()
    def fill_image(self, pilImage):
        # Convert the PIL image to RGBA format
        if pilImage.mode != 'RGBA':
            pilImage = pilImage.convert('RGBA')
        
        # Convert the PIL image to a NumPy array
        imageData = np.array(pilImage)
        
        # Cairo uses BGRA format, so we need to swap R and B channels
        # imageData = np.dstack((imageData[:,:,2], imageData[:,:,1], imageData[:,:,0], imageData[:,:,3]))

        # Create a Cairo surface from the NumPy array
        cairoImageSurface = cairo.ImageSurface.create_for_data(
            imageData, 
            cairo.FORMAT_ARGB32, 
            self.width, self.height
        )

        # Clear the current context
        self.ctx.set_source_rgba(1, 1, 1, 1)
        self.ctx.paint()

        # Draw the new image
        self.ctx.set_source_surface(cairoImageSurface, 0, 0)
        self.ctx.paint()
    def bezier_interpolation(self, p0, p1, p2, p3, num_points=100):
        points = []
        for t in np.linspace(0, 1, num_points):
            a = (1 - t) ** 3
            b = 3 * t * (1 - t) ** 2
            c = 3 * t ** 2 * (1 - t)
            d = t ** 3
            point = (a * np.array(p0) + b * np.array(p1) + c * np.array(p2) + d * np.array(p3))
            points.append(point)
        return points
    def draw_variable_width_curve(self, points, start_width, end_width, r, g, b, a):
        self.ctx.set_line_cap(cairo.LINE_CAP_ROUND)

        for i in range(1, len(points)):
            t = (i - 1) / (len(points) - 2)  # Normalized position along the curve
            width = start_width + t * (end_width - start_width)  # Linear interpolation

            self.ctx.set_line_width(width)
            self.ctx.set_source_rgba(r, g, b, a)
            
            # Draw each segment individually
            self.ctx.move_to(*points[i - 1])
            self.ctx.line_to(*points[i])
            self.ctx.stroke()
    def get_image_as_numpy_array(self):
        buf = self.surface.get_data()
        image = np.ndarray(shape=(self.height, self.width, 4), dtype=np.uint8, buffer=buf)
        image = image[:, :, :3]  # Alpha 채널 제거
        observation = np.transpose(image, (2, 0, 1))  # 축 변경: HWC to CHW
        return observation
    def save_to_file(self, filename):
        # change ABGR to ARGB
        buf = self.surface.get_data()
        image = np.ndarray(shape=(self.height, self.width, 4), dtype=np.uint8, buffer=buf)
        # image = image[:, :, [0, 1, 2, 3]]
        image = Image.fromarray(image)
        image.save(filename)
        # self.surface.write_to_png(filename)
    
    def clear_surface(self):
        self.ctx.set_source_rgba(1, 1, 1, 1)  # White color
        self.ctx.paint()
    
    def draw_action(self, action):
        # 여기에서는 제공된 UV 좌표를 실제 픽셀 좌표로 스케일링합니다.
        # action value range is [0, 1]
        uv_coords = action[:8].reshape((4, 2))
        control_points = uv_coords * np.array([self.width, self.height])
        start_width, end_width = action[8:10] * BezierDrawingCanvas.MAX_BRUSH_WIDTH  # 픽셀로 너비를 변환합니다. 
        r, g, b, a = action[10:14]  # 색상은 그대로 사용합니다.
        a = max(a, 0.1) # alpha 값이 너무 작으면 안보이므로 최소값을 0.1로 설정합니다.
        num_points = int(action[14] * 100) + 10  # 10에서 110 사이의 점을 생성합니다.
        points = self.bezier_interpolation(*control_points, num_points=num_points)
        self.draw_variable_width_curve(points, start_width, end_width, r, g, b, a)

    def draw_random_strokes(self, num_strokes=1):
        for _ in range(num_strokes):
            # 랜덤 컨트롤 포인트 생성
            p0 = (random.uniform(0, self.width), random.uniform(0, self.height))
            p1 = (random.uniform(0, self.width), random.uniform(0, self.height))
            p2 = (random.uniform(0, self.width), random.uniform(0, self.height))
            p3 = (random.uniform(0, self.width), random.uniform(0, self.height))
            num_points = random.randint(10, 100)
            # 베지어 곡선 점 계산
            points = self.bezier_interpolation(p0, p1, p2, p3, num_points)
            # 랜덤 너비 설정
            start_width = random.uniform(1, 100)
            end_width = random.uniform(1, 100)
            # 랜덤 색상 생성
            color = random_color()
            # 곡선 그리기
            self.draw_variable_width_curve(points, start_width, end_width, *color)

class DrawingEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}
    DIFFICULTY = 1
    @classmethod
    def inc_difficulty(cls):
        cls.DIFFICULTY = cls.DIFFICULTY+1
        print("Difficulty increased to", cls.DIFFICULTY)

    def __init__(self, perceptual_weight=1.0, l2_weight=0.0, max_steps=100):
        super(DrawingEnv, self).__init__()
        self.canvas = BezierDrawingCanvas() # vgg size
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(15,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=255, shape=(3, BezierDrawingCanvas.VGG_SIZE, BezierDrawingCanvas.VGG_SIZE*2), dtype=np.uint8)
        self.perceptual_weight = perceptual_weight
        self.l2_weight = l2_weight
        self.curstep = 0
        self.episode_length = min(max_steps, DrawingEnv.DIFFICULTY + random.randint(0, 10))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.eval().to(self.device)
        for param in self.vgg.parameters():
            param.requires_grad = False
        
        self.set_random_target()
        self.last_loss = self.get_current_loss()
    def set_random_target(self):
        with torch.no_grad():
            self.target = BezierDrawingCanvas() # vgg size
            self.target.draw_random_strokes(DrawingEnv.DIFFICULTY)
            self.update_target_cache()  # 미리 계산해둡니다.
    
    def update_target_cache(self):
        np_array, tensor, tensor_normalized, vgg_features = self.get_feature(self.target)
        self.target_array = np_array
        self.target_tensor = tensor
        self.target_tensor_normalized = tensor_normalized
        self.target_features = vgg_features

    def get_feature(self, canvas:BezierDrawingCanvas):
        with torch.no_grad():
            np_array =  canvas.get_image_as_numpy_array()
            tensor = torch.tensor(np_array).unsqueeze(0).to(self.device)
            tensor_normalized = self.normalize_tensor(tensor.float().div(255))
            vgg_features = self.vgg(tensor_normalized)
        return np_array, tensor, tensor_normalized, vgg_features

    def apply_target_from_canvas(self):
        self.canvas.fill_image(self.target_image)
        self.last_loss = self.get_current_loss()

    def get_observation(self):
        current_obs = self.canvas.get_image_as_numpy_array()  # Already CHW format
        target_obs = self.target_tensor.squeeze(0).cpu().numpy()  # PyTorch Tensor to NumPy array, CHW format
        observation = np.concatenate([target_obs, current_obs], axis=2)  # Concatenate along the channel dimension
        return observation
    
    def get_current_loss(self):
        with torch.no_grad():
            _, _, tensor_normalized, vgg_features = self.get_feature(self.canvas)
            current_tensor_normalized  = tensor_normalized
            current_features = vgg_features
            # Loss 계산
            perceptual_loss = torch.nn.functional.mse_loss(current_features, self.target_features).item()            
            l2_loss = torch.nn.functional.mse_loss(current_tensor_normalized, self.target_tensor_normalized).item()
            loss = self.perceptual_weight * perceptual_loss  + self.l2_weight * l2_loss
            return loss
    
    def step(self, action):
        if self.curstep >= self.episode_length:
            return self.get_observation(), 0, True, False, {}
        # add noise to action
        action = action + np.random.normal(0, 0.05, size=action.shape)
        # convert range [-1, 1] to [0, 1]        
        action = (action + 1) / 2
        self.canvas.draw_action(action)
        new_loss = self.get_current_loss()
        reward = -new_loss
        # reward = self.last_loss - new_loss
        # self.last_loss = new_loss
        self.curstep += 1
        return self.get_observation(), reward, False, False, {} # observation, reward, terminated, truncated, info 
        

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        
        self.set_random_target()
        self.canvas.clear_surface()
        self.curstep = 0        
        self.last_loss = self.get_current_loss()

        return self.get_observation(), {}

    def render(self, mode='human'):
        image_array = self.get_observation()
        image_array = image_array.transpose(1, 2, 0) 
        if mode == 'rgb_array':
            return image_array
        elif mode == 'human':
            img = Image.fromarray(image_array)
            img.show()

    def save_canvas_to_file(self, filename):
        self.canvas.save_to_file(filename)

    def save_target_to_file(self, filename):
        self.target.save_to_file(filename)
        

    def save_observation_to_file(self, filename):
        image_array = self.get_observation()
        image_array = image_array.transpose(1, 2, 0)
        img = Image.fromarray(image_array)
        img.save(filename)

        
    
    def normalize_tensor(self, tensor):
        # VGG 네트워크에 사용되는 평균과 표준편차
        VGG_MEAN = [0.485, 0.456, 0.406]
        VGG_STD = [0.229, 0.224, 0.225]
        mean = torch.tensor(VGG_MEAN).view(1, -1, 1, 1).to(self.device)
        std = torch.tensor(VGG_STD).view(1, -1, 1, 1).to(self.device)
        return (tensor - mean) / std

   
if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env
    env = DrawingEnv(perceptual_weight=1.0, 
                     l2_weight=0.0, 
                     max_steps=100)
    check_env(env, warn=True)
