from dotenv import load_dotenv
import math
load_dotenv(override=True)
     
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
from tqdm import tqdm, trange
import json
import argparse
from gen_bc_dataset import gen_bc_dataset
from drawing_env import DrawingEnv
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from models import DrawingPolicy
import wandb
from drawing_env import BezierDrawingCanvas

class DrawingDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.curr_img_dir = os.path.join(data_dir, 'curr_img')
        self.goal_img_dir = os.path.join(data_dir, 'goal_img')
        self.action_dir = os.path.join(data_dir, 'action')
        
        # 파일 목록 가져오기
        self.files = [f for f in os.listdir(self.curr_img_dir) if f.endswith('.png')]
        
        # 이미지 변환 파이프라인 설정
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # PIL Image -> Tensor (0-1 범위로 정규화)
            transforms.Lambda(self.remove_alpha_channel)  # RGBA -> RGB (알파 채널 제거)
        ])
    
    @staticmethod
    def remove_alpha_channel(x):
        return x[:3]
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        # 이미지 로드
        curr_img = Image.open(os.path.join(self.curr_img_dir, self.files[idx]))
        goal_img = Image.open(os.path.join(self.goal_img_dir, self.files[idx]))
        
        # 이미지 변환 및 정규화
        curr_img = self.transform(curr_img)  # [C, H, W] 형태의 텐서, 0-1 범위
        goal_img = self.transform(goal_img)  # [C, H, W] 형태의 텐서, 0-1 범위
        
        # 액션 로드
        with open(os.path.join(self.action_dir, self.files[idx].replace('.png', '.json')), 'r') as f:
            action_dict = json.load(f)
        
        # 액션을 numpy 배열로 변환
        action = np.zeros(15)
        # 컨트롤 포인트
        for i, point in enumerate(action_dict['control_points']):
            action[i*2:i*2+2] = point
        # 두께
        action[8] = action_dict['start_width']
        action[9] = action_dict['end_width']
        # 색상
        action[10:14] = action_dict['color']
        # 점 개수
        action[14] = (action_dict['num_points'] - 10) / 100.0
        
        return curr_img, goal_img, torch.FloatTensor(action)

def create_combined_image(current_imgs, goal_imgs, pred_imgs, resize_factor=0.5):
    """
    여러 이미지를 결합하여 하나의 이미지로 만드는 함수
    Args:
        current_imgs: 현재 이미지 리스트
        goal_imgs: 목표 이미지 리스트
        pred_imgs: 예측 이미지 리스트
        resize_factor: 이미지 크기 조정 비율 (0.5 = 50% 크기)
    Returns:
        PIL.Image: 모든 이미지가 결합된 하나의 이미지
    """
    sample_images = []
    for i in range(len(current_imgs)):
        # 현재 샘플의 세 이미지를 가로로 연결
        combined_img = np.concatenate([
            current_imgs[i],  # current
            goal_imgs[i],     # goal
            pred_imgs[i]      # predicted
        ], axis=1)
        
        # 이미지 크기 조정
        if resize_factor != 1.0:
            h, w = combined_img.shape[:2]
            new_h, new_w = int(h * resize_factor), int(w * resize_factor)
            combined_img = np.array(Image.fromarray(combined_img).resize((new_w, new_h), Image.Resampling.LANCZOS))
        
        # 이미지에 제목 추가
        img_with_title = Image.new('RGB', (combined_img.shape[1], combined_img.shape[0] + 30), (255, 255, 255))
        img_with_title.paste(Image.fromarray(combined_img), (0, 30))
        
        # 제목 텍스트 추가
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(img_with_title)
        try:
            font = ImageFont.truetype("arial.ttf", int(20 * resize_factor))
        except:
            font = ImageFont.load_default()
        
        # 각 이미지 위에 제목 추가
        draw.text((10, 5), "Current", fill=(0, 0, 0), font=font)
        draw.text((current_imgs[i].shape[1] * resize_factor + 10, 5), "Goal", fill=(0, 0, 0), font=font)
        draw.text((current_imgs[i].shape[1] * 2 * resize_factor + 10, 5), "Predicted", fill=(0, 0, 0), font=font)
        
        sample_images.append(img_with_title)
    
    # 모든 샘플 이미지를 세로로 연결
    total_height = sum(img.height for img in sample_images)
    max_width = max(img.width for img in sample_images)
    
    combined_all = Image.new('RGB', (max_width, total_height), (255, 255, 255))
    y_offset = 0
    for img in sample_images:
        combined_all.paste(img, (0, y_offset))
        y_offset += img.height
    
    return combined_all

def visualize_predictions(model, env, device, num_samples=4, num_initial_strokes=3, num_after_strokes=0, test=False, output_dir=None):
    """
    모델의 예측을 시각화하는 함수
    Args:
        model: 학습된 DrawingPolicy 모델
        env: DrawingEnv 인스턴스
        device: 계산에 사용할 디바이스
        num_samples: 생성할 샘플 수
        num_initial_strokes: 초기 캔버스에 그릴 스트로크 수
        num_after_strokes: 초기 캔버스에 그린 후 추가로 그릴 스트로크 수
        test: 테스트 모드 여부
        output_dir: 테스트 모드일 때 이미지를 저장할 디렉토리
    """
    model.eval()
    with torch.no_grad():
        current_imgs = []
        goal_imgs = []
        pred_imgs = []
        
        for i in range(num_samples):
            # 초기 상태 설정
            env.reset()
            env.canvas.clear_surface()
            
            # 초기 스트로크 그리기
            for _ in range(num_initial_strokes):
                action = env.random_action()
                env.canvas.draw_action(action)
            
            # 현재 캔버스 상태 저장 (복사본 생성)
            current_canvas = env.canvas.get_image_as_numpy_array().copy()  # [3, 224, 224]
            
            # 목표 이미지 생성 (현재 캔버스 + 추가 스트로크)
            target_canvas = BezierDrawingCanvas()
            target_canvas.fill_image(Image.fromarray(np.transpose(current_canvas, (1, 2, 0))))
            
            additional_action = env.random_action()
            target_canvas.draw_action(additional_action)
            
            for _ in range(num_after_strokes):
                additional_action = env.random_action()
                target_canvas.draw_action(additional_action)
            

            goal_canvas = target_canvas.get_image_as_numpy_array().copy()  # [3, 224, 224]
            
            # 모델 입력을 위한 텐서 변환
            current_img = torch.FloatTensor(current_canvas).unsqueeze(0).to(device) / 255.0  # [1, 3, 224, 224]
            goal_img = torch.FloatTensor(goal_canvas).unsqueeze(0).to(device) / 255.0        # [1, 3, 224, 224]
            
            # 모델 예측
            pred_action = model(current_img, goal_img)
            pred_action = pred_action.cpu().numpy()[0]
            
            
            # 예측된 액션 적용
            pred_canvas = BezierDrawingCanvas()
            pred_canvas.fill_image(Image.fromarray(np.transpose(current_canvas, (1, 2, 0))))
            pred_canvas.draw_action(pred_action)
            pred_canvas_img = pred_canvas.get_image_as_numpy_array().copy()  # [3, 224, 224]
            
            # 이미지 저장 (HWC 형식)
            current_imgs.append(current_canvas.transpose(1, 2, 0))  # [224, 224, 3]
            goal_imgs.append(goal_canvas.transpose(1, 2, 0))        # [224, 224, 3]
            pred_imgs.append(pred_canvas_img.transpose(1, 2, 0))    # [224, 224, 3]
        
        # 테스트 모드일 때 이미지 저장
        if test and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            combined_img = create_combined_image(current_imgs, goal_imgs, pred_imgs)
            combined_img.save(os.path.join(output_dir, 'all_samples_combined.png'))
    
    return current_imgs, goal_imgs, pred_imgs

def setup_wandb(project_name, entity_name, run_name):
    """Initialize Weights & Biases logging"""
    wandb.init(
        project=project_name,
        entity=entity_name,
        name=run_name,
        config={
            'project_name': project_name,
            'entity_name': entity_name,
            'run_name': run_name
        }
    )

def setup_training_environment(output_dir, learning_rate):
    """Setup training environment including device, directories, and model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create output directories
    checkpoint_dir = os.path.join(output_dir, 'ckpt')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize environment and model
    env = DrawingEnv()
    model = DrawingPolicy().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    return device, env, model, optimizer, checkpoint_dir

def create_data_loader(data_dir, batch_size):
    """Create data loader for training"""
    dataset = DrawingDataset(data_dir)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Windows에서 멀티프로세싱 문제 해결을 위해 0으로 설정
    )

def log_batch_metrics(wandb, loss, optimizer, global_epoch, difficulty_increase_epochs):
    """Log batch-level metrics to W&B"""
    wandb.log({
        'batch/loss': loss.item(),
        'batch/learning_rate': optimizer.param_groups[0]['lr'],
        'batch/global_epoch': global_epoch,
        'batch/stroke_count': get_previous_stroke_count(global_epoch, difficulty_increase_epochs)
    })

def log_epoch_metrics(wandb, avg_loss, optimizer, global_epoch, difficulty_increase_epochs):
    """Log epoch-level metrics to W&B"""
    wandb.log({
        'epoch/loss': avg_loss,
        'epoch/learning_rate': optimizer.param_groups[0]['lr'],
        'epoch/global_epoch': global_epoch,
        'epoch/prev_stroke_count': get_previous_stroke_count(global_epoch, difficulty_increase_epochs),
        'epoch/after_stroke_count': get_after_stroke_count(global_epoch, difficulty_increase_epochs)
    })

def print_training_progress(global_epoch, total_epochs, current_progress, batch_idx, 
                          train_loader, loss, difficulty_increase_epochs, phase_idx, num_phases):
    """Print training progress information"""
    if global_epoch % 5 == 0 and batch_idx == 0:
        current_epoch = (global_epoch % total_epochs) + 1
        print(f'Phase {phase_idx + 1}/{num_phases} | '
              f'Epoch {current_epoch}/{total_epochs} | '
              f'Global Epoch {global_epoch} | '
              f'Batch {batch_idx}/{len(train_loader)} | '
              f'Loss: {loss.item():.4f} | '
              f'Stroke_count: prev:{get_previous_stroke_count(global_epoch, difficulty_increase_epochs)} after:{get_after_stroke_count(global_epoch, difficulty_increase_epochs)}')

def save_checkpoint(model, checkpoint_dir, global_epoch):
    """Save model checkpoint"""
    checkpoint_path = os.path.join(checkpoint_dir, f'bc_weights_epoch_{global_epoch + 1}.pth')
    torch.save(model.state_dict(), checkpoint_path)
    print(f'Saved checkpoint to {checkpoint_path}')

def train_epoch(model, train_loader, optimizer, device, global_epoch, total_epochs, 
                difficulty_increase_epochs, wandb, phase_idx, num_phases):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    current_epoch = (global_epoch % total_epochs) + 1
    current_progress = (current_epoch / total_epochs) * 100
    
    for batch_idx, (current_imgs, goal_imgs, actions) in enumerate(train_loader):
        current_imgs = current_imgs.to(device)
        goal_imgs = goal_imgs.to(device)
        actions = actions.to(device)
        
        optimizer.zero_grad()
        pred_actions = model(current_imgs, goal_imgs)
        loss = nn.MSELoss()(pred_actions, actions)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        log_batch_metrics(wandb, loss, optimizer, global_epoch, difficulty_increase_epochs)
        print_training_progress(global_epoch, total_epochs, current_progress, batch_idx, 
                              train_loader, loss, difficulty_increase_epochs, phase_idx, num_phases)
    
    return total_loss / len(train_loader)

def train_bc(output_dir, num_epochs, learning_rate, batch_size, visualize_freq, 
            difficulty_increase_epochs, checkpoint_freq=100, start_epoch=0, phase_idx=0, num_phases=1):
    """Main training function for behavioral cloning"""
    device, env, model, optimizer, checkpoint_dir = setup_training_environment(output_dir, learning_rate)
    
    # Training loop
    global_epoch = start_epoch
    total_epochs = num_epochs
    
    
    train_loader = create_data_loader(output_dir, batch_size)
    
    for epoch in range(num_epochs):
        avg_loss = train_epoch(model, train_loader, optimizer, device, global_epoch, 
                             total_epochs, difficulty_increase_epochs, wandb, phase_idx, num_phases)
        
        log_epoch_metrics(wandb, avg_loss, optimizer, global_epoch, difficulty_increase_epochs)
        
        # Visualize predictions based on frequency
        if (global_epoch + 1) % visualize_freq == 0:
            current_imgs, goal_imgs, pred_imgs = visualize_predictions(
                model, env, device, 
                num_samples=3,
                num_initial_strokes=get_previous_stroke_count(global_epoch, difficulty_increase_epochs),
                num_after_strokes=get_after_stroke_count(global_epoch, difficulty_increase_epochs)
            )
            
            combined_img = create_combined_image(current_imgs, goal_imgs, pred_imgs, resize_factor=0.5)
            wandb.log({
                'predictions': wandb.Image(
                    np.array(combined_img),
                    caption=f'Phase {phase_idx + 1}/{num_phases} | Global {global_epoch + 1}'
                )
            })
            print("Visualization saved to W&B")
        
        # Save checkpoint based on frequency
        if (global_epoch + 1) % checkpoint_freq == 0:
            save_checkpoint(model, checkpoint_dir, global_epoch)
        
        global_epoch += 1
    
    
    return model

def run_visualization_test(output_dir):
    """Run visualization test mode"""
    print("Running visualization test...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = DrawingEnv()
    model = DrawingPolicy().to(device)
    
    model_path = os.path.join(output_dir, 'ckpt', 'bc_weights_final.pth')
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path))
    else:
        print("No trained model found. Using untrained model for visualization.")
    
    output_dir = os.path.join(output_dir, 'log', 'visualization_test')
    current_imgs, goal_imgs, pred_imgs = visualize_predictions(
        model, env, device,
        num_samples=3,
        num_initial_strokes=3,
        num_after_strokes=0,
        test=True,
        output_dir=output_dir
    )
    print(f"Visualization test completed. Images saved to {output_dir}")

def get_previous_stroke_count(global_epoch, difficulty_increase_epochs):
    return (global_epoch // difficulty_increase_epochs)

def get_after_stroke_count(global_epoch, difficulty_increase_epochs):
    return math.floor(min((0.5 * global_epoch) // difficulty_increase_epochs, 3))

def run_curriculum_phase(output_dir, phase_idx, num_phases, global_epoch, 
                        num_samples, num_epochs, learning_rate, batch_size, visualize_freq,
                        difficulty_increase_epochs, checkpoint_freq, model):
    
    """Run a single curriculum phase"""
    
    print(f"\nCurriculum Phase {phase_idx+1}/{num_phases}")
    prev_stroke_count = get_previous_stroke_count(global_epoch, difficulty_increase_epochs)
    after_stroke_count = get_after_stroke_count(global_epoch, difficulty_increase_epochs)
    print(f"Generating dataset...prev_stroke_count: {prev_stroke_count} after_stroke_count:{after_stroke_count} Number of samples: {num_samples}")
    gen_bc_dataset(num_samples, prev_stroke_count, after_stroke_count, output_dir)
    
    print("Starting training...")
    model = train_bc(
        output_dir=output_dir,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        visualize_freq=visualize_freq,
        difficulty_increase_epochs=difficulty_increase_epochs,
        checkpoint_freq=checkpoint_freq,
        start_epoch=global_epoch,
        phase_idx=phase_idx,
        num_phases=num_phases
    )
    
    global_epoch += num_epochs
    
    
    
    return model, global_epoch

def main():
    parser = argparse.ArgumentParser(description='Train behavioral cloning model')
    parser.add_argument('--num_samples', type=int, default=300, help='Number of samples to generate')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size for training')
    parser.add_argument('--output_dir', type=str, default='./temp/bc', help='Output directory path')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--curriculum_phases', type=int, default=10, help='Number of curriculum phases')
    parser.add_argument('--difficulty_increase_epochs', type=int, default=500, help='Increase number of strokes every k global epochs')
    parser.add_argument('--visualize_freq', type=int, default=10, help='Visualize predictions every k epochs')
    parser.add_argument('--checkpoint_freq', type=int, default=100, help='Save checkpoint every k epochs')
    parser.add_argument('--test', type=str, choices=['visualize'], help='Run test mode')
    args = parser.parse_args()
    
    if args.test == 'visualize':
        run_visualization_test(args.output_dir)
        return
    
    # Initialize W&B once for the entire curriculum training
    project_name = os.getenv('WANDB_PROJECT', 'drawing_bot')
    entity_name = os.getenv('WANDB_ENTITY')
    run_name = os.getenv('WANDB_RUN_NAME', f'bc_train_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    setup_wandb(project_name, entity_name, run_name)
    
    # 일반 학습 모드
    model = None
    global_epoch = 0
    
    try:
        for phase_idx in range(args.curriculum_phases):
            model, global_epoch = run_curriculum_phase(
                output_dir=args.output_dir,
                phase_idx=phase_idx,
                num_phases=args.curriculum_phases,
                global_epoch=global_epoch,
                num_samples=args.num_samples,
                num_epochs=args.num_epochs,
                learning_rate=args.learning_rate,
                batch_size=args.batch_size,
                visualize_freq=args.visualize_freq,
                difficulty_increase_epochs=args.difficulty_increase_epochs,
                checkpoint_freq=args.checkpoint_freq,
                model=model
            )
        
        # 최종 모델 저장
        torch.save(model.state_dict(), os.path.join(args.output_dir, 'ckpt', 'bc_weights_final.pth'))
        print(f"Training completed. Final model saved as '{os.path.join(args.output_dir, 'ckpt', 'bc_weights_final.pth')}'")
        print(f"To view the logs, run: wandb open {os.path.join(args.output_dir, 'log')}")
    
    finally:
        # Ensure W&B is properly closed
        wandb.finish()

if __name__ == "__main__":
    main() 