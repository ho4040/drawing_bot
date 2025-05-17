import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
import json
import math
import argparse
from gen_bc_dataset import gen_bc_dataset
from drawing_env import DrawingEnv
from torchvision import transforms

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=6, d_model=256, nhead=8, 
                 num_layers=6, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        
        # Patch Embedding
        self.patch_embed = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)
        num_patches = (img_size // patch_size) ** 2
        
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 15)  # 15차원 액션 벡터
        )
        
        # CLS 토큰
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
    def forward(self, x):
        # x: [B, C, H, W]
        B = x.shape[0]
        
        # Patch Embedding
        x = self.patch_embed(x)  # [B, d_model, H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, d_model]
        
        # CLS 토큰 추가
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        
        # Positional Encoding
        x = self.pos_encoder(x)
        
        # Transformer Encoder
        x = self.transformer_encoder(x)
        
        # CLS 토큰만 사용하여 액션 예측
        x = x[:, 0]
        
        # Classification head
        x = self.classifier(x)
        
        return x

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
            transforms.Lambda(lambda x: x[:3])  # RGBA -> RGB (알파 채널 제거)
        ])
    
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

class DrawingPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = VisionTransformer(
            img_size=224,
            patch_size=16,
            in_channels=6,  # curr_img(3) + goal_img(3)
            d_model=256,
            nhead=8,
            num_layers=6,
            dim_feedforward=1024,
            dropout=0.1
        )
    
    def forward(self, curr_img, goal_img):
        # 현재 캔버스와 목표 이미지를 채널 방향으로 연결
        x = torch.cat([curr_img, goal_img], dim=1)
        return self.vit(x)

def visualize_predictions(model, dataloader, device, loop_idx, save_dir):
    """현재 이미지, 목표 이미지, 예측 스트로크 적용 이미지를 시각화하여 저장"""
    model.eval()
    with torch.no_grad():
        # 5개의 샘플 가져오기
        samples = []
        for i, (curr_imgs, goal_imgs, actions) in enumerate(dataloader):
            if i >= 5:  # 5개 샘플만 사용
                break
            samples.append((curr_imgs, goal_imgs, actions))
        
        # 각 샘플에 대해 시각화
        all_images = []
        for curr_imgs, goal_imgs, actions in samples:
            curr_imgs = curr_imgs.to(device)
            goal_imgs = goal_imgs.to(device)
            
            # 예측된 액션
            pred_actions = model(curr_imgs, goal_imgs)
            
            # 이미지 변환 (텐서 -> numpy -> PIL)
            curr_img = curr_imgs[0].cpu().numpy().transpose(1, 2, 0)  # CHW -> HWC
            goal_img = goal_imgs[0].cpu().numpy().transpose(1, 2, 0)  # CHW -> HWC
            
            # 예측된 스트로크 적용
            env = DrawingEnv()
            env.canvas.clear_surface()
            env.canvas.fill_image(Image.fromarray((curr_img * 255).astype(np.uint8)))
            pred_action = pred_actions[0].cpu().numpy()
            env.canvas.draw_action(pred_action)
            pred_img = env.canvas.get_image_as_numpy_array().transpose(1, 2, 0)  # CHW -> HWC
            
            # 이미지 정규화 (0-1 -> 0-255)
            curr_img = (curr_img * 255).astype(np.uint8)
            goal_img = (goal_img * 255).astype(np.uint8)
            pred_img = (pred_img * 255).astype(np.uint8)
            
            # 이미지들을 가로로 연결
            combined_img = np.concatenate([curr_img, goal_img, pred_img], axis=1)
            all_images.append(combined_img)
        
        # 모든 샘플의 이미지를 세로로 연결
        final_image = np.concatenate(all_images, axis=0)
        
        # 이미지 저장
        save_path = os.path.join(save_dir, f'generation_loop_{loop_idx+1}.png')
        Image.fromarray(final_image).save(save_path)

def train_bc(data_dir, num_epochs=100, batch_size=32, learning_rate=1e-4, model=None):
    """
    Train the behavioral cloning model.
    
    Args:
        data_dir (str): Directory containing the dataset
        num_epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        learning_rate (float): Learning rate for optimizer
        model (DrawingPolicy, optional): Pre-trained model to continue training
    """
    # ./temp/bc/ckpt 디렉토리 없으면 생성
    ckpt_dir = os.path.join(data_dir, 'ckpt')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    
    # 시각화 결과 저장 디렉토리 생성
    log_dir = os.path.join(data_dir, 'log')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 데이터셋 및 데이터로더 설정
    dataset = DrawingDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 모델 및 옵티마이저 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    if model is None:
        model = DrawingPolicy().to(device)
    else:
        model = model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    criterion = nn.MSELoss()
    
    # 학습률 스케줄러
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # 학습 루프
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for curr_imgs, goal_imgs, actions in progress_bar:
            curr_imgs = curr_imgs.to(device)
            goal_imgs = goal_imgs.to(device)
            actions = actions.to(device)
            
            # 액션 예측
            pred_actions = model(curr_imgs, goal_imgs)
            
            # 손실 계산
            loss = criterion(pred_actions, actions)
            
            # 역전파 및 최적화
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{total_loss / (progress_bar.n + 1):.4f}"})
        
        # 학습률 조정
        scheduler.step()
        
        # 100 에폭마다 모델 저장
        if (epoch + 1) % 100 == 0:
            torch.save(model.state_dict(), f'./temp/bc/ckpt/bc_weights_epoch_{epoch+1}.pth')
    
    return model

def main():
    parser = argparse.ArgumentParser(description='Train behavioral cloning model')
    parser.add_argument('--num_samples', type=int, default=300, help='Number of samples to generate')
    parser.add_argument('--batch_size', type=int, default=300, help='Batch size for training')
    parser.add_argument('--output_dir', type=str, default='./temp/bc', help='Output directory path')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--generation_loop_count', type=int, default=10, help='Loop count for generating dataset')
    parser.add_argument('--strokes_increase_schedule', type=int, default=2, help='Each k generation loop, increase the number of strokes')
    args = parser.parse_args()
    
    model = None
    stroke_count = 0
    for i in range(args.generation_loop_count):
        if i % args.strokes_increase_schedule == 0:            
            stroke_count += 1            
            
        print(f"\nGeneration loop {i+1}/{args.generation_loop_count}")
        # 데이터셋 재 생성
        print(f"Generating dataset...Stroke count: {stroke_count} Number of samples: {args.num_samples}")
        data_dir = args.output_dir
        gen_bc_dataset(args.num_samples, stroke_count, data_dir)
        
        # 학습 수행
        print("Starting training...")
        model = train_bc(
            data_dir=data_dir,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            model=model
        )
        
        # 학습 완료 후 시각화
        print("Generating visualization...")
        dataset = DrawingDataset(data_dir)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        visualize_predictions(model, dataloader, torch.device('cuda' if torch.cuda.is_available() else 'cpu'), i, os.path.join(data_dir, 'log'))
    
    # 최종 모델 저장
    torch.save(model.state_dict(), 'bc_weights_final.pth')
    print(f"Training completed. Final model saved as 'bc_weights_final.pth'")

if __name__ == "__main__":
    main() 