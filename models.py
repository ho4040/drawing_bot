import torch
import torch.nn as nn
import math

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
        x = self.vit(x)
        
        # 액션 값을 [-1.0, 1.0] 범위로 제한
        x = torch.tanh(x)  # tanh는 출력을 [-1.0, 1.0] 범위로 제한
        
        return x 