"""
工具函式集：包含資料載入、光流計算、特徵 Warp 等核心功能
"""
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os
from pathlib import Path
from typing import Tuple, Optional
import zipfile
import json


def extract_scannet_frames(zip_path: str, extract_dir: str, max_frames: Optional[int] = None):
    """
    解壓縮 ScanNet frames zip 檔案
    
    Args:
        zip_path: scannet_frames_25k.zip 的路徑
        extract_dir: 解壓縮目標目錄
        max_frames: 最多解壓縮的幀數 (None 表示全部)
    """
    print(f"解壓縮 {zip_path} 到 {extract_dir}...")
    os.makedirs(extract_dir, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        all_files = zip_ref.namelist()
        
        # 篩選圖片檔案
        image_files = [f for f in all_files if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        if max_frames:
            image_files = image_files[:max_frames]
        
        print(f"共 {len(image_files)} 個檔案待解壓...")
        for i, file in enumerate(image_files):
            zip_ref.extract(file, extract_dir)
            if (i + 1) % 1000 == 0:
                print(f"已解壓 {i + 1}/{len(image_files)} 個檔案...")
    
    print("解壓縮完成！")
    return extract_dir


def get_image_paths(data_dir: str, max_images: Optional[int] = None):
    """
    取得資料目錄中所有圖片的路徑（已排序）
    
    Args:
        data_dir: 圖片目錄
        max_images: 最多讀取的圖片數
    
    Returns:
        排序後的圖片路徑列表
    """
    extensions = ('.jpg', '.jpeg', '.png')
    image_paths = []
    
    for root, dirs, files in os.walk(data_dir):
        # 排除 depth 目錄
        if 'depth' in root.lower():
            continue
        
        for file in files:
            if file.lower().endswith(extensions):
                full_path = os.path.join(root, file)
                # 再次確認路徑中不包含 depth
                if 'depth' not in full_path.lower():
                    image_paths.append(full_path)
    
    # 排序確保時序
    image_paths.sort()
    
    if max_images:
        image_paths = image_paths[:max_images]
    
    print(f"找到 {len(image_paths)} 張圖片（已排除 depth 圖）")
    return image_paths


def load_image(image_path: str, size: Tuple[int, int] = (384, 384)):
    """
    載入並預處理圖片
    
    Args:
        image_path: 圖片路徑
        size: 調整大小 (H, W)
    
    Returns:
        PIL Image 和 Tensor (C, H, W)
    """
    img = Image.open(image_path).convert('RGB')
    img_resized = img.resize((size[1], size[0]))  # PIL uses (W, H)
    
    # 轉換為 tensor [0, 1]
    img_tensor = torch.from_numpy(np.array(img_resized)).permute(2, 0, 1).float() / 255.0
    
    return img_resized, img_tensor


def warp_feature(feature: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """
    使用光流場對特徵進行 Warp (搬移)
    
    Args:
        feature: [B, C, H, W] 或 [B, N, D] 的特徵
        flow: [B, 2, H_flow, W_flow] 的光流場
    
    Returns:
        warped_feature: 與 feature 同形狀
    """
    if feature.dim() == 3:  # [B, N, D] - patch-level features
        # 需要將其 reshape 成 2D grid
        B, N, D = feature.shape
        # 假設是正方形 grid
        H = W = int(np.sqrt(N))
        assert H * W == N, f"無法將 {N} 個 patches 重組為正方形，請確認特徵形狀"
        
        feature_2d = feature.reshape(B, H, W, D).permute(0, 3, 1, 2)  # [B, D, H, W]
        warped_2d = warp_feature_2d(feature_2d, flow)
        return warped_2d.permute(0, 2, 3, 1).reshape(B, N, D)  # 轉回 [B, N, D]
    
    elif feature.dim() == 4:  # [B, C, H, W]
        return warp_feature_2d(feature, flow)
    
    else:
        raise ValueError(f"不支援的特徵維度: {feature.shape}")


def warp_feature_2d(feature: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """
    2D 特徵的 Warp 實作
    
    Args:
        feature: [B, C, H, W]
        flow: [B, 2, H_flow, W_flow]
    
    Returns:
        warped_feature: [B, C, H, W]
    """
    B, C, H, W = feature.shape
    _, _, H_flow, W_flow = flow.shape
    
    # 如果光流解析度與特徵不同，需要調整
    if H != H_flow or W != W_flow:
        flow = F.interpolate(flow, size=(H, W), mode='bilinear', align_corners=False)
        # 同時調整光流的量值
        flow[:, 0, :, :] *= (W / W_flow)
        flow[:, 1, :, :] *= (H / H_flow)
    
    # 建立基礎網格 [B, H, W, 2]
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=feature.device),
        torch.arange(W, device=feature.device),
        indexing='ij'
    )
    grid = torch.stack([grid_x, grid_y], dim=-1).float()  # [H, W, 2]
    grid = grid.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, W, 2]
    
    # 加上光流 (flow 是 [B, 2, H, W])
    flow_permuted = flow.permute(0, 2, 3, 1)  # [B, H, W, 2]
    new_grid = grid + flow_permuted
    
    # 正規化到 [-1, 1] (grid_sample 的要求)
    new_grid[..., 0] = 2.0 * new_grid[..., 0] / (W - 1) - 1.0
    new_grid[..., 1] = 2.0 * new_grid[..., 1] / (H - 1) - 1.0
    
    # 使用 grid_sample 進行 Warp
    warped = F.grid_sample(
        feature, 
        new_grid, 
        mode='bilinear', 
        padding_mode='border',
        align_corners=False
    )
    
    return warped


def compute_optical_flow(img_prev: torch.Tensor, img_curr: torch.Tensor, 
                        model) -> torch.Tensor:
    """
    計算兩幀之間的光流
    
    Args:
        img_prev: [B, 3, H, W] 前一幀 (值域 [0, 1])
        img_curr: [B, 3, H, W] 當前幀 (值域 [0, 1])
        model: RAFT 模型
    
    Returns:
        flow: [B, 2, H, W] 光流場
    """
    # RAFT 需要 [0, 255] 的輸入
    img_prev_255 = (img_prev * 255.0).clamp(0, 255)
    img_curr_255 = (img_curr * 255.0).clamp(0, 255)
    
    with torch.no_grad():
        # RAFT 回傳 list of flows (coarse to fine)
        flow_predictions = model(img_prev_255, img_curr_255)
        flow = flow_predictions[-1]  # 取最精細的預測
    
    return flow


def create_train_val_split(image_paths, train_ratio=0.8, sequence_aware=True):
    """
    建立訓練/驗證集分割
    
    Args:
        image_paths: 所有圖片路徑
        train_ratio: 訓練集比例
        sequence_aware: 是否考慮序列連續性（避免相鄰幀分散在不同集合）
    
    Returns:
        train_paths, val_paths
    """
    total = len(image_paths)
    
    if sequence_aware:
        # 按場景分組（假設路徑中包含 scene_id）
        from collections import defaultdict
        scene_groups = defaultdict(list)
        
        for path in image_paths:
            # 嘗試從路徑提取場景 ID
            parts = Path(path).parts
            scene_id = None
            for part in parts:
                if 'scene' in part.lower():
                    scene_id = part
                    break
            if scene_id is None:
                scene_id = 'default'
            scene_groups[scene_id].append(path)
        
        # 按場景分割
        train_paths = []
        val_paths = []
        for scene_id, paths in scene_groups.items():
            split_idx = int(len(paths) * train_ratio)
            train_paths.extend(paths[:split_idx])
            val_paths.extend(paths[split_idx:])
    else:
        # 簡單分割
        split_idx = int(total * train_ratio)
        train_paths = image_paths[:split_idx]
        val_paths = image_paths[split_idx:]
    
    print(f"訓練集: {len(train_paths)} 張，驗證集: {len(val_paths)} 張")
    return train_paths, val_paths


def save_checkpoint(model, optimizer, epoch, loss, save_path):
    """儲存訓練 checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, save_path)
    print(f"Checkpoint 已儲存至: {save_path}")


def load_checkpoint(model, optimizer, checkpoint_path):
    """載入訓練 checkpoint"""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"已載入 checkpoint (Epoch {epoch}, Loss {loss:.4f})")
    return epoch, loss


class AverageMeter:
    """計算並儲存平均值和當前值"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
