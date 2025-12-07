
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import cv2
import os
import random
import argparse
from datetime import datetime

from models_unified import UnifiedTempoVLM, UnifiedLoss, get_model_info

class ScanNetUnifiedDataset(Dataset):

    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        max_scenes: int = 100,
        frames_per_scene: int = 50,
        tasks: list = ['temporal', 'depth_regression'],
    ):
        self.data_root = Path(data_root)
        self.frames_per_scene = frames_per_scene
        self.tasks = tasks
        
        scenes_dir = self.data_root / 'scannet_frames_25k'
        all_scenes = sorted([d for d in scenes_dir.iterdir() if d.is_dir()])
        
        split_idx = int(len(all_scenes) * 0.8)
        if split == 'train':
            self.scenes = all_scenes[:split_idx][:max_scenes]
        else:
            self.scenes = all_scenes[split_idx:][:max_scenes // 5]
        

        self.samples = []
        self._collect_samples()
        
        print(f"[{split}] {len(self.scenes)} scenes, {len(self.samples)} samples")
        print(f"  Tasks: {tasks}")
    
    def _collect_samples(self):
        for scene_dir in tqdm(self.scenes, desc="collect samples"):
            color_dir = scene_dir / 'color'
            depth_dir = scene_dir / 'depth'
            pose_dir = scene_dir / 'pose'
            
            if not color_dir.exists():
                continue
            
            color_files = sorted(color_dir.glob('*.jpg'))[:self.frames_per_scene]
            
            for i in range(len(color_files) - 1):
                sample = {
                    'color1': color_files[i],
                    'color2': color_files[i + 1],
                    'scene': scene_dir.name,
                    'frame_idx': i,
                }
                
                if 'depth_order' in self.tasks or 'depth_regression' in self.tasks:
                    depth1 = depth_dir / (color_files[i].stem + '.png')
                    if depth1.exists():
                        sample['depth1'] = depth1
                
                if 'motion' in self.tasks:
                    pose1 = pose_dir / (color_files[i].stem + '.txt')
                    pose2 = pose_dir / (color_files[i + 1].stem + '.txt')
                    if pose1.exists() and pose2.exists():
                        sample['pose1'] = pose1
                        sample['pose2'] = pose2
                
                self.samples.append(sample)
    
    def __len__(self):
        return len(self.samples)
    
    def _load_depth(self, path):
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            return None
        return depth.astype(np.float32) / 1000.0
    
    def _load_pose(self, path):
        try:
            pose = np.loadtxt(str(path))
            return pose.reshape(4, 4)
        except:
            return None
    
    def _compute_relative_motion(self, pose1, pose2):
        if pose1 is None or pose2 is None:
            return None
        
        # 位移
        t1 = pose1[:3, 3]
        t2 = pose2[:3, 3]
        translation = t2 - t1
        
        # 旋轉
        R1 = pose1[:3, :3]
        R2 = pose2[:3, :3]
        R_rel = R2 @ R1.T
        
        # 歐拉角
        rotation = np.array([
            np.arctan2(R_rel[2, 1], R_rel[2, 2]),
            np.arctan2(-R_rel[2, 0], np.sqrt(R_rel[2, 1]**2 + R_rel[2, 2]**2)),
            np.arctan2(R_rel[1, 0], R_rel[0, 0])
        ])
        
        return np.concatenate([translation, rotation])
    
    def _sample_depth_regions(self, depth, image):
        if depth is None:
            return None, None, None
        
        h, w = depth.shape
        margin = 48
        
        for _ in range(30):
            y1 = random.randint(margin, h - margin)
            x1 = random.randint(margin, w - margin)
            y2 = random.randint(margin, h - margin)
            x2 = random.randint(margin, w - margin)
            
            if abs(y1 - y2) < 40 and abs(x1 - x2) < 40:
                continue
            
            region_a = depth[y1-24:y1+24, x1-24:x1+24]
            region_b = depth[y2-24:y2+24, x2-24:x2+24]
            
            valid_a = region_a[region_a > 0.1]
            valid_b = region_b[region_b > 0.1]
            
            if len(valid_a) > 50 and len(valid_b) > 50:
                depth_a = valid_a.mean()
                depth_b = valid_b.mean()
                
                if abs(depth_a - depth_b) > 0.2:
                    img_array = np.array(image)
                    crop_a = image.crop((
                        max(0, x1-32), max(0, y1-32),
                        min(w, x1+32), min(h, y1+32)
                    )).resize((64, 64))
                    crop_b = image.crop((
                        max(0, x2-32), max(0, y2-32),
                        min(w, x2+32), min(h, y2+32)
                    )).resize((64, 64))
                    
                    label = 0 if depth_a < depth_b else 1  # 0: A較近
                    return crop_a, crop_b, label
        
        return None, None, None
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # load images
        image1 = Image.open(sample['color1']).convert('RGB')
        image2 = Image.open(sample['color2']).convert('RGB')
        
        result = {
            'image1': image1,
            'image2': image2,
            'scene': sample['scene'],
        }
        
        # depth order
        if 'depth_order' in self.tasks and 'depth1' in sample:
            depth = self._load_depth(sample['depth1'])
            crop_a, crop_b, label = self._sample_depth_regions(depth, image1)
            result['region_a'] = crop_a
            result['region_b'] = crop_b
            result['depth_order_label'] = label
        
        # depth regression - 輸出 3 個區域的深度 [left, center, right]
        if 'depth_regression' in self.tasks and 'depth1' in sample:
            depth = self._load_depth(sample['depth1'])
            if depth is not None:
                h, w = depth.shape
                
                # 定義三個區域
                regions = {
                    'left': depth[:, :w//3],
                    'center': depth[:, w//3:2*w//3],
                    'right': depth[:, 2*w//3:]
                }
                
                depths = []
                valid_count = 0
                
                for name in ['left', 'center', 'right']:
                    region = regions[name]
                    valid = region[(region > 0.1) & (region < 10.0)]
                    if len(valid) > 100:
                        avg_depth = valid.mean()
                        depths.append(avg_depth)  # 直接使用米為單位
                        valid_count += 1
                    else:
                        depths.append(0.0)  # 無效區域標記為 0
                
                # 只有當至少 2 個區域有效時才使用
                if valid_count >= 2:
                    result['depth_regression_label'] = np.array(depths, dtype=np.float32)

        # motion prediction
        if 'motion' in self.tasks and 'pose1' in sample:
            pose1 = self._load_pose(sample['pose1'])
            pose2 = self._load_pose(sample['pose2'])
            motion = self._compute_relative_motion(pose1, pose2)
            result['motion_label'] = motion
        
        return result


# ============================================================
# GRU 序列訓練用的 Dataset
# ============================================================

class ScanNetSequenceDataset(Dataset):
    """
    用於 GRU 長期記憶訓練的序列 Dataset
    返回連續的幀序列而非單獨的幀對
    
    支援遮擋模擬（Occlusion Augmentation）訓練抗遮擋能力
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        max_scenes: int = 100,
        sequence_length: int = 8,  # 每個序列的幀數
        stride: int = 4,  # 序列之間的間隔
        tasks: list = ['temporal', 'depth_regression', 'motion'],
        # 遮擋模擬參數
        occlusion_prob: float = 0.3,         # 每幀被遮擋的機率
        occlusion_ratio_range: tuple = (0.3, 0.6),  # 遮擋面積比例範圍
        max_consecutive_occlusion: int = 3,  # 最多連續遮擋幾幀
    ):
        self.data_root = Path(data_root)
        self.sequence_length = sequence_length
        self.stride = stride
        self.tasks = tasks
        self.split = split
        
        # 遮擋模擬設定
        self.occlusion_prob = occlusion_prob if split == 'train' else 0.0
        self.occlusion_ratio_range = occlusion_ratio_range
        self.max_consecutive_occlusion = max_consecutive_occlusion
        
        scenes_dir = self.data_root / 'scannet_frames_25k'
        all_scenes = sorted([d for d in scenes_dir.iterdir() if d.is_dir()])
        
        split_idx = int(len(all_scenes) * 0.8)
        if split == 'train':
            self.scenes = all_scenes[:split_idx][:max_scenes]
        else:
            self.scenes = all_scenes[split_idx:][:max_scenes // 5]
        
        self.sequences = []
        self._collect_sequences()
        
        print(f"[{split}] {len(self.scenes)} scenes, {len(self.sequences)} sequences")
        print(f"  Sequence length: {sequence_length}, Stride: {stride}")
        print(f"  Tasks: {tasks}")
    
    def _collect_sequences(self):
        """收集所有有效的連續幀序列"""
        for scene_dir in tqdm(self.scenes, desc="Collecting sequences"):
            color_dir = scene_dir / 'color'
            depth_dir = scene_dir / 'depth'
            pose_dir = scene_dir / 'pose'
            
            if not color_dir.exists():
                continue
            
            color_files = sorted(color_dir.glob('*.jpg'))
            
            # 使用滑動窗口收集序列
            for start_idx in range(0, len(color_files) - self.sequence_length + 1, self.stride):
                sequence_frames = []
                valid_sequence = True
                
                for i in range(self.sequence_length):
                    frame_idx = start_idx + i
                    color_path = color_files[frame_idx]
                    
                    frame_info = {
                        'color': color_path,
                        'scene': scene_dir.name,
                        'frame_idx': frame_idx,
                    }
                    
                    # 檢查深度圖
                    depth_path = depth_dir / (color_path.stem + '.png')
                    if depth_path.exists():
                        frame_info['depth'] = depth_path
                    
                    # 檢查 pose（需要當前幀和下一幀的 pose 來計算 motion）
                    pose_path = pose_dir / (color_path.stem + '.txt')
                    if pose_path.exists():
                        frame_info['pose'] = pose_path
                    
                    # 下一幀的 pose（用於 motion 計算）
                    if i < self.sequence_length - 1:
                        next_color = color_files[frame_idx + 1]
                        next_pose_path = pose_dir / (next_color.stem + '.txt')
                        if next_pose_path.exists():
                            frame_info['next_pose'] = next_pose_path
                    
                    sequence_frames.append(frame_info)
                
                if valid_sequence and len(sequence_frames) == self.sequence_length:
                    self.sequences.append({
                        'scene': scene_dir.name,
                        'frames': sequence_frames,
                    })
    
    def __len__(self):
        return len(self.sequences)
    
    def _load_depth(self, path):
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            return None
        return depth.astype(np.float32) / 1000.0
    
    def _load_pose(self, path):
        try:
            pose = np.loadtxt(str(path))
            return pose.reshape(4, 4)
        except:
            return None
    
    def _compute_relative_motion(self, pose1, pose2):
        if pose1 is None or pose2 is None:
            return None
        
        t1 = pose1[:3, 3]
        t2 = pose2[:3, 3]
        translation = t2 - t1
        
        R1 = pose1[:3, :3]
        R2 = pose2[:3, :3]
        R_rel = R2 @ R1.T
        
        rotation = np.array([
            np.arctan2(R_rel[2, 1], R_rel[2, 2]),
            np.arctan2(-R_rel[2, 0], np.sqrt(R_rel[2, 1]**2 + R_rel[2, 2]**2)),
            np.arctan2(R_rel[1, 0], R_rel[0, 0])
        ])
        
        return np.concatenate([translation, rotation])
    
    def _get_depth_regions(self, depth):
        """獲取三個區域的平均深度 [left, center, right]"""
        if depth is None:
            return None
        
        h, w = depth.shape
        regions = {
            'left': depth[:, :w//3],
            'center': depth[:, w//3:2*w//3],
            'right': depth[:, 2*w//3:]
        }
        
        depths = []
        valid_count = 0
        
        for name in ['left', 'center', 'right']:
            region = regions[name]
            valid = region[(region > 0.1) & (region < 10.0)]
            if len(valid) > 100:
                depths.append(valid.mean())
                valid_count += 1
            else:
                depths.append(0.0)
        
        if valid_count >= 2:
            return np.array(depths, dtype=np.float32)
        return None
    
    def _apply_occlusion(self, image):
        """
        對圖像應用隨機遮擋（訓練時的數據增強）
        
        返回:
            occluded_image: 遮擋後的圖像
            is_occluded: 是否被遮擋
        """
        if random.random() > self.occlusion_prob:
            return image, False
        
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        cx, cy = w // 2, h // 2
        
        # 隨機遮擋比例
        ratio = random.uniform(*self.occlusion_ratio_range)
        size = int(min(w, h) * ratio / 2)
        
        # 隨機遮擋類型
        occ_type = random.choice(['black', 'white', 'noise', 'blur'])
        
        if occ_type == 'black':
            img_array[cy-size:cy+size, cx-size:cx+size] = 0
        elif occ_type == 'white':
            img_array[cy-size:cy+size, cx-size:cx+size] = 255
        elif occ_type == 'noise':
            noise = np.random.randint(0, 255, (size*2, size*2, 3), dtype=np.uint8)
            img_array[cy-size:cy+size, cx-size:cx+size] = noise
        elif occ_type == 'blur':
            roi = img_array[cy-size:cy+size, cx-size:cx+size]
            blurred = cv2.GaussianBlur(roi, (51, 51), 0)
            img_array[cy-size:cy+size, cx-size:cx+size] = blurred
        
        return Image.fromarray(img_array), True
    
    def _generate_occlusion_pattern(self, seq_len):
        """
        生成序列的遮擋模式
        確保連續遮擋不超過 max_consecutive_occlusion
        """
        pattern = []
        consecutive = 0
        
        for i in range(seq_len):
            if consecutive >= self.max_consecutive_occlusion:
                # 強制不遮擋
                pattern.append(False)
                consecutive = 0
            elif random.random() < self.occlusion_prob:
                pattern.append(True)
                consecutive += 1
            else:
                pattern.append(False)
                consecutive = 0
        
        return pattern

    def __getitem__(self, idx):
        """
        返回一個完整的序列（支援遮擋模擬）
        """
        sequence = self.sequences[idx]
        
        # 生成遮擋模式
        occlusion_pattern = self._generate_occlusion_pattern(self.sequence_length)
        
        result = {
            'scene': sequence['scene'],
            'images': [],  # List of PIL Images (可能被遮擋)
            'images_clean': [],  # List of PIL Images (原始未遮擋，用於計算 GT)
            'is_occluded': [],  # Boolean mask，標記哪些幀被遮擋
            'depth_regression_labels': [],  # List of [3] arrays
            'motion_labels': [],  # List of [6] arrays
            'valid_depth': [],  # Boolean mask
            'valid_motion': [],  # Boolean mask
        }
        
        for i, frame_info in enumerate(sequence['frames']):
            # 載入圖像
            image = Image.open(frame_info['color']).convert('RGB')
            
            # 保存原始圖像
            result['images_clean'].append(image)
            
            # 根據遮擋模式決定是否遮擋
            if occlusion_pattern[i]:
                occluded_image, _ = self._apply_occlusion(image)
                result['images'].append(occluded_image)
                result['is_occluded'].append(True)
            else:
                result['images'].append(image)
                result['is_occluded'].append(False)
            
            # 深度回歸標籤（使用原始圖像的深度）
            if 'depth_regression' in self.tasks and 'depth' in frame_info:
                depth = self._load_depth(frame_info['depth'])
                depth_label = self._get_depth_regions(depth)
                if depth_label is not None:
                    result['depth_regression_labels'].append(depth_label)
                    result['valid_depth'].append(True)
                else:
                    result['depth_regression_labels'].append(np.zeros(3, dtype=np.float32))
                    result['valid_depth'].append(False)
            else:
                result['depth_regression_labels'].append(np.zeros(3, dtype=np.float32))
                result['valid_depth'].append(False)
            
            # Motion 標籤（除了最後一幀）
            if 'motion' in self.tasks and i < len(sequence['frames']) - 1:
                if 'pose' in frame_info and 'next_pose' in frame_info:
                    pose1 = self._load_pose(frame_info['pose'])
                    pose2 = self._load_pose(frame_info['next_pose'])
                    motion = self._compute_relative_motion(pose1, pose2)
                    if motion is not None:
                        result['motion_labels'].append(motion.astype(np.float32))
                        result['valid_motion'].append(True)
                    else:
                        result['motion_labels'].append(np.zeros(6, dtype=np.float32))
                        result['valid_motion'].append(False)
                else:
                    result['motion_labels'].append(np.zeros(6, dtype=np.float32))
                    result['valid_motion'].append(False)
        
        # 轉換為 numpy arrays
        result['depth_regression_labels'] = np.stack(result['depth_regression_labels'])  # [T, 3]
        if result['motion_labels']:
            result['motion_labels'] = np.stack(result['motion_labels'])  # [T-1, 6]
        result['valid_depth'] = np.array(result['valid_depth'])
        result['valid_motion'] = np.array(result['valid_motion'])
        result['is_occluded'] = np.array(result['is_occluded'])
        
        return result


def sequence_collate(batch):
    """
    序列 Dataset 的 collate 函數
    由於序列長度固定，可以正常 batch
    支援遮擋模擬訓練
    """
    batch_size = len(batch)
    seq_len = len(batch[0]['images'])
    
    result = {
        'scene': [b['scene'] for b in batch],
        'images': [],  # [T][B] list of lists (可能被遮擋)
        'images_clean': [],  # [T][B] list of lists (原始未遮擋)
        'is_occluded': torch.stack([
            torch.tensor(b['is_occluded']) for b in batch
        ]),  # [B, T]
        'depth_regression_labels': torch.stack([
            torch.tensor(b['depth_regression_labels']) for b in batch
        ]),  # [B, T, 3]
        'valid_depth': torch.stack([
            torch.tensor(b['valid_depth']) for b in batch
        ]),  # [B, T]
    }
    
    # 重組 images: 從 [B][T] 到 [T][B]
    for t in range(seq_len):
        result['images'].append([b['images'][t] for b in batch])
        result['images_clean'].append([b['images_clean'][t] for b in batch])
    
    # Motion labels (長度為 T-1)
    if batch[0]['motion_labels'] is not None and len(batch[0]['motion_labels']) > 0:
        result['motion_labels'] = torch.stack([
            torch.tensor(b['motion_labels']) for b in batch
        ])  # [B, T-1, 6]
        result['valid_motion'] = torch.stack([
            torch.tensor(b['valid_motion']) for b in batch
        ])  # [B, T-1]
    
    return result


def custom_collate(batch):
    result = {
        'image1': [b['image1'] for b in batch],
        'image2': [b['image2'] for b in batch],
        'scene': [b['scene'] for b in batch],
    }
    
    if 'region_a' in batch[0]:
        valid_depth = [(b['region_a'], b['region_b'], b['depth_order_label'])
                       for b in batch if b['region_a'] is not None]
        if valid_depth:
            result['region_a'] = [v[0] for v in valid_depth]
            result['region_b'] = [v[1] for v in valid_depth]
            result['depth_order_label'] = torch.tensor([v[2] for v in valid_depth])
    
    if 'depth_regression_label' in batch[0]:
        valid_depth_reg = [b['depth_regression_label'] for b in batch 
                          if b.get('depth_regression_label') is not None]
        if valid_depth_reg:
            # 確保是 numpy array 並堆疊成 [B, 3]
            stacked = np.stack(valid_depth_reg, axis=0)
            result['depth_regression_label'] = torch.tensor(stacked, dtype=torch.float32)
    
    if 'motion_label' in batch[0]:
        valid_motion = [b['motion_label'] for b in batch if b['motion_label'] is not None]
        if valid_motion:
            result['motion_label'] = torch.tensor(np.stack(valid_motion), dtype=torch.float32)
    
    return result


class SceneDiverseBatchSampler(torch.utils.data.Sampler):
    """
    確保每個 batch 來自不同場景的 Sampler
    對比學習專用
    """
    def __init__(self, dataset, batch_size, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        
        # 按場景分組樣本
        self.scene_to_indices = {}
        for idx, sample in enumerate(dataset.samples):
            scene = sample['scene']
            if scene not in self.scene_to_indices:
                self.scene_to_indices[scene] = []
            self.scene_to_indices[scene].append(idx)
        
        self.scenes = list(self.scene_to_indices.keys())
        self.num_scenes = len(self.scenes)
        
    def __iter__(self):
        # 為每個場景打亂其內部樣本
        scene_iters = {}
        for scene in self.scenes:
            indices = self.scene_to_indices[scene].copy()
            random.shuffle(indices)
            scene_iters[scene] = iter(indices)
        
        # 輪流從不同場景採樣
        while True:
            batch = []
            used_scenes = set()
            
            # 打亂場景順序
            scenes_shuffled = self.scenes.copy()
            random.shuffle(scenes_shuffled)
            
            for scene in scenes_shuffled:
                if len(batch) >= self.batch_size:
                    break
                
                # 從這個場景取一個樣本
                try:
                    idx = next(scene_iters[scene])
                    batch.append(idx)
                    used_scenes.add(scene)
                except StopIteration:
                    # 這個場景的樣本用完了，重新打亂
                    indices = self.scene_to_indices[scene].copy()
                    random.shuffle(indices)
                    scene_iters[scene] = iter(indices)
                    try:
                        idx = next(scene_iters[scene])
                        batch.append(idx)
                        used_scenes.add(scene)
                    except StopIteration:
                        continue
            
            if len(batch) == self.batch_size:
                yield batch
            elif len(batch) > 0 and not self.drop_last:
                yield batch
            else:
                break
    
    def __len__(self):
        # 估計 batch 數量
        total_samples = len(self.dataset)
        if self.drop_last:
            return total_samples // self.batch_size
        else:
            return (total_samples + self.batch_size - 1) // self.batch_size



class UnifiedTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 解析任務
        if 'all' in args.tasks:
            self.tasks = ['temporal', 'depth_order', 'depth_regression', 'motion']
        else:
            self.tasks = args.tasks
        
        print(f"Training tasks: {self.tasks}")
        
        print("\n載入 Qwen2-VL...")
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct",
            trust_remote_code=True
        )
        self.qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct",
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        ).eval()
        
        for param in self.qwen_model.parameters():
            param.requires_grad = False
        
        print("\ncreate UnifiedTempoVLM...")
        self.model = UnifiedTempoVLM(
            feat_dim=args.feat_dim,
            hidden_dim=args.hidden_dim,
            use_transformer_encoder=True,   # ✅ 使用 Transformer（默認已啟用）
            num_encoder_layers=2,           # 可調整：1-3 層
            num_heads=8,                    # 可調整：4-12 個 head
        )
        
        if args.pretrained and not args.no_pretrained:
            print(f"\ntry to load pretrained weights: {args.pretrained}")
            try:
                self.model.load_pretrained_temporal(args.pretrained)
            except Exception as e:
                print(f"⚠️ 載入預訓練權重失敗: {e}")
                print("   將從頭訓練所有參數")
        else:
            print("\ntrain from scratch")
        
        if args.freeze_temporal:
            print("\nforze temporal branch...")
            for name, param in self.model.named_parameters():
                if 'temporal' in name or 'shared_encoder' in name:
                    param.requires_grad = False
                    print(f"  forzen: {name}")
        
        self.model = self.model.to(self.device).float()

        # model info
        info = get_model_info(self.model)
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\nmodel weights: {info['total_params']:,} (trainable: {trainable:,})")

        # loss function（使用固定權重，更穩定）
        self.loss_fn = UnifiedLoss(
            num_tasks=5,
            use_uncertainty_weighting=False,  # 🔥 關閉自動權重
            task_weights={
                'temporal': 0.1,          # InfoNCE loss 很大，降低權重
                'depth_order': 1.0,       
                'depth_regression': 3.0,  # 🔥 重點任務
                'motion': 2.0,            # 🔥 重點任務
                'scene_class': 0.5,       
                'occlusion_recon': 1.5,   
                'memory_quality_reg': 0.5,
            }
        )

        # optimizer（只優化模型參數）
        all_params = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        # 如果使用 uncertainty weighting 才加入 loss 參數
        if hasattr(self.loss_fn, 'log_vars'):
            all_params += list(self.loss_fn.parameters())
        
        self.optimizer = torch.optim.AdamW(
            all_params,
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.epochs
        )
        
        # dataset
        self.train_dataset = ScanNetUnifiedDataset(
            args.data_root, 'train',
            max_scenes=args.max_scenes,
            frames_per_scene=args.frames_per_scene,
            tasks=self.tasks
        )
        self.val_dataset = ScanNetUnifiedDataset(
            args.data_root, 'val',
            max_scenes=args.max_scenes,
            frames_per_scene=args.frames_per_scene,
            tasks=self.tasks
        )
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=custom_collate
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=custom_collate
        )
        
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Resume training
        self.start_epoch = 0
        self.best_loss = float('inf')
        
        if args.resume:
            self._load_checkpoint(args.resume, args.resume_epoch)
    
    def _load_checkpoint(self, checkpoint_path, resume_epoch=None):
        """載入 checkpoint 繼續訓練"""
        print(f"\n📥 載入 checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print("  ✅ 模型權重已載入")
        else:
            self.model.load_state_dict(checkpoint)
            print("  ✅ 模型權重已載入 (直接格式)")
        
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("  ✅ 優化器狀態已載入")
        
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("  ✅ 學習率調度器已載入")
        
        if resume_epoch is not None:
            self.start_epoch = resume_epoch
        elif 'epoch' in checkpoint:
            self.start_epoch = checkpoint['epoch'] + 1
        else:
            import re
            match = re.search(r'epoch_?(\d+)', checkpoint_path)
            if match:
                self.start_epoch = int(match.group(1)) + 1
        
        if 'best_loss' in checkpoint:
            self.best_loss = checkpoint['best_loss']
            print(f"  ✅ 最佳 loss: {self.best_loss:.4f}")
        
        print(f"  ✅ 將從 epoch {self.start_epoch} 繼續訓練")
    
    def extract_features(self, images):
        """提取特徵"""
        features = []
        for image in images:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Describe."}
                ]
            }]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=[text], images=[image],
                padding=True, return_tensors="pt"
            )
            inputs = {k: v.to(self.device) if hasattr(v, 'to') else v
                      for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.qwen_model(**inputs, output_hidden_states=True)
                feat = outputs.hidden_states[-1].mean(dim=1).float()
                features.append(feat)
        
        return torch.cat(features, dim=0)
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        loss_history = {task: [] for task in self.tasks}
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        batch_count = 0
        for batch in pbar:
            # Debug: 顯示前 3 個 batch 的場景（僅在 epoch 1）
            if epoch == 1 and batch_count < 3:
                scenes = batch['scene']
                print(f"\n[Batch {batch_count}] Scenes: {scenes}")
                if len(set(scenes)) < len(scenes):
                    print(f"  ⚠️ Warning: Only {len(set(scenes))} unique scenes in batch of {len(scenes)}")
            batch_count += 1
            
            self.optimizer.zero_grad()
            
            # feature extraction
            feat1 = self.extract_features(batch['image1'])
            feat2 = self.extract_features(batch['image2'])
            
            region_a_feat = None
            region_b_feat = None
            if 'region_a' in batch and batch['region_a']:
                region_a_feat = self.extract_features(batch['region_a'])
                region_b_feat = self.extract_features(batch['region_b'])
            
            # forwarding
            outputs, _ = self.model(
                curr_feat=feat2,
                prev_feat=feat1,
                region_a_feat=region_a_feat,
                region_b_feat=region_b_feat,
                tasks=self.tasks
            )
            
            targets = {}
            if 'depth_order_label' in batch:
                targets['depth_order'] = batch['depth_order_label'].to(self.device)
            if 'depth_regression_label' in batch:
                targets['depth_regression'] = batch['depth_regression_label'].to(self.device)
            if 'motion_label' in batch:
                targets['motion'] = batch['motion_label'].to(self.device)
            
            # loss calculation
            loss, loss_dict = self.loss_fn(outputs, targets, feat1)
            
            if loss > 0:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            
            total_loss += loss.item()
            
            # 只記錄任務的 loss，忽略權重相關的 keys
            for task, l in loss_dict.items():
                if task in loss_history:  # 只記錄我們初始化的任務
                    loss_history[task].append(l)
            
            # 記錄對比學習診斷信息（如果有的話）
            if 'temporal_pos_sim' in loss_dict and 'temporal_neg_sim' in loss_dict:
                if 'temporal_pos_sim' not in loss_history:
                    loss_history['temporal_pos_sim'] = []
                    loss_history['temporal_neg_sim'] = []
                loss_history['temporal_pos_sim'].append(loss_dict['temporal_pos_sim'])
                loss_history['temporal_neg_sim'].append(loss_dict['temporal_neg_sim'])
            
            desc = f"Epoch {epoch} | "
            for task in self.tasks:
                if loss_history[task]:
                    avg_loss = np.mean(loss_history[task][-20:])
                    desc += f"{task[:4]}:{avg_loss:.4f} "
                    
                    # 為 depth_order 添加準確率估計（更直觀）
                    if task == 'depth_order' and avg_loss > 0:
                        est_acc = 100 * (1 - min(avg_loss / 0.693, 1.0))  # 粗略估計
                        desc += f"(~{est_acc:.0f}%) "
            
            # 顯示對比學習診斷信息（如果在訓練 temporal）
            if 'temporal' in self.tasks and 'temporal_pos_sim' in loss_history:
                if loss_history['temporal_pos_sim']:
                    pos_sim = np.mean(loss_history['temporal_pos_sim'][-20:])
                    neg_sim = np.mean(loss_history['temporal_neg_sim'][-20:])
                    desc += f"| pos:{pos_sim:.3f} neg:{neg_sim:.3f} "
                    
                    # 顯示當前 batch 的場景（驗證多樣性）
                    batch_scenes = set(batch['scene'])
                    if len(batch_scenes) < len(batch['scene']):
                        desc += f"⚠️ {len(batch_scenes)} scenes "
            
            pbar.set_description(desc)
        
        self.scheduler.step()
        
        return total_loss / len(self.train_loader), loss_history
    
    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        
        metrics = {
            'temporal_consistency': [],
            'depth_order_acc': [],
            'motion_error': [],
            'rotation_error': [],
            'motion_scale_ratio': [],
        }
        
        depth_correct = 0
        depth_total = 0
        
        for batch in tqdm(self.val_loader, desc="Evaluating"):
            feat1 = self.extract_features(batch['image1'])
            feat2 = self.extract_features(batch['image2'])
            
            # Temporal Consistency(need better metrics)
            if 'temporal' in self.tasks:
                outputs, _ = self.model(feat2, feat1, tasks=['temporal'])
                refined = outputs['temporal']
                consistency = F.cosine_similarity(refined, feat1, dim=-1).mean()
                metrics['temporal_consistency'].append(consistency.item())
            
            # depth order accuracy
            if 'depth_order' in self.tasks and 'region_a' in batch and batch['region_a']:
                region_a_feat = self.extract_features(batch['region_a'])
                region_b_feat = self.extract_features(batch['region_b'])
                
                outputs, _ = self.model(
                    feat2, feat1,
                    region_a_feat=region_a_feat,
                    region_b_feat=region_b_feat,
                    tasks=['depth_order']
                )
                
                pred = outputs['depth_order'].argmax(dim=-1)
                gt = batch['depth_order_label'].to(self.device)
                depth_correct += (pred == gt).sum().item()
                depth_total += len(gt)
            
            # motion error
            if 'motion' in self.tasks and 'motion_label' in batch:
                outputs, _ = self.model(feat2, feat1, tasks=['motion'])
                pred = outputs['motion']
                gt = batch['motion_label'].to(self.device)
                
                # 平移誤差 (只看 xyz)
                trans_error = (pred[:, :3] - gt[:, :3]).abs().mean()
                # 旋轉誤差 (弧度)
                rot_error = (pred[:, 3:] - gt[:, 3:]).abs().mean()
                
                metrics['motion_error'].append(trans_error.item())
                metrics['rotation_error'].append(rot_error.item())
                
                # 計算 scale 比例（用於診斷）
                pred_scale = pred[:, :3].abs().mean()
                gt_scale = gt[:, :3].abs().mean()
                if gt_scale > 1e-6:
                    scale_ratio = (pred_scale / gt_scale).item()
                    metrics['motion_scale_ratio'].append(scale_ratio)
        
        results = {}
        if metrics['temporal_consistency']:
            results['temporal_consistency'] = np.mean(metrics['temporal_consistency'])
        if depth_total > 0:
            results['depth_order_acc'] = depth_correct / depth_total
        if metrics['motion_error']:
            results['motion_mae'] = np.mean(metrics['motion_error'])
            results['rotation_mae'] = np.mean(metrics['rotation_error'])
            if metrics['motion_scale_ratio']:
                results['motion_scale_ratio'] = np.mean(metrics['motion_scale_ratio'])
        
        return results
    
    def train(self):
        best_metric = 0 if self.best_loss == float('inf') else -self.best_loss
        history = []
        
        total_epochs = self.args.epochs
        start_epoch = self.start_epoch
        
        if start_epoch > 0:
            print(f"\n從 epoch {start_epoch} 繼續訓練，總共訓練到 epoch {total_epochs}")
        
        for epoch in range(start_epoch + 1, total_epochs + 1):
            train_loss, loss_history = self.train_epoch(epoch)
            val_results = self.evaluate()
            
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{total_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Validation:")
            for k, v in val_results.items():
                print(f"    {k}: {v:.4f}")

            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"  Learning Rate: {current_lr:.2e}")
            
            # 顯示自動學習的 Loss 權重
            task_weights = self.loss_fn.get_task_weights()
            if task_weights:
                print(f"  Auto Task Weights:")
                for task, weight in task_weights.items():
                    print(f"    {task}: {weight:.4f}")
            
            history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'lr': current_lr,
                **val_results,
                **{f'weight_{k}': v for k, v in task_weights.items()}
            })
            
            metric = val_results.get('temporal_consistency', 0) + \
                     val_results.get('depth_order_acc', 0)
            
            if metric > best_metric:
                best_metric = metric
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'loss_fn_state_dict': self.loss_fn.state_dict(),  # 保存 Loss 權重
                    'best_loss': train_loss,
                    'val_results': val_results,
                    'tasks': self.tasks,
                }, self.output_dir / 'best_unified_model.pt')
                print(f"  ✅ 儲存最佳模型")
            
            save_every = getattr(self.args, 'save_every', 5)
            if epoch % save_every == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'loss_fn_state_dict': self.loss_fn.state_dict(),  # 保存 Loss 權重
                    'best_loss': train_loss,
                    'tasks': self.tasks,
                }, self.output_dir / f'checkpoint_epoch{epoch}.pt')
                print(f"  💾 儲存 checkpoint: epoch {epoch}")
        

        history_path = self.output_dir / 'training_history.json'
        
        if history_path.exists() and start_epoch > 0:
            with open(history_path, 'r') as f:
                old_history = json.load(f)
            old_history = [h for h in old_history if h['epoch'] < start_epoch + 1]
            history = old_history + history
        
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

        print(f"\n training complete! save the result to: {self.output_dir}")


# ============================================================
# GRU 序列訓練專用 Trainer
# ============================================================

class GRUSequenceTrainer:
    """
    支援 GRU 長期記憶的序列訓練器
    使用 Truncated BPTT 來訓練長序列
    """
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # GRU 訓練專用任務（不包含 depth_order，因為它不需要時序）
        self.tasks = ['temporal', 'depth_regression', 'motion']
        print(f"GRU Training tasks: {self.tasks}")
        
        # 載入 Qwen2-VL
        print("\n載入 Qwen2-VL...")
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct",
            trust_remote_code=True
        )
        self.qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct",
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        ).eval()
        
        for param in self.qwen_model.parameters():
            param.requires_grad = False
        
        # 創建帶 GRU 的模型
        print("\n創建 UnifiedTempoVLM (with GRU memory)...")
        self.model = UnifiedTempoVLM(
            feat_dim=args.feat_dim,
            hidden_dim=args.hidden_dim,
            use_gru_memory=True,            # 啟用 GRU
            use_transformer_encoder=True,   # ✅ 使用 Transformer
            num_encoder_layers=2,           # 可調整層數
            num_heads=8,                    # 可調整 head 數
        )
        self.model = self.model.to(self.device).float()
        
        # 模型資訊
        info = get_model_info(self.model)
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\n模型參數: {info['total_params']:,} (可訓練: {trainable:,})")
        
        # Loss function（使用固定權重，更穩定）
        self.loss_fn = UnifiedLoss(
            num_tasks=5,
            use_uncertainty_weighting=False,  # 🔥 關閉自動權重，使用手動調校的權重
            task_weights={
                'temporal': 0.1,          # InfoNCE loss 很大，降低權重
                'depth_order': 1.0,       
                'depth_regression': 3.0,  # 🔥 重點任務
                'motion': 2.0,            # 🔥 重點任務
                'scene_class': 0.5,       
                'occlusion_recon': 1.5,   
                'memory_quality_reg': 0.5,
            }
        )
        
        # Optimizer（只優化模型參數，loss 不需要）
        all_params = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        # 如果使用 uncertainty weighting 才加入 loss 參數
        if hasattr(self.loss_fn, 'log_vars'):
            all_params += list(self.loss_fn.parameters())
        
        self.optimizer = torch.optim.AdamW(
            all_params,
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.epochs
        )
        
        # 是否啟用遮擋模擬
        enable_occlusion = not getattr(args, 'no_occlusion_aug', False)
        occlusion_prob = getattr(args, 'occlusion_prob', 0.3) if enable_occlusion else 0.0
        occlusion_ratio_range = (
            getattr(args, 'occlusion_ratio_min', 0.3),
            getattr(args, 'occlusion_ratio_max', 0.6)
        ) if enable_occlusion else (0.3, 0.6)
        max_consecutive_occlusion = getattr(args, 'max_consecutive_occlusion', 3) if enable_occlusion else 3
        
        # 序列 Dataset（訓練集啟用遮擋模擬）
        self.train_dataset = ScanNetSequenceDataset(
            args.data_root, 'train',
            max_scenes=args.max_scenes,
            sequence_length=args.sequence_length,
            stride=args.stride,
            tasks=self.tasks,
            occlusion_prob=occlusion_prob,
            occlusion_ratio_range=occlusion_ratio_range,
            max_consecutive_occlusion=max_consecutive_occlusion
        )
        # 驗證集不使用遮擋模擬
        self.val_dataset = ScanNetSequenceDataset(
            args.data_root, 'val',
            max_scenes=args.max_scenes,
            sequence_length=args.sequence_length,
            stride=args.stride,
            tasks=self.tasks,
            occlusion_prob=0.0  # 驗證集不遮擋
        )
        
        # 由於序列訓練的記憶體需求較大，batch_size 通常要小一些
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=sequence_collate
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=sequence_collate
        )
        
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.start_epoch = 0
        self.best_loss = float('inf')
        
        # ========== 對比學習：Feature Bank ==========
        # 維護一個跨序列的負樣本 bank
        self.feature_bank_size = 256  # 存儲最近 256 個特徵
        self.feature_bank = None  # [bank_size, feat_dim]
        self.feature_bank_ptr = 0
        # 🔥 提高溫度參數，降低 InfoNCE loss 的絕對值
        # 原本 0.07 → loss ≈ 3-4
        # 現在 0.2 → loss ≈ 1-2
        self.contrastive_temperature = 0.2
        print(f"✅ 對比學習已啟用 (Feature Bank Size: {self.feature_bank_size}, Temperature: {self.contrastive_temperature})")
    
    @torch.no_grad()
    def update_feature_bank(self, features):
        """更新 feature bank（用於對比學習的負樣本）"""
        features = features.detach()
        batch_size = features.shape[0]
        feat_dim = features.shape[1]
        
        if self.feature_bank is None:
            self.feature_bank = torch.zeros(self.feature_bank_size, feat_dim, device=self.device)
        
        # 環形緩衝區更新
        if self.feature_bank_ptr + batch_size <= self.feature_bank_size:
            self.feature_bank[self.feature_bank_ptr:self.feature_bank_ptr + batch_size] = features
        else:
            # 需要分兩部分寫入
            first_part = self.feature_bank_size - self.feature_bank_ptr
            self.feature_bank[self.feature_bank_ptr:] = features[:first_part]
            self.feature_bank[:batch_size - first_part] = features[first_part:]
        
        self.feature_bank_ptr = (self.feature_bank_ptr + batch_size) % self.feature_bank_size
    
    def compute_contrastive_loss(self, curr_refined, prev_feat):
        """
        計算 GRU 模式的對比學習 loss
        - 正樣本：當前幀的 refined 特徵 vs 前一幀特徵（時序連續性）
        - 負樣本：來自 feature bank 的其他序列特徵
        """
        batch_size = curr_refined.shape[0]
        
        # 正則化
        curr_norm = F.normalize(curr_refined, p=2, dim=-1)  # [B, D]
        prev_norm = F.normalize(prev_feat, p=2, dim=-1)     # [B, D]
        
        # 正樣本相似度（對角線）
        pos_sim = (curr_norm * prev_norm).sum(dim=-1)  # [B]
        
        # 負樣本相似度（來自 feature bank）
        if self.feature_bank is not None and self.feature_bank_ptr > 0:
            # 取出有效的 bank 特徵
            valid_size = min(self.feature_bank_ptr, self.feature_bank_size)
            bank_features = self.feature_bank[:valid_size]  # [N, D]
            bank_norm = F.normalize(bank_features, p=2, dim=-1)
            
            # 計算與 bank 的相似度
            neg_sim = curr_norm @ bank_norm.T  # [B, N]
            
            # InfoNCE loss
            tau = self.contrastive_temperature
            pos_exp = torch.exp(pos_sim / tau)  # [B]
            neg_exp = torch.exp(neg_sim / tau)  # [B, N]
            neg_sum = neg_exp.sum(dim=1)        # [B]
            
            contrastive_loss = -torch.log(pos_exp / (pos_exp + neg_sum + 1e-8)).mean()
            
            # 診斷信息
            with torch.no_grad():
                avg_pos_sim = pos_sim.mean().item()
                avg_neg_sim = neg_sim.mean().item()
        else:
            # Bank 還沒填滿，使用簡單的相似度 loss
            contrastive_loss = 1 - pos_sim.mean()
            avg_pos_sim = pos_sim.mean().item()
            avg_neg_sim = 0.0
        
        return contrastive_loss, avg_pos_sim, avg_neg_sim
    
    def extract_features(self, images):
        """提取單幀特徵"""
        features = []
        for image in images:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Describe."}
                ]
            }]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=[text], images=[image],
                padding=True, return_tensors="pt"
            )
            inputs = {k: v.to(self.device) if hasattr(v, 'to') else v
                      for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.qwen_model(**inputs, output_hidden_states=True)
                feat = outputs.hidden_states[-1].mean(dim=1).float()
                features.append(feat)
        
        return torch.cat(features, dim=0)
    
    def train_epoch(self, epoch):
        """
        GRU 序列訓練的一個 epoch
        使用 Truncated BPTT：
        1. 每個序列從頭開始（hidden_state = None）
        2. 在序列內累積梯度
        3. 序列結束後更新參數
        + 對比學習：使用 Feature Bank 作為負樣本
        + 遮擋感知訓練：使用原始圖像特徵作為監督目標
        """
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        loss_history = {
            'temporal': [],
            'depth_regression': [],
            'motion': [],
            'memory_quality': [],
            'contrastive': [],      # 🆕 對比學習 loss
            'contrastive_pos': [],  # 🆕 正樣本相似度
            'contrastive_neg': [],  # 🆕 負樣本相似度
            'occlusion_recon': [],  # 🆕 遮擋重建 loss
        }
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch in pbar:
            self.optimizer.zero_grad()
            
            batch_size = len(batch['scene'])
            seq_len = len(batch['images'])
            
            # 獲取遮擋信息（如果有的話）
            is_occluded = batch.get('is_occluded', None)  # [B, T]
            has_clean_images = 'images_clean' in batch
            
            # 初始化隱藏狀態（每個新序列開始時重置）
            hidden_state = None
            
            # 累積整個序列的 loss
            seq_loss = 0
            seq_steps = 0
            contrastive_loss_sum = 0
            contrastive_steps = 0
            occlusion_recon_loss_sum = 0
            occlusion_recon_steps = 0
            
            prev_feat = None
            prev_feat_clean = None  # 用於遮擋訓練的原始特徵
            
            # 遍歷序列中的每一幀
            for t in range(seq_len):
                # 提取當前幀的特徵（可能被遮擋）
                curr_feat = self.extract_features(batch['images'][t])  # [B, feat_dim]
                
                # 如果有遮擋訓練，提取原始（未遮擋）圖像特徵
                curr_feat_clean = None
                if has_clean_images:
                    curr_feat_clean = self.extract_features(batch['images_clean'][t])  # [B, feat_dim]
                
                # 準備任務
                tasks_to_run = ['temporal', 'depth_regression']
                if t < seq_len - 1:  # motion 需要下一幀
                    tasks_to_run.append('motion')
                
                # Forward（使用 GRU hidden state）
                outputs, hidden_state = self.model(
                    curr_feat=curr_feat,
                    prev_feat=prev_feat,
                    hidden_state=hidden_state,
                    tasks=tasks_to_run
                )
                
                # 記錄 memory quality（用於監控）
                if 'memory_quality' in outputs:
                    loss_history['memory_quality'].append(outputs['memory_quality'].item())
                
                # ========== 🆕 遮擋重建 Loss ==========
                # 如果當前幀被遮擋，GRU refined 特徵應該接近原始特徵
                if is_occluded is not None and has_clean_images and 'temporal' in outputs:
                    refined_feat = outputs['temporal']  # GRU refined 特徵
                    
                    # 檢查哪些樣本的當前幀被遮擋
                    curr_occluded = is_occluded[:, t].to(self.device)  # [B]
                    
                    if curr_occluded.any():
                        # 只對被遮擋的幀計算重建損失
                        occluded_refined = refined_feat[curr_occluded]  # [N, D]
                        clean_target = curr_feat_clean[curr_occluded]    # [N, D]
                        
                        # L2 重建損失
                        recon_loss = F.mse_loss(occluded_refined, clean_target)
                        
                        # 額外：使用 cosine similarity 確保方向正確
                        cos_sim = F.cosine_similarity(occluded_refined, clean_target, dim=-1)
                        cos_loss = 1 - cos_sim.mean()
                        
                        # 總重建損失（結合 L2 和 cosine）
                        total_recon_loss = recon_loss + 0.5 * cos_loss
                        
                        occlusion_recon_loss_sum = occlusion_recon_loss_sum + total_recon_loss
                        occlusion_recon_steps += 1
                        
                        loss_history['occlusion_recon'].append(total_recon_loss.item())
                
                # ========== 對比學習 Loss ==========
                if 'temporal' in outputs and prev_feat is not None:
                    refined_feat = outputs['temporal']  # GRU refined 特徵
                    
                    # 使用原始特徵作為正樣本目標（如果可用）
                    target_prev_feat = prev_feat_clean if prev_feat_clean is not None else prev_feat
                    
                    # 計算對比學習 loss
                    contrastive_loss, pos_sim, neg_sim = self.compute_contrastive_loss(
                        refined_feat, target_prev_feat
                    )
                    
                    contrastive_loss_sum = contrastive_loss_sum + contrastive_loss
                    contrastive_steps += 1
                    
                    loss_history['contrastive'].append(contrastive_loss.item())
                    loss_history['contrastive_pos'].append(pos_sim)
                    loss_history['contrastive_neg'].append(neg_sim)
                    
                    # 更新 Feature Bank（使用原始特徵，避免污染 bank）
                    if curr_feat_clean is not None:
                        self.update_feature_bank(curr_feat_clean)
                    else:
                        self.update_feature_bank(refined_feat)
                
                # 準備 targets
                targets = {}
                
                # Depth regression targets
                depth_labels = batch['depth_regression_labels'][:, t, :]  # [B, 3]
                valid_depth = batch['valid_depth'][:, t]  # [B]
                if valid_depth.any():
                    valid_depth_device = valid_depth.to(self.device)
                    targets['depth_regression'] = depth_labels[valid_depth].to(self.device)
                    # 需要調整 outputs 也只取有效的
                    if 'depth_regression' in outputs and valid_depth.sum() > 0:
                        outputs['depth_regression'] = outputs['depth_regression'][valid_depth_device]
                
                # Motion targets（只有非最後一幀才有）
                if t < seq_len - 1 and 'motion_labels' in batch:
                    motion_labels = batch['motion_labels'][:, t, :]  # [B, 6]
                    valid_motion = batch['valid_motion'][:, t]  # [B]
                    if valid_motion.any():
                        valid_motion_device = valid_motion.to(self.device)
                        targets['motion'] = motion_labels[valid_motion].to(self.device)
                        if 'motion' in outputs and valid_motion.sum() > 0:
                            outputs['motion'] = outputs['motion'][valid_motion_device]
                            # ⚠️ 同時過濾 motion_log_var
                            if 'motion_log_var' in outputs:
                                outputs['motion_log_var'] = outputs['motion_log_var'][valid_motion_device]
                
                # 計算這一幀的 task loss（不包含 temporal，因為用對比學習取代）
                if targets:
                    # 移除 temporal 輸出，避免重複計算
                    outputs_for_loss = {k: v for k, v in outputs.items() if k != 'temporal'}
                    frame_loss, frame_loss_dict = self.loss_fn(outputs_for_loss, targets, None)
                    if frame_loss > 0:
                        seq_loss = seq_loss + frame_loss
                        seq_steps += 1
                        
                        for task, l in frame_loss_dict.items():
                            if task in loss_history:
                                loss_history[task].append(l)
                
                # 更新 prev_feat（用於下一幀的 motion 計算）
                prev_feat = curr_feat
                prev_feat_clean = curr_feat_clean
                
                # Detach hidden state 以實現 Truncated BPTT
                # 這樣梯度只會在序列內傳播，不會跨序列
                if hidden_state is not None:
                    hidden_state = hidden_state.detach()
            
            # 序列結束，計算平均 loss 並更新參數
            if seq_steps > 0 or contrastive_steps > 0 or occlusion_recon_steps > 0:
                # ============================================================
                # 🔥 修正後的 Loss 權重分配
                # ============================================================
                # 問題分析：
                # - contrastive loss ≈ 3-4（InfoNCE，值很大）
                # - depth loss ≈ 0.15（值小但重要）
                # - motion loss ≈ 0.07（值更小）
                # - occ_recon loss ≈ 0.1-0.2
                #
                # 解決方案：大幅降低對比學習權重，提升 task loss 權重
                # ============================================================
                
                total_seq_loss = torch.tensor(0.0, device=self.device)
                
                # Task losses (depth, motion) - 這些已經在 UnifiedLoss 中加權過
                if seq_steps > 0:
                    task_loss = seq_loss / seq_steps
                    # 🔥 提升 task loss 權重
                    task_weight = 3.0  # 原本是 1.0
                    total_seq_loss = total_seq_loss + task_weight * task_loss
                
                # Contrastive learning loss
                if contrastive_steps > 0:
                    cont_loss = contrastive_loss_sum / contrastive_steps
                    # 🔥 大幅降低對比學習權重（因為 loss 值太大）
                    contrastive_weight = 0.1  # 原本是 1.0
                    total_seq_loss = total_seq_loss + contrastive_weight * cont_loss
                
                # Occlusion reconstruction loss
                if occlusion_recon_steps > 0:
                    occ_loss = occlusion_recon_loss_sum / occlusion_recon_steps
                    # 遮擋重建權重
                    occlusion_recon_weight = 1.5  # 原本是 2.0
                    total_seq_loss = total_seq_loss + occlusion_recon_weight * occ_loss
                
                total_seq_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                total_loss += total_seq_loss.item()
                num_batches += 1
            
            # 更新進度條（包含對比學習和遮擋重建診斷）
            desc = f"Epoch {epoch} | "
            for task in ['depth_regression', 'motion']:
                if loss_history[task]:
                    desc += f"{task[:5]}:{np.mean(loss_history[task][-20:]):.4f} "
            
            # 顯示對比學習信息
            if loss_history['contrastive']:
                cont_loss = np.mean(loss_history['contrastive'][-20:])
                pos_sim = np.mean(loss_history['contrastive_pos'][-20:])
                neg_sim = np.mean(loss_history['contrastive_neg'][-20:])
                desc += f"| CL:{cont_loss:.3f} pos:{pos_sim:.3f} neg:{neg_sim:.3f} "
            
            # 顯示遮擋重建信息
            if loss_history['occlusion_recon']:
                occ_recon = np.mean(loss_history['occlusion_recon'][-20:])
                desc += f"| occ_recon:{occ_recon:.3f} "
            
            if loss_history['memory_quality']:
                desc += f"mem_q:{np.mean(loss_history['memory_quality'][-20:]):.3f}"
            pbar.set_description(desc)
        
        self.scheduler.step()
        
        return total_loss / max(num_batches, 1), loss_history
    
    @torch.no_grad()
    def evaluate(self):
        """評估模型"""
        self.model.eval()
        
        metrics = {
            'temporal_consistency': [],
            'depth_error': [],
            'motion_error': [],
            'memory_quality': [],
        }
        
        for batch in tqdm(self.val_loader, desc="Evaluating"):
            batch_size = len(batch['scene'])
            seq_len = len(batch['images'])
            
            hidden_state = None
            prev_feat = None
            
            for t in range(seq_len):
                curr_feat = self.extract_features(batch['images'][t])
                
                outputs, hidden_state = self.model(
                    curr_feat=curr_feat,
                    prev_feat=prev_feat,
                    hidden_state=hidden_state,
                    tasks=['temporal', 'depth_regression', 'motion'] if t < seq_len - 1 else ['temporal', 'depth_regression']
                )
                
                # Temporal consistency
                if 'temporal' in outputs and prev_feat is not None:
                    consistency = F.cosine_similarity(outputs['temporal'], prev_feat, dim=-1).mean()
                    metrics['temporal_consistency'].append(consistency.item())
                
                # Memory quality
                if 'memory_quality' in outputs:
                    metrics['memory_quality'].append(outputs['memory_quality'].item())
                
                # Depth error
                if 'depth_regression' in outputs:
                    depth_labels = batch['depth_regression_labels'][:, t, :]
                    valid_depth = batch['valid_depth'][:, t]
                    if valid_depth.any():
                        pred = outputs['depth_regression'][valid_depth.to(self.device)]
                        gt = depth_labels[valid_depth].to(self.device)
                        error = (pred - gt).abs().mean()
                        metrics['depth_error'].append(error.item())
                
                # Motion error
                if t < seq_len - 1 and 'motion' in outputs and 'motion_labels' in batch:
                    motion_labels = batch['motion_labels'][:, t, :]
                    valid_motion = batch['valid_motion'][:, t]
                    if valid_motion.any():
                        pred = outputs['motion'][valid_motion.to(self.device)]
                        gt = motion_labels[valid_motion].to(self.device)
                        error = (pred - gt).abs().mean()
                        metrics['motion_error'].append(error.item())
                
                prev_feat = curr_feat
                if hidden_state is not None:
                    hidden_state = hidden_state.detach()
        
        results = {}
        for k, v in metrics.items():
            if v:
                results[k] = np.mean(v)
        
        return results
    
    def train(self):
        """主訓練循環"""
        history = []
        
        print(f"\n{'='*60}")
        print(f"開始 GRU 序列訓練")
        print(f"序列長度: {self.args.sequence_length}")
        print(f"Batch size: {self.args.batch_size}")
        print(f"{'='*60}")
        
        for epoch in range(self.start_epoch + 1, self.args.epochs + 1):
            train_loss, loss_history = self.train_epoch(epoch)
            val_results = self.evaluate()
            
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{self.args.epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Validation:")
            for k, v in val_results.items():
                print(f"    {k}: {v:.4f}")
            
            # 顯示 loss weights
            task_weights = self.loss_fn.get_task_weights()
            if task_weights:
                print(f"  Auto Task Weights:")
                for task, weight in task_weights.items():
                    print(f"    {task}: {weight:.4f}")
            
            history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                **val_results,
            })
            
            # 儲存 checkpoint
            if epoch % self.args.save_every == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'loss_fn_state_dict': self.loss_fn.state_dict(),
                    'best_loss': train_loss,
                }, self.output_dir / f'gru_checkpoint_epoch{epoch}.pt')
                print(f"  💾 儲存 checkpoint: epoch {epoch}")
        
        # 儲存訓練歷史
        with open(self.output_dir / 'gru_training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"\n✅ GRU 訓練完成! 結果儲存於: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Unified Multi-Task Training')
    
    parser.add_argument('--data_root', type=str, default='./scannet_data')
    parser.add_argument('--output_dir', type=str, default='./checkpoints_unified')
    parser.add_argument('--max_scenes', type=int, default=100)
    parser.add_argument('--frames_per_scene', type=int, default=50)
    
    parser.add_argument('--feat_dim', type=int, default=1536)
    parser.add_argument('--hidden_dim', type=int, default=768)
    parser.add_argument('--pretrained', type=str, default=None,
                        help='預訓練時序 Adapter 路徑 (注意: 結構可能不相容)')
    parser.add_argument('--freeze_temporal', action='store_true',
                        help='凍結時序分支')
    parser.add_argument('--no_pretrained', action='store_true',
                        help='不載入預訓練權重，從頭訓練')
    parser.add_argument('--resume', type=str, default=None,
                        help='從 checkpoint 繼續訓練 (載入完整模型+優化器狀態)')
    parser.add_argument('--resume_epoch', type=int, default=None,
                        help='指定從哪個 epoch 開始 (若不指定則自動檢測)')
    
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    
    parser.add_argument('--tasks', type=str, nargs='+',
                        default=['temporal', 'depth_regression'],
                        help='訓練任務: temporal, depth_order, motion, all')
    parser.add_argument('--temporal_weight', type=float, default=1.0)
    parser.add_argument('--depth_order_weight', type=float, default=1.0)
    parser.add_argument('--motion_weight', type=float, default=1.0)
    
    parser.add_argument('--save_every', type=int, default=2,
                        help='每幾個 epoch 儲存一次 checkpoint')
    
    # GRU 訓練相關參數
    parser.add_argument('--use_gru', action='store_true',
                        help='使用 GRU 序列訓練模式')
    parser.add_argument('--sequence_length', type=int, default=8,
                        help='GRU 訓練時的序列長度')
    parser.add_argument('--stride', type=int, default=4,
                        help='序列之間的滑動步長')
    
    # 遮擋模擬訓練參數
    parser.add_argument('--occlusion_prob', type=float, default=0.3,
                        help='訓練時每個序列應用遮擋的機率')
    parser.add_argument('--occlusion_ratio_min', type=float, default=0.3,
                        help='遮擋區域最小比例')
    parser.add_argument('--occlusion_ratio_max', type=float, default=0.6,
                        help='遮擋區域最大比例')
    parser.add_argument('--max_consecutive_occlusion', type=int, default=3,
                        help='最大連續遮擋幀數')
    parser.add_argument('--no_occlusion_aug', action='store_true',
                        help='禁用遮擋數據增強')
    
    args = parser.parse_args()
    
    # 根據模式選擇訓練器
    if args.use_gru:
        print("\n" + "="*60)
        print("🧠 使用 GRU 序列訓練模式")
        if not args.no_occlusion_aug:
            print(f"🎭 遮擋模擬訓練已啟用:")
            print(f"   - 遮擋機率: {args.occlusion_prob}")
            print(f"   - 遮擋比例: {args.occlusion_ratio_min} ~ {args.occlusion_ratio_max}")
            print(f"   - 最大連續遮擋: {args.max_consecutive_occlusion} 幀")
        print("="*60)
        trainer = GRUSequenceTrainer(args)
    else:
        print("\n" + "="*60)
        print("📦 使用標準訓練模式（無 GRU 記憶）")
        print("="*60)
        trainer = UnifiedTrainer(args)
    
    trainer.train()


if __name__ == "__main__":
    main()
