
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
        tasks: list = ['temporal', 'depth_order', 'motion'],
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
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        max_scenes: int = 100,
        sequence_length: int = 8,  # 每個序列的幀數
        stride: int = 4,  # 序列之間的間隔
        tasks: list = ['temporal', 'depth_regression', 'motion'],
    ):
        self.data_root = Path(data_root)
        self.sequence_length = sequence_length
        self.stride = stride
        self.tasks = tasks
        
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
    
    def __getitem__(self, idx):
        """
        返回一個完整的序列
        """
        sequence = self.sequences[idx]
        
        result = {
            'scene': sequence['scene'],
            'images': [],  # List of PIL Images
            'depth_regression_labels': [],  # List of [3] arrays
            'motion_labels': [],  # List of [6] arrays
            'valid_depth': [],  # Boolean mask
            'valid_motion': [],  # Boolean mask
        }
        
        for i, frame_info in enumerate(sequence['frames']):
            # 載入圖像
            image = Image.open(frame_info['color']).convert('RGB')
            result['images'].append(image)
            
            # 深度回歸標籤
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
        
        return result


def sequence_collate(batch):
    """
    序列 Dataset 的 collate 函數
    由於序列長度固定，可以正常 batch
    """
    batch_size = len(batch)
    seq_len = len(batch[0]['images'])
    
    result = {
        'scene': [b['scene'] for b in batch],
        'images': [],  # [T][B] list of lists
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

        # loss function (使用自動 Loss 平衡)
        self.loss_fn = UnifiedLoss(
            num_tasks=5,
            use_uncertainty_weighting=True
        )

        # optimizer (包含模型參數和 Loss 函數的可學習參數)
        all_params = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        all_params += list(self.loss_fn.parameters())  # 加入 Loss 的 log_vars 參數
        
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
        
        for batch in pbar:
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
            
            for task, l in loss_dict.items():
                loss_history[task].append(l)
            
            desc = f"Epoch {epoch} | "
            for task in self.tasks:
                if loss_history[task]:
                    desc += f"{task[:4]}:{np.mean(loss_history[task][-20:]):.4f} "
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
            use_gru_memory=True,  # 啟用 GRU
        )
        self.model = self.model.to(self.device).float()
        
        # 模型資訊
        info = get_model_info(self.model)
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\n模型參數: {info['total_params']:,} (可訓練: {trainable:,})")
        
        # Loss function
        self.loss_fn = UnifiedLoss(
            num_tasks=5,
            use_uncertainty_weighting=True
        )
        
        # Optimizer
        all_params = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        all_params += list(self.loss_fn.parameters())
        
        self.optimizer = torch.optim.AdamW(
            all_params,
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.epochs
        )
        
        # 序列 Dataset
        self.train_dataset = ScanNetSequenceDataset(
            args.data_root, 'train',
            max_scenes=args.max_scenes,
            sequence_length=args.sequence_length,
            stride=args.stride,
            tasks=self.tasks
        )
        self.val_dataset = ScanNetSequenceDataset(
            args.data_root, 'val',
            max_scenes=args.max_scenes,
            sequence_length=args.sequence_length,
            stride=args.stride,
            tasks=self.tasks
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
        """
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        loss_history = {
            'temporal': [],
            'depth_regression': [],
            'motion': [],
            'memory_quality': [],
        }
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch in pbar:
            self.optimizer.zero_grad()
            
            batch_size = len(batch['scene'])
            seq_len = len(batch['images'])
            
            # 初始化隱藏狀態（每個新序列開始時重置）
            hidden_state = None
            
            # 累積整個序列的 loss
            seq_loss = 0
            seq_steps = 0
            
            prev_feat = None
            
            # 遍歷序列中的每一幀
            for t in range(seq_len):
                # 提取當前幀的特徵
                curr_feat = self.extract_features(batch['images'][t])  # [B, feat_dim]
                
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
                
                # 計算這一幀的 loss
                if targets:
                    frame_loss, frame_loss_dict = self.loss_fn(outputs, targets, prev_feat)
                    if frame_loss > 0:
                        seq_loss = seq_loss + frame_loss
                        seq_steps += 1
                        
                        for task, l in frame_loss_dict.items():
                            if task in loss_history:
                                loss_history[task].append(l)
                
                # 更新 prev_feat（用於下一幀的 motion 計算）
                prev_feat = curr_feat
                
                # Detach hidden state 以實現 Truncated BPTT
                # 這樣梯度只會在序列內傳播，不會跨序列
                if hidden_state is not None:
                    hidden_state = hidden_state.detach()
            
            # 序列結束，計算平均 loss 並更新參數
            if seq_steps > 0:
                avg_seq_loss = seq_loss / seq_steps
                avg_seq_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                total_loss += avg_seq_loss.item()
                num_batches += 1
            
            # 更新進度條
            desc = f"Epoch {epoch} | "
            for task in ['temporal', 'depth_regression', 'motion']:
                if loss_history[task]:
                    desc += f"{task[:5]}:{np.mean(loss_history[task][-20:]):.4f} "
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
    parser.add_argument('--max_scenes', type=int, default=50)
    parser.add_argument('--frames_per_scene', type=int, default=30)
    
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
                        default=['temporal', 'depth_order', 'motion'],
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
    
    args = parser.parse_args()
    
    # 根據模式選擇訓練器
    if args.use_gru:
        print("\n" + "="*60)
        print("🧠 使用 GRU 序列訓練模式")
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
