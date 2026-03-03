import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
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
from datasets import load_dataset

from models_unified import UnifiedTempoVLM, UnifiedLoss, get_model_info

# ============================================================
# ScanNet Dataset Classes
# ============================================================

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
        self.split = split
        
        scenes_dir = self.data_root / 'scannet_frames_25k'
        all_scenes = sorted([d for d in scenes_dir.iterdir() if d.is_dir()])
        
        split_idx = int(len(all_scenes) * 0.8)
        if split == 'train':
            self.scenes = all_scenes[:split_idx][:max_scenes]
        else:
            self.scenes = all_scenes[split_idx:][:max_scenes // 5]

        self.samples = []
        self._collect_samples()
        
        print(f"[{split}] ScanNet loaded: {len(self.scenes)} scenes, {len(self.samples)} samples")
    
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
                    
                    label = 0 if depth_a < depth_b else 1
                    return crop_a, crop_b, label
        
        return None, None, None
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image1 = Image.open(sample['color1']).convert('RGB')
        image2 = Image.open(sample['color2']).convert('RGB')
        
        result = {
            'image1': image1,
            'image2': image2,
            'scene': sample['scene'],
        }

        # Random depth scale augmentation
        depth_scale_factor = 1.0
        if self.split == 'train':
            depth_scale_factor = random.uniform(0.75, 1.5)
        
        if 'depth_order' in self.tasks and 'depth1' in sample:
            depth = self._load_depth(sample['depth1'])
            if depth is not None:
                depth = depth * depth_scale_factor
                crop_a, crop_b, label = self._sample_depth_regions(depth, image1)
                result['region_a'] = crop_a
                result['region_b'] = crop_b
                result['depth_order_label'] = label
        
        if 'depth_regression' in self.tasks and 'depth1' in sample:
            depth = self._load_depth(sample['depth1'])
            if depth is not None:
                depth = depth * depth_scale_factor
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
                    valid = region[(region > 0.1) & (region < 20.0)]
                    if len(valid) > 100:
                        depths.append(valid.mean())
                        valid_count += 1
                    else:
                        depths.append(0.0)
                
                if valid_count >= 2:
                    result['depth_regression_label'] = np.array(depths, dtype=np.float32)

        if 'motion' in self.tasks and 'pose1' in sample:
            pose1 = self._load_pose(sample['pose1'])
            pose2 = self._load_pose(sample['pose2'])
            motion = self._compute_relative_motion(pose1, pose2)
            if motion is not None:
                motion[:3] = motion[:3] * depth_scale_factor
                result['motion_label'] = motion
        
        return result

class ScanNetSequenceDataset(Dataset):
    """
    Sequence dataset for GRU long-term memory training (ScanNet)
    """
    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        max_scenes: int = 100,
        sequence_length: int = 8,
        stride: int = 4,
        tasks: list = ['temporal', 'depth_regression', 'motion'],
        occlusion_prob: float = 0.3,
        occlusion_ratio_range: tuple = (0.3, 0.6),
        max_consecutive_occlusion: int = 3,
    ):
        self.data_root = Path(data_root)
        self.sequence_length = sequence_length
        self.stride = stride
        self.tasks = tasks
        self.split = split
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
        
        print(f"[{split}] ScanNet Sequence loaded: {len(self.sequences)} sequences")
    
    def _collect_sequences(self):
        for scene_dir in tqdm(self.scenes, desc="Collecting ScanNet sequences"):
            color_dir = scene_dir / 'color'
            depth_dir = scene_dir / 'depth'
            pose_dir = scene_dir / 'pose'
            
            if not color_dir.exists():
                continue
            
            color_files = sorted(color_dir.glob('*.jpg'))
            
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
                    
                    depth_path = depth_dir / (color_path.stem + '.png')
                    if depth_path.exists():
                        frame_info['depth'] = depth_path
                    
                    pose_path = pose_dir / (color_path.stem + '.txt')
                    if pose_path.exists():
                        frame_info['pose'] = pose_path
                    
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
        if depth is None: return None
        return depth.astype(np.float32) / 1000.0
    
    def _load_pose(self, path):
        try:
            pose = np.loadtxt(str(path))
            return pose.reshape(4, 4)
        except:
            return None
    
    def _compute_relative_motion(self, pose1, pose2):
        if pose1 is None or pose2 is None: return None
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
        if depth is None: return None
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
            valid = region[(region > 0.1) & (region < 20.0)]
            if len(valid) > 100:
                depths.append(valid.mean())
                valid_count += 1
            else:
                depths.append(0.0)
        
        if valid_count >= 2:
            return np.array(depths, dtype=np.float32)
        return None
    
    def _apply_occlusion(self, image):
        if random.random() > self.occlusion_prob:
            return image, False
        
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        cx, cy = w // 2, h // 2
        ratio = random.uniform(*self.occlusion_ratio_range)
        size = int(min(w, h) * ratio / 2)
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
        pattern = []
        consecutive = 0
        for i in range(seq_len):
            if consecutive >= self.max_consecutive_occlusion:
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
        sequence = self.sequences[idx]

        current_scene = sequence['scene']
        while True:
            # Randomly select another sequence
            neg_idx = random.randint(0, len(self.sequences) - 1)
            
            # Ensure the selected scene is different (true negative sample)
            if self.sequences[neg_idx]['scene'] != current_scene:
                neg_sequence = self.sequences[neg_idx]
                break
        
        # Use the first frame of the negative sequence as the global negative image
        neg_frame_info = neg_sequence['frames'][0]
        neg_image = Image.open(neg_frame_info['color']).convert('RGB')
        # ============================================================

        occlusion_pattern = self._generate_occlusion_pattern(self.sequence_length)
        
        # Sequence-level depth scale augmentation
        depth_scale_factor = 1.0
        if self.split == 'train':
            depth_scale_factor = random.uniform(0.75, 1.5)

        result = {
            'scene': sequence['scene'],
            'global_negative_image': neg_image,  # [New] store negative image in output dict
            'images': [],
            'images_clean': [],
            'is_occluded': [],
            'depth_regression_labels': [],
            'motion_labels': [],
            'valid_depth': [],
            'valid_motion': [],
        }
        
        for i, frame_info in enumerate(sequence['frames']):
            image = Image.open(frame_info['color']).convert('RGB')
            result['images_clean'].append(image)
            
            if occlusion_pattern[i]:
                occluded_image, _ = self._apply_occlusion(image)
                result['images'].append(occluded_image)
                result['is_occluded'].append(True)
            else:
                result['images'].append(image)
                result['is_occluded'].append(False)
            
            if 'depth_regression' in self.tasks and 'depth' in frame_info:
                depth = self._load_depth(frame_info['depth'])
                if depth is not None:
                    depth = depth * depth_scale_factor
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
            else:
                result['depth_regression_labels'].append(np.zeros(3, dtype=np.float32))
                result['valid_depth'].append(False)
            
            if 'motion' in self.tasks and i < len(sequence['frames']) - 1:
                if 'pose' in frame_info and 'next_pose' in frame_info:
                    pose1 = self._load_pose(frame_info['pose'])
                    pose2 = self._load_pose(frame_info['next_pose'])
                    motion = self._compute_relative_motion(pose1, pose2)
                    if motion is not None:
                        motion[:3] = motion[:3] * depth_scale_factor
                        result['motion_labels'].append(motion.astype(np.float32))
                        result['valid_motion'].append(True)
                    else:
                        result['motion_labels'].append(np.zeros(6, dtype=np.float32))
                        result['valid_motion'].append(False)
                else:
                    result['motion_labels'].append(np.zeros(6, dtype=np.float32))
                    result['valid_motion'].append(False)
        
        result['depth_regression_labels'] = np.stack(result['depth_regression_labels'])
        if result['motion_labels']:
            result['motion_labels'] = np.stack(result['motion_labels'])
        result['valid_depth'] = np.array(result['valid_depth'])
        result['valid_motion'] = np.array(result['valid_motion'])
        result['is_occluded'] = np.array(result['is_occluded'])
        
        return result

# ============================================================
# [NEW] NYU Depth V2 Sequence Dataset
# ============================================================

class NYUDepthSequenceDataset(Dataset):
    """
    Hugging Face NYU Depth V2 dataset adapter (pseudo-sequence)
    Uses jagennath-hari/nyuv2 for format compatibility (PIL image + uint16 depth)
    [Fix] add global negative mining
    """
    def __init__(self, split='train', sequence_length=8):
        self.split = split
        self.sequence_length = sequence_length
        
        # NYU V2 is usually split into train and test (no validation split)
        hf_split = 'train' if split == 'train' else 'test'
        
        print(f"Loading NYU Depth V2 [{hf_split}] from Hugging Face (jagennath-hari/nyuv2)...")
        
        try:
            self.dataset = load_dataset("jagennath-hari/nyuv2", split=hf_split)
        except ValueError:
            print(f"Warning: split '{hf_split}' not found, trying 'validation'...")
            self.dataset = load_dataset("jagennath-hari/nyuv2", split="validation")
            
        print(f"[{split}] NYU Depth V2 loaded: {len(self.dataset)} samples")

    def __len__(self):
        return len(self.dataset)

    def _get_depth_regions(self, depth_np):
        """Compute average depth of left/center/right regions"""
        if depth_np.ndim == 3:
            depth_np = depth_np.squeeze()
            
        h, w = depth_np.shape
        regions = {
            'left': depth_np[:, :w//3],
            'center': depth_np[:, w//3:2*w//3],
            'right': depth_np[:, 2*w//3:]
        }
        depths = []
        valid_count = 0
        for name in ['left', 'center', 'right']:
            region = regions[name]
            valid = region[(region > 0.1) & (region < 10.0)]
            if len(valid) > 50:
                depths.append(valid.mean())
                valid_count += 1
            else:
                depths.append(0.0)
        
        if valid_count >= 2:
            return np.array(depths, dtype=np.float32)
        return None

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # ============================================================
        # [New] Global negative mining (NYU)
        # ============================================================
        # Randomly select another image as a negative sample
        while True:
            neg_idx = random.randint(0, len(self.dataset) - 1)
            if neg_idx != idx:
                break
        
        neg_item = self.dataset[neg_idx]
        
        # Process negative-sample image
        if 'rgb' in neg_item: neg_image = neg_item['rgb']
        elif 'image' in neg_item: neg_image = neg_item['image']
        else: neg_image = Image.new('RGB', (640, 480)) # Fallback
        
        if not isinstance(neg_image, Image.Image):
            neg_image = Image.fromarray(np.array(neg_image)).convert('RGB')
        else:
            neg_image = neg_image.convert('RGB')
        # ============================================================
        
        # 1. Image preprocessing
        if 'rgb' in item:
            image = item['rgb']
        elif 'image' in item:
            image = item['image']
        else:
            raise KeyError(f"Unknown image key in dataset: {list(item.keys())}")
            
        if not isinstance(image, Image.Image):
             image = Image.fromarray(np.array(image)).convert('RGB')
        else:
             image = image.convert('RGB')
        
        # 2. Depth preprocessing
        if 'depth' in item:
            depth_map = item['depth']
        elif 'depth_map' in item:
            depth_map = item['depth_map']
        else:
             raise KeyError(f"Unknown depth key in dataset: {list(item.keys())}")

        depth_np = np.array(depth_map).astype(np.float32)
        
        if depth_np.max() > 100.0: 
            depth_np /= 1000.0
        elif depth_np.dtype == np.uint8: 
            depth_np = depth_np / 255.0 * 10.0

        # 3. Depth augmentation
        if self.split == 'train':
            scale = random.uniform(0.75, 1.5)
            depth_np = depth_np * scale

        # 4. Build pseudo sequence
        depth_label_single = self._get_depth_regions(depth_np)
        
        has_depth = (depth_label_single is not None)
        if not has_depth:
            depth_label_single = np.zeros(3, dtype=np.float32)

        result = {
            'scene': f"nyu_{idx}",
            'global_negative_image': neg_image, # [New] this field is required
            'images': [image] * self.sequence_length, 
            'images_clean': [image] * self.sequence_length,
            'is_occluded': np.zeros(self.sequence_length, dtype=bool),
            'depth_regression_labels': np.stack([depth_label_single] * self.sequence_length),
            'valid_depth': np.full(self.sequence_length, has_depth, dtype=bool),
            'motion_labels': np.zeros((self.sequence_length - 1, 6), dtype=np.float32),
            'valid_motion': np.zeros(self.sequence_length - 1, dtype=bool) 
        }
        return result


# ============================================================
# Collate Functions
# ============================================================

def sequence_collate(batch):
    batch_size = len(batch)
    seq_len = len(batch[0]['images'])
    
    result = {
        'scene': [b['scene'] for b in batch],
        'global_negative_image': [b['global_negative_image'] for b in batch],
        'images': [],
        'images_clean': [],
        'is_occluded': torch.stack([torch.tensor(b['is_occluded']) for b in batch]),
        'depth_regression_labels': torch.stack([torch.tensor(b['depth_regression_labels']) for b in batch]),
        'valid_depth': torch.stack([torch.tensor(b['valid_depth']) for b in batch]),
    }
    
    for t in range(seq_len):
        result['images'].append([b['images'][t] for b in batch])
        result['images_clean'].append([b['images_clean'][t] for b in batch])
    
    # Motion labels (may be empty or invalid)
    if 'motion_labels' in batch[0]:
        result['motion_labels'] = torch.stack([torch.tensor(b['motion_labels']) for b in batch])
        result['valid_motion'] = torch.stack([torch.tensor(b['valid_motion']) for b in batch])
    
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
            stacked = np.stack(valid_depth_reg, axis=0)
            result['depth_regression_label'] = torch.tensor(stacked, dtype=torch.float32)
    
    if 'motion_label' in batch[0] and batch[0]['motion_label'] is not None:
        valid_motion = [b['motion_label'] for b in batch if b['motion_label'] is not None]
        if valid_motion:
            result['motion_label'] = torch.tensor(np.stack(valid_motion), dtype=torch.float32)
    
    return result


# ============================================================
# Trainers
# ============================================================

class UnifiedTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if 'all' in args.tasks:
            self.tasks = ['temporal', 'depth_order', 'depth_regression', 'motion']
        else:
            self.tasks = args.tasks
        
        print(f"Training tasks: {self.tasks}")
        print("\nLoading Qwen2-VL...")
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", trust_remote_code=True)
        self.qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct", torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
        ).eval()
        for param in self.qwen_model.parameters(): param.requires_grad = False
        
        print("\ncreate UnifiedTempoVLM...")
        self.model = UnifiedTempoVLM(
            feat_dim=args.feat_dim, hidden_dim=args.hidden_dim, use_transformer_encoder=True,
            num_encoder_layers=2, num_heads=8
        )
        
        if args.pretrained and not args.no_pretrained:
            print(f"\ntry to load pretrained weights: {args.pretrained}")
            try:
                self.model.load_pretrained_temporal(args.pretrained)
            except Exception as e:
                print(f"Warning: failed to load pretrained weights: {e}")
        else:
            print("\ntrain from scratch")
        
        if args.freeze_temporal:
            for name, param in self.model.named_parameters():
                if 'temporal' in name or 'shared_encoder' in name:
                    param.requires_grad = False
        
        self.model = self.model.to(self.device).float()
        
        self.loss_fn = UnifiedLoss(
            num_tasks=5,
            use_uncertainty_weighting=False,
            task_weights={
                'temporal': 0.1, 'depth_order': 1.0, 'depth_regression': 5.0,
                'motion': 2.0, 'scene_class': 0.5, 'occlusion_recon': 1.5, 'memory_quality_reg': 0.5,
            }
        )

        all_params = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        if hasattr(self.loss_fn, 'log_vars'): all_params += list(self.loss_fn.parameters())
        
        self.optimizer = torch.optim.AdamW(all_params, lr=args.lr, weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.epochs)
        
        self.train_dataset = ScanNetUnifiedDataset(
            args.data_root, 'train', max_scenes=args.max_scenes, frames_per_scene=args.frames_per_scene, tasks=self.tasks
        )
        self.val_dataset = ScanNetUnifiedDataset(
            args.data_root, 'val', max_scenes=args.max_scenes, frames_per_scene=args.frames_per_scene, tasks=self.tasks
        )
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=custom_collate)
        self.val_loader = DataLoader(self.val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=custom_collate)
        
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.start_epoch = 0
        self.best_loss = float('inf')
        
        if args.resume: self._load_checkpoint(args.resume, args.resume_epoch)
    
    def _load_checkpoint(self, checkpoint_path, resume_epoch=None):
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
        if 'optimizer_state_dict' in checkpoint: self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint: self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch = resume_epoch if resume_epoch else checkpoint.get('epoch', 0) + 1
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        print(f"Loaded checkpoint from epoch {self.start_epoch}")

    def extract_features(self, images):
        features = []
        for image in images:
            messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": "Describe."}]}]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt")
            inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
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
            feat1 = self.extract_features(batch['image1'])
            feat2 = self.extract_features(batch['image2'])
            region_a_feat = self.extract_features(batch['region_a']) if 'region_a' in batch and batch['region_a'] else None
            region_b_feat = self.extract_features(batch['region_b']) if 'region_b' in batch and batch['region_b'] else None
            
            outputs, _ = self.model(curr_feat=feat2, prev_feat=feat1, region_a_feat=region_a_feat, region_b_feat=region_b_feat, tasks=self.tasks)
            targets = {}
            if 'depth_order' in batch: targets['depth_order'] = batch['depth_order_label'].to(self.device)
            if 'depth_regression_label' in batch: targets['depth_regression'] = batch['depth_regression_label'].to(self.device)
            if 'motion_label' in batch: targets['motion'] = batch['motion_label'].to(self.device)
            
            loss, loss_dict = self.loss_fn(outputs, targets, feat1)
            if loss > 0:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            total_loss += loss.item()
            for task, l in loss_dict.items():
                if task in loss_history: loss_history[task].append(l)
            
            desc = f"Epoch {epoch} | "
            for task in self.tasks:
                if loss_history[task]: desc += f"{task[:4]}:{np.mean(loss_history[task][-20:]):.4f} "
            pbar.set_description(desc)
        self.scheduler.step()
        return total_loss / len(self.train_loader), loss_history

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        metrics = {'temporal_consistency': [], 'depth_order_acc': [], 'motion_error': [], 'rotation_error': []}
        depth_correct = 0; depth_total = 0
        for batch in tqdm(self.val_loader, desc="Evaluating"):
            feat1 = self.extract_features(batch['image1'])
            feat2 = self.extract_features(batch['image2'])
            if 'temporal' in self.tasks:
                outputs, _ = self.model(feat2, feat1, tasks=['temporal'])
                metrics['temporal_consistency'].append(F.cosine_similarity(outputs['temporal'], feat1, dim=-1).mean().item())
            if 'depth_order' in self.tasks and 'region_a' in batch and batch['region_a']:
                region_a_feat = self.extract_features(batch['region_a'])
                region_b_feat = self.extract_features(batch['region_b'])
                outputs, _ = self.model(feat2, feat1, region_a_feat=region_a_feat, region_b_feat=region_b_feat, tasks=['depth_order'])
                pred = outputs['depth_order'].argmax(dim=-1)
                gt = batch['depth_order_label'].to(self.device)
                depth_correct += (pred == gt).sum().item(); depth_total += len(gt)
        results = {}
        if metrics['temporal_consistency']: results['temporal_consistency'] = np.mean(metrics['temporal_consistency'])
        if depth_total > 0: results['depth_order_acc'] = depth_correct / depth_total
        return results

    def train(self):
        best_metric = -float('inf')
        for epoch in range(self.start_epoch + 1, self.args.epochs + 1):
            train_loss, _ = self.train_epoch(epoch)
            val_results = self.evaluate()
            print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val: {val_results}")
            metric = val_results.get('temporal_consistency', 0) + val_results.get('depth_order_acc', 0)
            if metric > best_metric:
                best_metric = metric
                torch.save({'model_state_dict': self.model.state_dict()}, self.output_dir / 'best_unified_model.pt')


class GRUSequenceTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tasks = ['temporal', 'depth_regression', 'motion']
        print(f"GRU Training tasks: {self.tasks}")
        
        print("\nLoading Qwen2-VL...")
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", trust_remote_code=True)
        self.qwen_model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", torch_dtype=torch.float16, device_map="auto", trust_remote_code=True).eval()
        for param in self.qwen_model.parameters(): param.requires_grad = False
        
        print("\nCreating UnifiedTempoVLM (with GRU memory)...")
        self.model = UnifiedTempoVLM(
            feat_dim=args.feat_dim, hidden_dim=args.hidden_dim, use_gru_memory=True,
            use_transformer_encoder=True, num_encoder_layers=2, num_heads=8
        )
        self.model = self.model.to(self.device).float()
        
        # Loss & Optimizer
        self.loss_fn = UnifiedLoss(
            num_tasks=5, use_uncertainty_weighting=False,
            task_weights={'temporal': 0.1, 'depth_order': 1.0, 'depth_regression': 5.0, 'motion': 2.0, 'scene_class': 0.5, 'occlusion_recon': 1.5, 'memory_quality_reg': 0.5}
        )
        all_params = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        if hasattr(self.loss_fn, 'log_vars'): all_params += list(self.loss_fn.parameters())
        self.optimizer = torch.optim.AdamW(all_params, lr=args.lr, weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.epochs)
        
        # Datasets & Augmentation
        enable_occlusion = not getattr(args, 'no_occlusion_aug', False)
        occlusion_prob = getattr(args, 'occlusion_prob', 0.3) if enable_occlusion else 0.0
        occlusion_ratio_range = (getattr(args, 'occlusion_ratio_min', 0.3), getattr(args, 'occlusion_ratio_max', 0.6)) if enable_occlusion else (0.3, 0.6)
        max_consecutive_occlusion = getattr(args, 'max_consecutive_occlusion', 3) if enable_occlusion else 3
        
        # 1. ScanNet Dataset
        self.scannet_train = ScanNetSequenceDataset(
            args.data_root, 'train', max_scenes=args.max_scenes, sequence_length=args.sequence_length, stride=args.stride,
            tasks=self.tasks, occlusion_prob=occlusion_prob, occlusion_ratio_range=occlusion_ratio_range, max_consecutive_occlusion=max_consecutive_occlusion
        )
        self.val_dataset = ScanNetSequenceDataset(
            args.data_root, 'val', max_scenes=args.max_scenes, sequence_length=args.sequence_length, stride=args.stride, tasks=self.tasks, occlusion_prob=0.0
        )
        
        # 2. NYU Depth V2 Dataset (for mixed training)
        print("Adding NYU Depth V2 for mixed training...")
        self.nyu_train = NYUDepthSequenceDataset(split='train', sequence_length=args.sequence_length)
        
        # 3. Mixed Training
        self.train_dataset = ConcatDataset([self.scannet_train, self.nyu_train])
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=sequence_collate)
        self.val_loader = DataLoader(self.val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=sequence_collate)
        
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.start_epoch = 0; self.best_loss = float('inf')
        
        if args.resume: self._load_checkpoint(args.resume, args.resume_epoch)
        
        self.feature_bank_size = 2048; self.feature_bank = None; self.feature_bank_ptr = 0; self.contrastive_temperature = 0.07

    def _load_checkpoint(self, checkpoint_path, resume_epoch=None):
        print(f"\nLoading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            try:
                self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                print("  Model weights loaded (strict=False)")
            except Exception as e:
                print(f"  Model loading warning: {e}")
        else:
            self.model.load_state_dict(checkpoint, strict=False)
            print("  Model weights loaded (raw format, strict=False)")
        if 'optimizer_state_dict' in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("  Optimizer state loaded")
            except ValueError:
                print("  Optimizer state mismatch; reinitializing optimizer")
        if 'scheduler_state_dict' in checkpoint:
            try:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print("  Scheduler state loaded")
            except ValueError:
                print("  Scheduler state mismatch; reinitializing scheduler")
        if resume_epoch is not None:
            self.start_epoch = resume_epoch
        elif 'epoch' in checkpoint:
            self.start_epoch = checkpoint['epoch'] + 1
        else:
            import re
            match = re.search(r'epoch_?(\d+)', checkpoint_path)
            if match: self.start_epoch = int(match.group(1)) + 1
        if 'best_loss' in checkpoint:
            self.best_loss = checkpoint['best_loss']
            print(f"  Best loss: {self.best_loss:.4f}")
        print(f"  Resuming training from epoch {self.start_epoch}")

    @torch.no_grad()
    def update_feature_bank(self, features):
        features = features.detach()
        batch_size = features.shape[0]
        if self.feature_bank is None: self.feature_bank = torch.zeros(self.feature_bank_size, features.shape[1], device=self.device)
        if self.feature_bank_ptr + batch_size <= self.feature_bank_size:
            self.feature_bank[self.feature_bank_ptr:self.feature_bank_ptr+batch_size] = features
        else:
            first = self.feature_bank_size - self.feature_bank_ptr
            self.feature_bank[self.feature_bank_ptr:] = features[:first]
            self.feature_bank[:batch_size-first] = features[first:]
        self.feature_bank_ptr = (self.feature_bank_ptr + batch_size) % self.feature_bank_size

    def compute_contrastive_loss(self, curr_refined, prev_feat, global_neg_feat=None):
        """
        Compute GRU-mode contrastive loss (with stop-gradient)
        """
        # 1. Feature normalization
        curr_norm = F.normalize(curr_refined, p=2, dim=-1)
        prev_norm = F.normalize(prev_feat, p=2, dim=-1)
        
        # ============================================================
        # [Update] Stop-gradient mechanism (SimSiam style)
        # ============================================================
        # Bring current features closer to previous features without backprop through previous features
        # This helps avoid representational collapse
        pos_sim = (curr_norm * prev_norm.detach()).sum(dim=-1)  # [B]
        
        # 3. Compute feature-bank negative similarity (bank negatives)
        bank_neg_exp = torch.tensor(0.0, device=self.device)
        if self.feature_bank is not None and self.feature_bank_ptr > 0:
            valid_size = min(self.feature_bank_ptr, self.feature_bank_size)
            bank_features = self.feature_bank[:valid_size]
            # Features in the bank are already detached
            bank_norm = F.normalize(bank_features, p=2, dim=-1)
            
            neg_sim = curr_norm @ bank_norm.T  # [B, N_bank]
            bank_neg_exp = torch.exp(neg_sim / self.contrastive_temperature).sum(dim=1) # [B]
        
        # 4. Compute global negative similarity (if available)
        global_neg_exp = torch.tensor(0.0, device=self.device)
        if global_neg_feat is not None:
            # Global negatives should also be detached; they are reference negatives only
            global_neg_norm = F.normalize(global_neg_feat.detach(), p=2, dim=-1) 
            
            global_neg_sim = (curr_norm * global_neg_norm).sum(dim=-1) # [B]
            global_neg_exp = torch.exp(global_neg_sim / self.contrastive_temperature) # [B]

        # 5. Compose InfoNCE loss
        pos_exp = torch.exp(pos_sim / self.contrastive_temperature)
        
        # Denominator = positive + bank negatives + global negatives
        denominator = pos_exp + bank_neg_exp + global_neg_exp + 1e-8
        
        contrastive_loss = -torch.log(pos_exp / denominator).mean()
        
        # Diagnostics
        with torch.no_grad():
            avg_pos = pos_sim.mean().item()
            avg_neg = global_neg_sim.mean().item() if global_neg_feat is not None else 0.0
            
        return contrastive_loss, avg_pos, avg_neg
    def extract_features(self, images):
        features = []
        for image in images:
            messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": "Describe."}]}]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt")
            inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.qwen_model(**inputs, output_hidden_states=True)
                features.append(outputs.hidden_states[-1].mean(dim=1).float())
        return torch.cat(features, dim=0)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        # [New] Gradient accumulation steps
        # Batch size 2 * 4 steps ~= effective batch size 8
        accumulation_steps = 2
        
        loss_history = {
            'temporal': [], 'depth_regression': [], 'motion': [], 'memory_quality': [],
            'contrastive': [], 'contrastive_pos': [], 'contrastive_neg': [], 'occlusion_recon': [],
        }
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        # [Update] Use enumerate index i to track accumulation steps
        for i, batch in enumerate(pbar):
            # Do not zero gradients here; clear them only after accumulation
            # self.optimizer.zero_grad() moved below
            
            batch_size = len(batch['scene'])
            seq_len = len(batch['images'])
            is_occluded = batch.get('is_occluded', None)
            has_clean_images = 'images_clean' in batch
            
            hidden_state = None
            prev_feat = None
            prev_feat_clean = None
            
            seq_loss = 0
            seq_steps = 0
            contrastive_loss_sum = 0
            contrastive_steps = 0
            occlusion_recon_loss_sum = 0
            occlusion_recon_steps = 0
            features_for_bank_update = [] 
            
            # [Step 0] Prepare global negatives (project to 128-d)
            global_neg_proj = None
            if 'global_negative_image' in batch:
                g_neg_raw = self.extract_features(batch['global_negative_image'])
                # Project + detach
                global_neg_proj = self.model.encode_and_project(g_neg_raw).detach()

            for t in range(seq_len):
                curr_feat = self.extract_features(batch['images'][t])
                curr_feat_clean = None
                if has_clean_images:
                    curr_feat_clean = self.extract_features(batch['images_clean'][t])
                
                tasks_to_run = ['temporal', 'depth_regression']
                run_motion = False
                if t < seq_len - 1 and 'motion_labels' in batch:
                     if batch['valid_motion'][:, t].any():
                         tasks_to_run.append('motion')
                         run_motion = True

                outputs, hidden_state = self.model(
                    curr_feat=curr_feat,
                    prev_feat=prev_feat,
                    hidden_state=hidden_state,
                    tasks=tasks_to_run
                )
                
                if 'memory_quality' in outputs:
                    loss_history['memory_quality'].append(outputs['memory_quality'].item())
                
                # 1. Occlusion reconstruction loss
                if is_occluded is not None and has_clean_images and 'temporal' in outputs:
                    curr_occluded = is_occluded[:, t].to(self.device)
                    if curr_occluded.any():
                        # Use 1536-d temporal features
                        reconstruction_feat = outputs['temporal']
                        recon_loss = F.mse_loss(reconstruction_feat[curr_occluded], curr_feat_clean[curr_occluded])
                        cos_sim = F.cosine_similarity(reconstruction_feat[curr_occluded], curr_feat_clean[curr_occluded], dim=-1)
                        cos_loss = 1 - cos_sim.mean()
                        
                        total_recon = recon_loss + 0.5 * cos_loss
                        occlusion_recon_loss_sum += total_recon
                        occlusion_recon_steps += 1
                        loss_history['occlusion_recon'].append(total_recon.item())

                # 2. Contrastive loss (using 128-d projection)
                if 'contrastive_feat' in outputs and prev_feat is not None:
                    curr_proj = outputs['contrastive_feat']
                    
                    target_prev_raw = prev_feat_clean if prev_feat_clean is not None else prev_feat
                    # Target projection + detach
                    target_prev_proj = self.model.encode_and_project(target_prev_raw).detach()
                    
                    c_loss, p_sim, n_sim = self.compute_contrastive_loss(
                        curr_proj, 
                        target_prev_proj,
                        global_neg_feat=global_neg_proj
                    )
                    
                    contrastive_loss_sum += c_loss
                    contrastive_steps += 1
                    loss_history['contrastive'].append(c_loss.item())
                    loss_history['contrastive_pos'].append(p_sim)
                    loss_history['contrastive_neg'].append(n_sim)
                    
                    features_for_bank_update.append(curr_proj.detach())

                # 3. Task Losses
                targets = {}
                valid_depth = batch['valid_depth'][:, t]
                if valid_depth.any():
                    valid_depth_device = valid_depth.to(self.device)
                    targets['depth_regression'] = batch['depth_regression_labels'][:, t, :][valid_depth].to(self.device)
                    if 'depth_regression' in outputs:
                        outputs['depth_regression'] = outputs['depth_regression'][valid_depth_device]
                    if 'depth_scale' in outputs:
                        outputs['depth_scale'] = outputs['depth_scale'][valid_depth_device]

                if run_motion and 'motion' in outputs:
                    valid_motion = batch['valid_motion'][:, t]
                    if valid_motion.any():
                        valid_motion_device = valid_motion.to(self.device)
                        if outputs['motion'].shape[0] == batch_size: 
                            outputs['motion'] = outputs['motion'][valid_motion_device]
                            targets['motion'] = batch['motion_labels'][:, t, :][valid_motion].to(self.device)
                            if 'motion_log_var' in outputs:
                                outputs['motion_log_var'] = outputs['motion_log_var'][valid_motion_device]

                if targets:
                    outputs_for_loss = {}
                    if 'depth_regression' in targets:
                        outputs_for_loss['depth_regression'] = outputs['depth_regression']
                        if 'depth_scale' in outputs: outputs_for_loss['depth_scale'] = outputs['depth_scale']
                    if 'motion' in targets:
                        outputs_for_loss['motion'] = outputs['motion']
                        if 'motion_log_var' in outputs: outputs_for_loss['motion_log_var'] = outputs['motion_log_var']

                    frame_loss, frame_loss_dict = self.loss_fn(outputs_for_loss, targets, None)
                    
                    if frame_loss > 0:
                        seq_loss += frame_loss
                        seq_steps += 1
                        for k, v in frame_loss_dict.items():
                            if k in loss_history: loss_history[k].append(v)
                
                prev_feat = curr_feat
                prev_feat_clean = curr_feat_clean
                if hidden_state is not None:
                    hidden_state = hidden_state.detach()

            # Update feature bank
            if len(features_for_bank_update) > 0:
                self.update_feature_bank(features_for_bank_update[-1])

            # Total loss aggregation
            total_seq_loss = torch.tensor(0.0, device=self.device)
            if seq_steps > 0: total_seq_loss += 3.0 * (seq_loss / seq_steps)
            if contrastive_steps > 0: total_seq_loss += 0.5 * (contrastive_loss_sum / contrastive_steps)
            if occlusion_recon_steps > 0: total_seq_loss += 1.5 * (occlusion_recon_loss_sum / occlusion_recon_steps)
            
            # [Key] Gradient accumulation logic
            # 1. Divide loss by accumulation steps to avoid oversized gradients
            if total_seq_loss > 0:
                loss_to_backward = total_seq_loss / accumulation_steps
                loss_to_backward.backward()
                
                # Restore displayed loss scale since backward uses scaled loss
                total_loss += total_seq_loss.item() 
            
            # 2. Update params when accumulation threshold is reached or at last batch
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(self.train_loader):
                # Apply optimizer step
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad() # clear gradients
                
                num_batches += 1
            
            # Update progress bar
            desc = f"Epoch {epoch} | "
            for k in ['depth_regression', 'motion']:
                if loss_history[k]: desc += f"{k[:5]}:{np.mean(loss_history[k][-20:]):.4f} "
            if loss_history['contrastive']:
                cl = np.mean(loss_history['contrastive'][-20:])
                if loss_history['contrastive_pos']:
                    pos = np.mean(loss_history['contrastive_pos'][-20:])
                    neg = np.mean(loss_history['contrastive_neg'][-20:])
                    desc += f"| CL:{cl:.3f} pos:{pos:.3f} neg:{neg:.3f} "
                else:
                    desc += f"| CL:{cl:.3f} "
            
            pbar.set_description(desc)
            
        self.scheduler.step()
        
        return total_loss / max(num_batches, 1), loss_history

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        metrics = {'depth_error': [], 'motion_error': []}
        
        for batch in tqdm(self.val_loader, desc="Evaluating"):
            seq_len = len(batch['images'])
            hidden_state = None
            prev_feat = None
            
            for t in range(seq_len):
                curr_feat = self.extract_features(batch['images'][t])
                
                tasks = ['temporal', 'depth_regression']
                if t < seq_len - 1: 
                    tasks.append('motion')
                
                outputs, hidden_state = self.model(
                    curr_feat=curr_feat, 
                    prev_feat=prev_feat, 
                    hidden_state=hidden_state, 
                    tasks=tasks
                )
                
                # [Depth evaluation]
                valid_depth = batch['valid_depth'][:, t]
                if valid_depth.any() and 'depth_regression' in outputs:
                    pred = outputs['depth_regression'][valid_depth.to(self.device)]
                    gt = batch['depth_regression_labels'][:, t, :][valid_depth].to(self.device)
                    # Simple L1 error
                    metrics['depth_error'].append((pred - gt).abs().mean().item())
                
                # [Motion evaluation] - fix KeyError case
                # Must check whether 'motion' exists (usually absent at first frame)
                if 'motion' in outputs and 'motion_labels' in batch:
                    valid_motion = batch['valid_motion'][:, t]
                    if valid_motion.any():
                        pred = outputs['motion'][valid_motion.to(self.device)]
                        gt = batch['motion_labels'][:, t, :][valid_motion].to(self.device)
                        # Simple L1 error (translation + rotation)
                        metrics['motion_error'].append((pred - gt).abs().mean().item())
                
                prev_feat = curr_feat
                if hidden_state is not None: 
                    hidden_state = hidden_state.detach()
        
        # Avoid NumPy warnings from empty lists
        results = {}
        for k, v in metrics.items():
            if v:
                results[k] = np.mean(v)
            else:
                results[k] = 0.0
                
        return results

    def train(self):
        print(f"\n{'='*60}\nStarting GRU sequence training (mixed ScanNet + NYU)\n{'='*60}")
        for epoch in range(self.start_epoch + 1, self.args.epochs + 1):
            train_loss, _ = self.train_epoch(epoch)
            val_results = self.evaluate()
            print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val: {val_results}")
            
            if epoch % self.args.save_every == 0:
                torch.save({
                    'epoch': epoch, 'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss_fn_state_dict': self.loss_fn.state_dict(),
                }, self.output_dir / f'gru_checkpoint_epoch{epoch}.pt')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./scannet_data')
    parser.add_argument('--output_dir', type=str, default='./checkpoints_unified')
    parser.add_argument('--max_scenes', type=int, default=100)
    parser.add_argument('--frames_per_scene', type=int, default=50)
    parser.add_argument('--feat_dim', type=int, default=1536)
    parser.add_argument('--hidden_dim', type=int, default=768)
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--freeze_temporal', action='store_true')
    parser.add_argument('--no_pretrained', action='store_true')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--resume_epoch', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--tasks', type=str, nargs='+', default=['temporal', 'depth_regression'])
    parser.add_argument('--save_every', type=int, default=2)
    parser.add_argument('--use_gru', action='store_true')
    parser.add_argument('--sequence_length', type=int, default=8)
    parser.add_argument('--stride', type=int, default=4)
    parser.add_argument('--occlusion_prob', type=float, default=0.3)
    parser.add_argument('--occlusion_ratio_min', type=float, default=0.3)
    parser.add_argument('--occlusion_ratio_max', type=float, default=0.6)
    parser.add_argument('--max_consecutive_occlusion', type=int, default=3)
    parser.add_argument('--no_occlusion_aug', action='store_true')
    
    args = parser.parse_args()
    
    if args.use_gru:
        trainer = GRUSequenceTrainer(args)
    else:
        trainer = UnifiedTrainer(args)
    
    trainer.train()

if __name__ == "__main__":
    main()
