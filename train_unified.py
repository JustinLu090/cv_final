
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
        
        # dpeth regression 
        if 'depth_regression' in self.tasks and 'depth1' in sample:
            depth = self._load_depth(sample['depth1'])
            if depth is not None:
                # average depth in center region
                h, w = depth.shape
                center_region = depth[h//4:3*h//4, w//4:3*w//4]
                valid = center_region[(center_region > 0.1) & (center_region < 10.0)]
                if len(valid) > 100:
                    avg_depth = valid.mean()
                    # normalize to [0, 1] (0.5m to 5m)
                    normalized_depth = np.clip((avg_depth - 0.5) / 4.5, 0, 1)
                    result['depth_regression_label'] = normalized_depth

        # motion prediction
        if 'motion' in self.tasks and 'pose1' in sample:
            pose1 = self._load_pose(sample['pose1'])
            pose2 = self._load_pose(sample['pose2'])
            motion = self._compute_relative_motion(pose1, pose2)
            result['motion_label'] = motion
        
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
            result['depth_regression_label'] = torch.tensor(valid_depth_reg, dtype=torch.float32)
    
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

        # loss function
        self.loss_fn = UnifiedLoss(
            temporal_weight=args.temporal_weight,
            depth_order_weight=args.depth_order_weight,
            motion_weight=args.motion_weight,
        )

        # optimizer
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
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
            outputs = self.model(
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
        }
        
        depth_correct = 0
        depth_total = 0
        
        for batch in tqdm(self.val_loader, desc="Evaluating"):
            feat1 = self.extract_features(batch['image1'])
            feat2 = self.extract_features(batch['image2'])
            
            # Temporal Consistency(need better metrics)
            if 'temporal' in self.tasks:
                outputs = self.model(feat2, feat1, tasks=['temporal'])
                refined = outputs['temporal']
                consistency = F.cosine_similarity(refined, feat1, dim=-1).mean()
                metrics['temporal_consistency'].append(consistency.item())
            
            # depth order accuracy
            if 'depth_order' in self.tasks and 'region_a' in batch and batch['region_a']:
                region_a_feat = self.extract_features(batch['region_a'])
                region_b_feat = self.extract_features(batch['region_b'])
                
                outputs = self.model(
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
                outputs = self.model(feat2, feat1, tasks=['motion'])
                pred = outputs['motion']
                gt = batch['motion_label'].to(self.device)
                error = (pred - gt).abs().mean()
                metrics['motion_error'].append(error.item())
        
        results = {}
        if metrics['temporal_consistency']:
            results['temporal_consistency'] = np.mean(metrics['temporal_consistency'])
        if depth_total > 0:
            results['depth_order_acc'] = depth_correct / depth_total
        if metrics['motion_error']:
            results['motion_mae'] = np.mean(metrics['motion_error'])
        
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
            
            history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'lr': current_lr,
                **val_results
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
    
    args = parser.parse_args()
    
    trainer = UnifiedTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
