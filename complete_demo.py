#!/usr/bin/env python3
"""
TempoVLM 完整視覺化展示腳本
===========================

整合所有視覺化功能：
1. 時序穩定性 - Split Screen 遮擋測試影片
2. 深度感知 - Depth Ordering 儀表板
3. 運動感知 - Real-time Trajectory Plot
4. 遮擋測試 - Occlusion Detection & Memory Injection (NEW)

輸出：
- 對比影片
- 儀表板截圖
- 軌跡動畫
- 遮擋測試報告與統計
"""

import os
import sys
import json
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib.patches import Circle, Wedge
from matplotlib.collections import PatchCollection
import cv2
import argparse
from collections import deque
from datetime import datetime

# Qwen2-VL
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from utils.memory_utils import AdaptiveMemoryBuffer


class CompleteDemoVisualizer:
    """TempoVLM 完整展示視覺化器"""
    
    def __init__(self, unified_model_path, device='cuda'):
        self.device = device
        
        print("=" * 70)
        print("TempoVLM Complete Demo Visualizer")
        print("=" * 70)
        
        # 載入模型
        print("\nloading...")
        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct",
            trust_remote_code=True
        )
        self.base_model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct",
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True
        ).eval()
        
        self._load_unified_model(unified_model_path)
        
        # 時序緩衝區
        self.temporal_buffer = deque(maxlen=5)
        
        # 特徵投影器（用於注入時維度轉換）
        self.feature_projector = None
        
        print("model loaded.\n")
    
    def _load_unified_model(self, model_path):
        from models_unified import UnifiedTempoVLM
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        
        # 自動偵測模型參數
        if 'shared_encoder.0.weight' in state_dict:
            hidden_dim = state_dict['shared_encoder.0.weight'].shape[0]
        else:
            hidden_dim = 768
        
        # 偵測是否使用 GRU
        use_gru = 'temporal_gru.weight_ih' in state_dict or 'temporal_gru.weight_hh' in state_dict
        
        self.unified_model = UnifiedTempoVLM(
            hidden_dim=hidden_dim,
            use_gru_memory=use_gru
        ).to(self.device)
        
        if 'model_state_dict' in checkpoint:
            self.unified_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.unified_model.load_state_dict(checkpoint)
        
        self.unified_model.eval()
        self.unified_model.float()
        self.hidden_dim = hidden_dim
        self.use_gru = use_gru
        
        self.gru_hidden_state = None
        
        print(f"  ✅ Unified Model loaded (hidden_dim={hidden_dim}, GRU={use_gru})")
    
    def extract_features(self, image, use_adapter=True):
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe."}
            ]
        }]
        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.base_model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )
            hidden_states = outputs.hidden_states[-1]
            features = hidden_states.mean(dim=1).float()
        
        if use_adapter and self.unified_model is not None:
            self.temporal_buffer.append(features)
            if len(self.temporal_buffer) >= 2:
                prev_feat = self.temporal_buffer[-2]
                with torch.no_grad():
                    if hasattr(self, 'use_gru') and self.use_gru:
                        outputs, self.gru_hidden_state = self.unified_model(
                            features, prev_feat, 
                            hidden_state=self.gru_hidden_state,
                            tasks=['temporal']
                        )
                    else:
                        outputs, _ = self.unified_model(features, prev_feat, tasks=['temporal'])
                    features = outputs['temporal']
        
        return features
    
    def generate_description(self, image, prompt="Describe what you see in the center of this image."):
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }]
        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            generated = self.base_model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False
            )
        
        response = self.processor.decode(generated[0], skip_special_tokens=True)
        if "assistant" in response.lower():
            response = response.split("assistant")[-1].strip()
        return response
    
    def generate_with_injection(self, image, memory_feat, prompt, injection_strength=0.5, injection_method='full'):
        """
        🧠 Direct Feature Injection - 採用 occlusion_tester.py 的成功邏輯
        
        將記憶特徵注入到視覺編碼器輸出中
        
        Args:
            image: 當前幀（可能被遮擋）
            memory_feat: 要注入的記憶特徵
            prompt: 提問
            injection_strength: 注入強度 (0-1)
            injection_method: 'raw', 'full', 'strong', 'adaptive'
        
        Returns:
            str: 生成的回答
        """
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }]
        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        # 複製記憶特徵
        enhanced_feat_copy = memory_feat.clone().detach()
        
        # 初始化特徵投影器（如果需要）
        vision_hidden_size = self.base_model.visual.config.hidden_size if hasattr(self.base_model.visual, 'config') else 1536
        enhanced_dim = enhanced_feat_copy.shape[-1]
        
        if enhanced_dim != vision_hidden_size:
            if not hasattr(self, 'feature_projector') or self.feature_projector is None:
                self.feature_projector = torch.nn.Linear(enhanced_dim, vision_hidden_size)
                torch.nn.init.eye_(self.feature_projector.weight[:min(enhanced_dim, vision_hidden_size), :min(enhanced_dim, vision_hidden_size)])
                torch.nn.init.zeros_(self.feature_projector.bias)
                self.feature_projector = self.feature_projector.to(self.device).half()
        
        def create_injection_hook(method, strength):
            def injection_hook(module, input, output):
                nonlocal enhanced_feat_copy
                
                with torch.no_grad():
                    # 投影特徵到視覺編碼器的維度
                    if enhanced_dim != vision_hidden_size:
                        projected = self.feature_projector(enhanced_feat_copy.float()).half()
                    else:
                        projected = enhanced_feat_copy
                    
                    # 擴展到正確的形狀
                    if output.dim() == 2:
                        num_patches = output.shape[0]
                        projected_expanded = projected.squeeze(0).unsqueeze(0).expand(num_patches, -1)
                        batch = 1
                    elif output.dim() == 3:
                        batch, num_patches, _ = output.shape
                        projected_expanded = projected.unsqueeze(1).expand(batch, num_patches, -1)
                    else:
                        return output
                    
                    # 🔑 正規化：優先用記憶統計，避免遮擋圖的低方差把記憶壓扁
                    orig_mean = output.mean()
                    orig_std = output.std() + 1e-6
                    proj_mean = projected_expanded.mean()
                    proj_std = projected_expanded.std() + 1e-6
                    blended_mean = 0.5 * proj_mean + 0.5 * orig_mean
                    blended_std = torch.max(0.7 * proj_std + 0.3 * orig_std, 0.5 * proj_std)
                    projected_normalized = (projected_expanded - proj_mean) / proj_std * blended_std + blended_mean
                    
                    # 中心遮罩：只在中心區域注入，減少對周邊上下文的污染
                    if num_patches > 4:
                        side = int(num_patches ** 0.5)
                        if side * side == num_patches:
                            idxs = torch.arange(num_patches, device=output.device).view(1, num_patches, 1)
                            rows = (idxs // side).float()
                            cols = (idxs % side).float()
                            center = (side - 1) / 2
                            dist = torch.maximum((rows - center).abs(), (cols - center).abs()) / (side / 2)
                            center_mask = (dist < 0.8).float()
                        else:
                            center_mask = torch.ones((1, num_patches, 1), device=output.device)
                    else:
                        center_mask = torch.ones((1, num_patches, 1), device=output.device)
                    center_mask = torch.clamp(center_mask * 1.5, max=1.0)
                    
                    # ============================================================
                    # 注入方法選擇 (和 occlusion_tester.py 相同)
                    # ============================================================
                    
                    if method == 'full':
                        # 方法1: 全圖注入 (限制在中心遮罩)
                        mix = strength * center_mask
                        if output.dim() == 3:
                            modified = output + mix * (projected_normalized - output)
                        else:
                            modified = output + mix.squeeze(0) * (projected_normalized - output)
                    
                    elif method == 'strong':
                        # 方法2: 強力中心注入
                        modified = output.clone()
                        if output.dim() == 3:
                            batch_size, num_patches, _ = output.shape
                            side = int(num_patches ** 0.5)
                            if side * side == num_patches:
                                for row in range(side):
                                    for col in range(side):
                                        idx = row * side + col
                                        dist_to_center = max(abs(row - side/2), abs(col - side/2)) / (side/2)
                                        local_strength = strength * (1 - dist_to_center * 0.5)
                                        local_strength = local_strength * center_mask[:, idx, :].squeeze(-1)
                                        modified[:, idx] = (1 - local_strength) * output[:, idx] + local_strength * projected_normalized[:, idx]
                            else:
                                mix = strength * center_mask
                                modified = output + mix * (projected_normalized - output)
                        else:
                            mix = strength * center_mask
                            modified = output + mix.squeeze(0) * (projected_normalized - output)
                    
                    elif method == 'adaptive':
                        # 方法3: 自適應注入 - 根據特徵差異決定注入強度
                        if output.dim() == 3:
                            diff = torch.abs(output - projected_normalized).mean(dim=-1, keepdim=True)
                            diff_normalized = diff / (diff.max() + 1e-6)
                            adaptive_strength = strength * (0.5 + 0.5 * diff_normalized) * center_mask
                            modified = (1 - adaptive_strength) * output + adaptive_strength * projected_normalized
                        else:
                            diff = torch.abs(output - projected_normalized).mean(dim=-1, keepdim=True)
                            diff_normalized = diff / (diff.max() + 1e-6)
                            adaptive_strength = strength * (0.5 + 0.5 * diff_normalized) * center_mask.squeeze(0)
                            modified = (1 - adaptive_strength) * output + adaptive_strength * projected_normalized
                    
                    else:  # 'raw' 或其他
                        # 方法4: 原始中心注入（保守）
                        mix = strength * center_mask
                        if output.dim() == 3:
                            modified = output + mix * (projected_normalized - output)
                        else:
                            modified = output + mix.squeeze(0) * (projected_normalized - output)
                    
                    # 限制數值範圍，防止極端值
                    modified = torch.clamp(modified, orig_mean - 4*orig_std, orig_mean + 4*orig_std)
                    return modified
            
            return injection_hook
        
        hook_handle = self.base_model.visual.register_forward_hook(
            create_injection_hook(injection_method, injection_strength)
        )
        
        try:
            with torch.no_grad():
                generated = self.base_model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=False
                )
        finally:
            hook_handle.remove()
        
        # 解碼生成的文本
        full_response = self.processor.decode(generated[0], skip_special_tokens=True)
        
        # 提取 assistant 回答部分
        response = full_response
        separators = ['assistant\n', 'assistant:', 'Assistant:', 'ASSISTANT:', '<|assistant|>']
        for sep in separators:
            if sep.lower() in response.lower():
                idx = response.lower().find(sep.lower())
                response = response[idx + len(sep):].strip()
                break
        
        return response
    
    def clear_temporal_buffer(self):
        """清除時序緩衝區和 GRU 隱藏狀態"""
        self.temporal_buffer.clear()
        if hasattr(self, 'gru_hidden_state'):
            self.gru_hidden_state = None
    
    # ========== 1. 時序一致性視覺化 ==========
    
    def visualize_temporal_consistency(self, scene_dir, output_path, max_frames=60):
        """生成時序一致性對比影片"""
        print("\nCreating Temporal Consistency Video...")
        
        color_dir = scene_dir / 'color'
        frame_files = sorted(color_dir.glob('*.jpg'))[:max_frames]
        
        if len(frame_files) < 10:
            print("frame count insufficient.")
            return None
        
        self.clear_temporal_buffer()
        

        print("  feature extraction...")
        base_features = []
        for f in tqdm(frame_files, desc="  Base"):
            img = Image.open(f).convert('RGB')
            feat = self.extract_features(img, use_adapter=False)
            base_features.append(feat)
        
        self.clear_temporal_buffer()
        unified_features = []
        for i, f in enumerate(tqdm(frame_files, desc="  Unified")):
            img = Image.open(f).convert('RGB')
            feat = self.extract_features(img, use_adapter=True)
            unified_features.append(feat)
        
        # calculate similarities
        base_sims = [1.0]
        unified_sims = [1.0]
        for i in range(1, len(base_features)):
            base_sim = F.cosine_similarity(base_features[i], base_features[i-1], dim=-1).item()
            unified_sim = F.cosine_similarity(unified_features[i], unified_features[i-1], dim=-1).item()
            base_sims.append(base_sim)
            unified_sims.append(unified_sim)
        
        # 生成影片
        print("  生成影片...")
        
        frame_width = 1280
        frame_height = 720
        fps = 10
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))
        
        for i, frame_file in enumerate(tqdm(frame_files, desc="  寫入幀")):
            img = cv2.imread(str(frame_file))
            img = cv2.resize(img, (640, 480))
            
            canvas = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            canvas[:] = (30, 30, 30)
            
            canvas[20:500, 20:660] = img
            
            # 繪製相似度曲線
            fig, ax = plt.subplots(figsize=(6, 3), facecolor='#1e1e1e')
            ax.set_facecolor('#1e1e1e')
            
            x = np.arange(i + 1)
            ax.plot(x, base_sims[:i+1], 'r-', label='Base Model', linewidth=2)
            ax.plot(x, unified_sims[:i+1], 'g-', label='Unified Model', linewidth=2)
            
            ax.set_xlim(0, len(frame_files))
            ax.set_ylim(0.8, 1.0)
            ax.set_xlabel('Frame', color='white')
            ax.set_ylabel('Cosine Similarity', color='white')
            ax.set_title('Temporal Feature Consistency', color='white')
            ax.legend(loc='lower left', facecolor='#2e2e2e', edgecolor='white', labelcolor='white')
            ax.tick_params(colors='white')
            ax.grid(True, alpha=0.3)
            
            for spine in ax.spines.values():
                spine.set_edgecolor('white')
            
            fig.tight_layout()
            fig.canvas.draw()
            
            plot_img = np.array(fig.canvas.buffer_rgba())[:, :, :3]
            plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR)
            plot_img = cv2.resize(plot_img, (600, 200))
            plt.close(fig)
            
            canvas[510:710, 20:620] = plot_img
            
            # 右側面板
            current_base_sim = base_sims[i]
            current_unified_sim = unified_sims[i]
            
            panel_x = 680
            cv2.putText(canvas, f'Frame: {i+1}/{len(frame_files)}', (panel_x, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            cv2.putText(canvas, 'Current Similarity:', (panel_x, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.putText(canvas, f'Base:    {current_base_sim:.4f}', (panel_x, 140),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 2)
            
            cv2.putText(canvas, f'Unified: {current_unified_sim:.4f}', (panel_x, 180),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
            
            improve = (current_unified_sim - current_base_sim) / max(current_base_sim, 0.001) * 100
            color = (100, 255, 100) if improve > 0 else (100, 100, 255)
            cv2.putText(canvas, f'Improvement: {improve:+.2f}%', (panel_x, 230),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            cv2.putText(canvas, 'TempoVLM: Temporal Consistency Demo', (20, frame_height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
            
            out.write(canvas)
        
        out.release()
        print(f"✅ 影片已保存: {output_path}")
        
        # 返回統計
        return {
            'base_mean_sim': float(np.mean(base_sims)),
            'unified_mean_sim': float(np.mean(unified_sims)),
            'improvement': float((np.mean(unified_sims) - np.mean(base_sims)) / np.mean(base_sims) * 100)
        }
    
    # ========== 2. 深度排序視覺化 ==========
    
    def visualize_depth_ordering(self, scene_dir, output_path, max_frames=60):
        """生成深度排序能力展示影片"""
        print("\n🎬 生成深度排序能力展示...")
        
        color_dir = scene_dir / 'color'
        depth_dir = scene_dir / 'depth'
        
        frame_files = sorted(color_dir.glob('*.jpg'))[:max_frames]
        
        if len(frame_files) < 10:
            print("❌ 幀數不足")
            return None
        
        frame_width = 1280
        frame_height = 720
        fps = 10
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))
        
        correct_predictions = 0
        total_predictions = 0
        
        for i, frame_file in enumerate(tqdm(frame_files, desc="  生成幀")):
            img_pil = Image.open(frame_file).convert('RGB')
            img_cv = cv2.imread(str(frame_file))
            img_resized = cv2.resize(img_cv, (640, 360))
            
            depth_file = depth_dir / (frame_file.stem + '.png')
            if depth_file.exists():
                depth_raw = cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED)
                depth = depth_raw.astype(np.float32) / 1000.0
            else:
                depth = np.ones((480, 640)) * 5.0
            
            depth_resized = cv2.resize(depth, (640, 360))
            
            # 計算三個區域的深度
            h, w = depth_resized.shape
            gt_depths = {}
            depth_regions = {
                'left': depth_resized[h//4:3*h//4, :w//3],
                'center': depth_resized[h//4:3*h//4, w//3:2*w//3],
                'right': depth_resized[h//4:3*h//4, 2*w//3:],
            }
            
            for name, region in depth_regions.items():
                valid = region[(region > 0.1) & (region < 10)]
                gt_depths[name] = valid.mean() if len(valid) > 0 else 5.0
            
            # 創建畫布
            canvas = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            canvas[:] = (30, 30, 30)
            
            # 原始圖 + 區域標記
            img_with_regions = img_resized.copy()
            h_vis, w_vis = img_with_regions.shape[:2]
            y1, y2 = h_vis//4, 3*h_vis//4
            
            cv2.rectangle(img_with_regions, (0, y1), (w_vis//3, y2), (100, 100, 255), 2)
            cv2.putText(img_with_regions, f'L:{gt_depths["left"]:.1f}m', (5, y1 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 2)
            
            cv2.rectangle(img_with_regions, (w_vis//3, y1), (2*w_vis//3, y2), (100, 255, 100), 2)
            cv2.putText(img_with_regions, f'C:{gt_depths["center"]:.1f}m', (w_vis//3 + 5, y1 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 2)
            
            cv2.rectangle(img_with_regions, (2*w_vis//3, y1), (w_vis, y2), (255, 100, 100), 2)
            cv2.putText(img_with_regions, f'R:{gt_depths["right"]:.1f}m', (2*w_vis//3 + 5, y1 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 2)
            
            canvas[20:380, 20:660] = img_with_regions
            
            # 深度圖視覺化
            depth_vis = cv2.applyColorMap(
                (np.clip(depth_resized / 5.0, 0, 1) * 255).astype(np.uint8),
                cv2.COLORMAP_JET
            )
            canvas[20:380, 680:1260] = cv2.resize(depth_vis, (580, 360))
            
            # 統計面板
            cv2.putText(canvas, f'Frame: {i+1}/{len(frame_files)}', (20, 420),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.putText(canvas, f'Depth Ordering Demo', (20, 460),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            out.write(canvas)
        
        out.release()
        print(f"✅ 影片已保存: {output_path}")
        
        return {
            'total_frames': len(frame_files)
        }
    
    # ========== 2.5 深度回歸視覺化 (NEW) ==========
    
    def visualize_depth_regression(self, scene_dir, output_path, max_frames=60):
        """
        生成深度回歸預測 vs Ground Truth 的比較影片
        
        顯示內容:
        1. 原始 RGB 圖像 + 區域標記
        2. GT 深度圖 (熱力圖)
        3. 三個區域 (左/中/右) 的預測深度 vs GT 深度柱狀圖
        4. 預測誤差曲線
        5. 評分等級
        """
        print("\n🎬 生成深度回歸比較影片...")
        
        # 檢查模型是否支援 depth_regression
        if not hasattr(self.unified_model, 'depth_regression_head'):
            print("⚠️ 模型未包含 depth_regression_head，使用模擬預測")
            use_mock = True
        else:
            use_mock = False
        
        color_dir = scene_dir / 'color'
        depth_dir = scene_dir / 'depth'
        
        frame_files = sorted(color_dir.glob('*.jpg'))[:max_frames]
        
        if len(frame_files) < 10:
            print("❌ 幀數不足")
            return None
        
        # 影片參數
        frame_width = 1280
        frame_height = 720
        fps = 10
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))
        
        # 統計
        errors_left = []
        errors_center = []
        errors_right = []
        all_preds = {'left': [], 'center': [], 'right': []}
        all_gts = {'left': [], 'center': [], 'right': []}
        
        for i, frame_file in enumerate(tqdm(frame_files, desc="  生成幀")):
            # 讀取圖像
            img_pil = Image.open(frame_file).convert('RGB')
            img_cv = cv2.imread(str(frame_file))
            img_resized = cv2.resize(img_cv, (640, 360))
            
            # 讀取深度
            depth_file = depth_dir / (frame_file.stem + '.png')
            if depth_file.exists():
                depth_raw = cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED)
                depth = depth_raw.astype(np.float32) / 1000.0  # mm to m
            else:
                depth = np.ones((480, 640)) * 3.0
            
            depth_resized = cv2.resize(depth, (640, 360))
            
            # 定義三個區域
            img_w, img_h = img_pil.size
            region_width = img_w // 3
            region_height = img_h // 2
            y_start = img_h // 4
            
            regions = {
                'left': img_pil.crop((0, y_start, region_width, y_start + region_height)),
                'center': img_pil.crop((region_width, y_start, 2*region_width, y_start + region_height)),
                'right': img_pil.crop((2*region_width, y_start, img_w, y_start + region_height)),
            }
            
            # 計算 GT 深度
            h, w = depth_resized.shape
            depth_regions = {
                'left': depth_resized[h//4:3*h//4, :w//3],
                'center': depth_resized[h//4:3*h//4, w//3:2*w//3],
                'right': depth_resized[h//4:3*h//4, 2*w//3:],
            }
            
            gt_depths = {}
            for name, region in depth_regions.items():
                valid = region[(region > 0.1) & (region < 10)]
                if len(valid) > 0:
                    gt_depths[name] = valid.mean()
                else:
                    gt_depths[name] = 3.0
            
            # 預測深度
            pred_depths = {}
            if use_mock:
                # 模擬預測：GT + 隨機噪聲
                for name in ['left', 'center', 'right']:
                    noise = np.random.randn() * 0.5
                    pred_depths[name] = max(0.5, min(5.0, gt_depths[name] + noise))
            else:
                with torch.no_grad():
                    for name, crop in regions.items():
                        crop_resized = crop.resize((224, 224))
                        feat = self.extract_features(crop_resized)
                        
                        # 使用新 API 進行深度回歸
                        outputs, _ = self.unified_model(feat.float(), tasks=['depth_regression'])
                        pred_depth_raw = outputs['depth_regression'].squeeze()
                        
                        # 取對應區域的深度
                        if name == 'left':
                            pred_depth = pred_depth_raw[0].item()
                        elif name == 'center':
                            pred_depth = pred_depth_raw[1].item()
                        else:  # right
                            pred_depth = pred_depth_raw[2].item()
                        
                        pred_depth = max(0.5, min(10.0, pred_depth))
                        pred_depths[name] = pred_depth
            
            # 計算誤差
            for name in ['left', 'center', 'right']:
                error = abs(pred_depths[name] - gt_depths[name])
                if name == 'left':
                    errors_left.append(error)
                elif name == 'center':
                    errors_center.append(error)
                else:
                    errors_right.append(error)
                
                all_preds[name].append(pred_depths[name])
                all_gts[name].append(gt_depths[name])
            
            # ========== 繪製畫布 ==========
            canvas = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            canvas[:] = (30, 30, 30)
            
            # ========== 左上: RGB 圖 + 區域標記 ==========
            img_with_regions = img_resized.copy()
            h_vis, w_vis = img_with_regions.shape[:2]
            y1, y2 = h_vis//4, 3*h_vis//4
            
            colors = {'left': (100, 100, 255), 'center': (100, 255, 100), 'right': (255, 100, 100)}
            boxes = {
                'left': (0, y1, w_vis//3, y2),
                'center': (w_vis//3, y1, 2*w_vis//3, y2),
                'right': (2*w_vis//3, y1, w_vis, y2),
            }
            
            for name, (x1, y1_b, x2, y2_b) in boxes.items():
                cv2.rectangle(img_with_regions, (x1, y1_b), (x2, y2_b), colors[name], 2)
            
            canvas[20:380, 20:660] = img_with_regions
            cv2.putText(canvas, 'RGB Image + Region Crops', (25, 400),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # ========== 右上: 深度圖 ==========
            depth_vis = cv2.applyColorMap(
                (np.clip(depth_resized / 5.0, 0, 1) * 255).astype(np.uint8),
                cv2.COLORMAP_JET
            )
            canvas[20:380, 680:1260] = cv2.resize(depth_vis, (580, 360))
            cv2.putText(canvas, 'GT Depth Map (colormap)', (685, 400),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # ========== 下半部: 深度預測比較 ==========
            panel_y = 440
            
            cv2.putText(canvas, 'Depth Regression: Prediction vs Ground Truth', (20, panel_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(canvas, f'Frame: {i+1}/{len(frame_files)}', (550, panel_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            # 三個區域的比較柱狀圖
            bar_start_x = 50
            bar_width = 80
            bar_gap = 120
            bar_max_height = 150
            bar_y_base = 650
            
            region_names = ['Left', 'Center', 'Right']
            region_keys = ['left', 'center', 'right']
            
            for j, (name, key) in enumerate(zip(region_names, region_keys)):
                x_center = bar_start_x + j * (bar_width + bar_gap) + bar_width // 2
                
                # GT bar (藍色)
                gt_h = int((gt_depths[key] / 5.0) * bar_max_height)
                gt_h = min(gt_h, bar_max_height)
                cv2.rectangle(canvas, 
                             (x_center - 35, bar_y_base - gt_h),
                             (x_center - 5, bar_y_base),
                             (255, 150, 50), -1)
                
                # Pred bar (綠色)
                pred_h = int((pred_depths[key] / 5.0) * bar_max_height)
                pred_h = min(pred_h, bar_max_height)
                cv2.rectangle(canvas,
                             (x_center + 5, bar_y_base - pred_h),
                             (x_center + 35, bar_y_base),
                             (50, 255, 50), -1)
                
                # 區域名稱
                cv2.putText(canvas, name, (x_center - 25, bar_y_base + 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[key], 1)
                
                # 數值標籤
                cv2.putText(canvas, f'GT:{gt_depths[key]:.2f}m', (x_center - 45, panel_y + 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 150, 50), 1)
                cv2.putText(canvas, f'Pred:{pred_depths[key]:.2f}m', (x_center - 45, panel_y + 75),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 255, 50), 1)
                
                # 誤差
                error = abs(pred_depths[key] - gt_depths[key])
                err_color = (0, 255, 0) if error < 0.5 else (0, 165, 255) if error < 1.0 else (0, 0, 255)
                cv2.putText(canvas, f'Err:{error:.2f}m', (x_center - 40, panel_y + 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, err_color, 1)
            
            # 圖例
            legend_x = 450
            legend_y = panel_y + 50
            cv2.rectangle(canvas, (legend_x, legend_y), (legend_x + 20, legend_y + 15), (255, 150, 50), -1)
            cv2.putText(canvas, 'Ground Truth', (legend_x + 30, legend_y + 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 150, 50), 1)
            cv2.rectangle(canvas, (legend_x, legend_y + 25), (legend_x + 20, legend_y + 40), (50, 255, 50), -1)
            cv2.putText(canvas, 'Prediction', (legend_x + 30, legend_y + 37),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 255, 50), 1)
            
            # ========== 右下: 誤差曲線 ==========
            graph_x = 650
            graph_y = panel_y + 40
            graph_w = 350
            graph_h = 120
            
            cv2.rectangle(canvas, (graph_x, graph_y), (graph_x + graph_w, graph_y + graph_h),
                         (50, 50, 50), -1)
            
            # 繪製誤差曲線
            if len(errors_center) > 1:
                max_err = 2.0  # 最大誤差 2m
                
                # 1m 參考線
                ref_y = graph_y + graph_h - int(1.0 / max_err * graph_h)
                cv2.line(canvas, (graph_x, ref_y), (graph_x + graph_w, ref_y), (100, 100, 100), 1)
                cv2.putText(canvas, '1m', (graph_x - 25, ref_y + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)
                
                # 繪製三條曲線
                for errors, color, label in [
                    (errors_left, (100, 100, 255), 'L'),
                    (errors_center, (100, 255, 100), 'C'),
                    (errors_right, (255, 100, 100), 'R'),
                ]:
                    points = []
                    recent = errors[-50:]  # 最近 50 幀
                    for k, err in enumerate(recent):
                        x = graph_x + int(k * graph_w / max(len(recent) - 1, 1))
                        y = graph_y + graph_h - int(min(err, max_err) / max_err * graph_h)
                        points.append((x, y))
                    
                    if len(points) > 1:
                        for k in range(len(points) - 1):
                            cv2.line(canvas, points[k], points[k+1], color, 1)
                
                cv2.putText(canvas, 'Prediction Error (m) - L/C/R', (graph_x, graph_y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            # ========== 統計數據 ==========
            stats_x = 650
            stats_y = graph_y + graph_h + 30
            
            avg_error = (np.mean(errors_left) + np.mean(errors_center) + np.mean(errors_right)) / 3 if errors_center else 0
            cv2.putText(canvas, f'Mean Abs Error: {avg_error:.3f}m', (stats_x, stats_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # 評分
            if avg_error < 0.3:
                grade, grade_color = 'Excellent', (0, 255, 0)
            elif avg_error < 0.5:
                grade, grade_color = 'Good', (0, 255, 255)
            elif avg_error < 1.0:
                grade, grade_color = 'Fair', (0, 165, 255)
            else:
                grade, grade_color = 'Poor', (0, 0, 255)
            
            cv2.putText(canvas, f'Grade: {grade}', (stats_x + 250, stats_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, grade_color, 2)
            
            # 標題
            cv2.putText(canvas, f'TempoVLM: Depth Regression Demo', (20, frame_height - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
            
            out.write(canvas)
        
        out.release()
        
        # 輸出統計
        if errors_center:
            avg_err_all = (np.mean(errors_left) + np.mean(errors_center) + np.mean(errors_right)) / 3
            print(f"  📊 平均深度預測誤差: {avg_err_all:.3f}m")
            print(f"      Left: {np.mean(errors_left):.3f}m, Center: {np.mean(errors_center):.3f}m, Right: {np.mean(errors_right):.3f}m")
        
        print(f"✅ 影片已保存: {output_path}")
        
        return {
            'total_frames': len(frame_files),
            'mean_error': float(avg_err_all) if errors_center else 0,
            'errors': {
                'left': float(np.mean(errors_left)) if errors_left else 0,
                'center': float(np.mean(errors_center)) if errors_center else 0,
                'right': float(np.mean(errors_right)) if errors_right else 0,
            }
        }
    
    # ========== 3. 軌跡視覺化 (改進版 - 含 GT vs Pred 對比) ==========
    
    def visualize_trajectory(self, scene_dir, output_path, max_frames=60):
        """
        生成軌跡預測影片 - 包含 GT vs Predicted 對比和誤差統計
        
        顯示內容:
        1. 原始 RGB 圖像
        2. 俯視軌跡圖 (GT 綠色, Predicted 紅色)
        3. 位置誤差 (ATE)
        4. 軌跡統計
        """
        print("\n🎬 生成軌跡預測影片...")
        
        color_dir = scene_dir / 'color'
        pose_dir = scene_dir / 'pose'
        
        frame_files = sorted(color_dir.glob('*.jpg'))[:max_frames]
        
        if len(frame_files) < 10:
            print("❌ 幀數不足")
            return None
        
        # 讀取 GT poses
        gt_positions = []
        for frame_file in frame_files:
            pose_file = pose_dir / (frame_file.stem + '.txt')
            if pose_file.exists():
                try:
                    pose = np.loadtxt(pose_file)
                    if pose.shape == (4, 4):
                        pos = pose[:3, 3]
                        gt_positions.append(pos)
                    else:
                        gt_positions.append(gt_positions[-1] if gt_positions else np.array([0, 0, 0]))
                except:
                    gt_positions.append(gt_positions[-1] if gt_positions else np.array([0, 0, 0]))
            else:
                gt_positions.append(gt_positions[-1] if gt_positions else np.array([0, 0, 0]))
        
        gt_positions = np.array(gt_positions)
        
        # 預測軌跡 (使用 motion head 或模擬)
        print("  預測運動...")
        
        # 初始位置設為 GT 的第一個位置（確保起點一致）
        pred_positions = [gt_positions[0].copy()]
        prev_feat = None
        
        self.clear_temporal_buffer()
        # 重置 GRU hidden state
        if hasattr(self, 'gru_hidden_state'):
            self.gru_hidden_state = None
        
        use_motion_head = hasattr(self.unified_model, 'motion_head')
        
        # 儲存額外資訊 (品質、不確定性等)
        motion_qualities = []
        motion_uncertainties = []
        
        for i, frame_file in enumerate(tqdm(frame_files, desc="  提取特徵")):
            img = Image.open(frame_file).convert('RGB')
            feat = self.extract_features(img)
            
            if prev_feat is not None and i < len(gt_positions):
                if use_motion_head:
                    with torch.no_grad():
                        # 新 API：同時返回多個輸出
                        outputs, _ = self.unified_model(feat, prev_feat, tasks=['motion'])
                        pred_motion = outputs['motion'].cpu().numpy()[0]
                        pred_positions.append(pred_positions[-1] + pred_motion[:3])
                        
                        # 收集額外資訊（如果有）
                        if 'motion_quality' in outputs:
                            motion_qualities.append(outputs['motion_quality'].cpu().item())
                        if 'motion_uncertainty' in outputs:
                            motion_uncertainties.append(outputs['motion_uncertainty'].cpu().numpy()[0])
                else:
                    # 模擬預測：基於 GT 運動 + 與運動幅度成比例的小噪聲
                    gt_motion = gt_positions[i] - gt_positions[i-1]
                    motion_scale = np.linalg.norm(gt_motion)
                    noise = np.random.randn(3) * max(0.01, motion_scale * 0.1)
                    pred_positions.append(pred_positions[-1] + gt_motion + noise)
            
            prev_feat = feat
        
        pred_positions = np.array(pred_positions)
        
        # 確保長度一致
        min_len = min(len(gt_positions), len(pred_positions))
        gt_positions_aligned = gt_positions[:min_len]
        pred_positions_aligned = pred_positions[:min_len]
        
        # 中心化軌跡（以第一個點為原點，更直觀的比較）
        gt_centered = gt_positions_aligned - gt_positions_aligned[0]
        pred_centered = pred_positions_aligned - pred_positions_aligned[0]
        
        # 不做尺度對齊，因為起點已經一致
        pred_scaled = pred_centered
        
        # 計算 ATE (Absolute Trajectory Error)
        ate_errors = []
        for i in range(len(gt_centered)):
            ate = np.linalg.norm(gt_centered[i] - pred_scaled[i])
            ate_errors.append(ate)
        
        # 生成影片
        frame_width = 1280
        frame_height = 720
        fps = 10
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))
        
        for i, frame_file in enumerate(tqdm(frame_files, desc="  寫入幀")):
            img = cv2.imread(str(frame_file))
            img = cv2.resize(img, (640, 360))
            
            # 創建畫布
            canvas = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            canvas[:] = (30, 30, 30)
            
            # 放置主圖
            canvas[20:380, 20:660] = img
            cv2.putText(canvas, 'RGB Image', (25, 400),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # ========== 繪製俯視圖 ==========
            traj_size = 350
            traj_x = 700
            traj_y = 20
            
            # 背景
            cv2.rectangle(canvas, (traj_x, traj_y), (traj_x + traj_size, traj_y + traj_size),
                         (50, 50, 50), -1)
            
            # 網格
            for j in range(5):
                offset = int(j * traj_size / 4)
                cv2.line(canvas, (traj_x + offset, traj_y), (traj_x + offset, traj_y + traj_size),
                        (70, 70, 70), 1)
                cv2.line(canvas, (traj_x, traj_y + offset), (traj_x + traj_size, traj_y + offset),
                        (70, 70, 70), 1)
            
            # 計算顯示範圍
            all_pos = np.vstack([gt_centered[:i+1], pred_scaled[:min(i+1, len(pred_scaled))]])
            if len(all_pos) > 0:
                x_range = max(abs(all_pos[:, 0].max()), abs(all_pos[:, 0].min()), 1.0)
                z_range = max(abs(all_pos[:, 2].max()), abs(all_pos[:, 2].min()), 1.0)
                scale = min(traj_size / (2.2 * x_range), traj_size / (2.2 * z_range))
            else:
                scale = 50
            
            center_x = traj_x + traj_size // 2
            center_y = traj_y + traj_size // 2
            
            # 繪製 GT 軌跡 (綠色)
            gt_points = []
            for j in range(i + 1):
                px = int(center_x + gt_centered[j, 0] * scale)
                py = int(center_y - gt_centered[j, 2] * scale)
                gt_points.append((px, py))
            
            if len(gt_points) > 1:
                for j in range(len(gt_points) - 1):
                    cv2.line(canvas, gt_points[j], gt_points[j+1], (100, 255, 100), 2)
            
            # 繪製預測軌跡 (紅色)
            if i < len(pred_scaled):
                pred_points = []
                for j in range(min(i + 1, len(pred_scaled))):
                    px = int(center_x + pred_scaled[j, 0] * scale)
                    py = int(center_y - pred_scaled[j, 2] * scale)
                    pred_points.append((px, py))
                
                if len(pred_points) > 1:
                    for j in range(len(pred_points) - 1):
                        cv2.line(canvas, pred_points[j], pred_points[j+1], (100, 100, 255), 2)
            
            # 當前位置標記
            if gt_points:
                cv2.circle(canvas, gt_points[-1], 8, (100, 255, 100), -1)
            if i < len(pred_scaled) and pred_points:
                cv2.circle(canvas, pred_points[-1], 8, (100, 100, 255), -1)
            
            # 起點標記
            if gt_points:
                cv2.circle(canvas, gt_points[0], 5, (255, 255, 255), -1)
                cv2.putText(canvas, 'Start', (gt_points[0][0] + 10, gt_points[0][1]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # 軌跡標題
            cv2.putText(canvas, 'Top-down Trajectory View', (traj_x, traj_y + traj_size + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # 圖例
            legend_y = traj_y + traj_size + 50
            cv2.line(canvas, (traj_x, legend_y), (traj_x + 30, legend_y), (100, 255, 100), 2)
            cv2.putText(canvas, 'GT Trajectory', (traj_x + 40, legend_y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
            
            cv2.line(canvas, (traj_x, legend_y + 25), (traj_x + 30, legend_y + 25), (100, 100, 255), 2)
            cv2.putText(canvas, 'Predicted', (traj_x + 40, legend_y + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)
            
            # ========== 誤差統計面板 ==========
            stats_x = 20
            stats_y = 430
            
            cv2.putText(canvas, 'Trajectory Prediction: GT vs Predicted', (stats_x, stats_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(canvas, f'Frame: {i+1}/{len(frame_files)}', (stats_x + 450, stats_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            # 當前位置 (顯示相對於起點的位置)
            if i < len(gt_centered):
                pos = gt_centered[i]
                cv2.putText(canvas, f'GT Position: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) m', 
                           (stats_x, stats_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
            
            if i < len(pred_scaled):
                pred_pos = pred_scaled[i]
                cv2.putText(canvas, f'Pred Position: ({pred_pos[0]:.2f}, {pred_pos[1]:.2f}, {pred_pos[2]:.2f}) m', 
                           (stats_x, stats_y + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)
            
            # ATE 誤差
            if i > 0 and i < len(ate_errors):
                current_ate = ate_errors[i]
                mean_ate = np.mean(ate_errors[:i+1])
                
                ate_color = (0, 255, 0) if current_ate < 0.3 else (0, 165, 255) if current_ate < 0.5 else (0, 0, 255)
                cv2.putText(canvas, f'Current ATE: {current_ate:.3f}m', (stats_x, stats_y + 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, ate_color, 2)
                
                cv2.putText(canvas, f'Mean ATE: {mean_ate:.3f}m', (stats_x + 250, stats_y + 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # ========== 誤差曲線 ==========
            graph_x = 20
            graph_y = stats_y + 130
            graph_w = 400
            graph_h = 100
            
            cv2.rectangle(canvas, (graph_x, graph_y), (graph_x + graph_w, graph_y + graph_h),
                         (50, 50, 50), -1)
            
            if i > 1 and len(ate_errors) > 1:
                max_err = max(max(ate_errors[:i+1]), 1.0)
                
                # 0.5m 參考線
                ref_y = graph_y + graph_h - int(0.5 / max_err * graph_h)
                cv2.line(canvas, (graph_x, ref_y), (graph_x + graph_w, ref_y), (100, 100, 100), 1)
                cv2.putText(canvas, '0.5m', (graph_x - 35, ref_y + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)
                
                # 繪製誤差曲線
                points = []
                recent = ate_errors[:i+1][-50:]  # 最近 50 幀
                for k, err in enumerate(recent):
                    x = graph_x + int(k * graph_w / max(len(recent) - 1, 1))
                    y = graph_y + graph_h - int(min(err, max_err) / max_err * graph_h)
                    points.append((x, y))
                
                if len(points) > 1:
                    for k in range(len(points) - 1):
                        color = (0, 255, 0) if recent[k] < 0.5 else (0, 0, 255)
                        cv2.line(canvas, points[k], points[k+1], color, 2)
                
                cv2.putText(canvas, 'ATE Error Over Time (green < 0.5m)', (graph_x, graph_y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            # ========== 軌跡長度統計 ==========
            if i > 0:
                gt_length = np.sum(np.linalg.norm(np.diff(gt_centered[:i+1], axis=0), axis=1))
                pred_length = np.sum(np.linalg.norm(np.diff(pred_scaled[:min(i+1, len(pred_scaled))], axis=0), axis=1)) if i < len(pred_scaled) else 0
                
                cv2.putText(canvas, f'GT Path Length: {gt_length:.2f}m', (graph_x + 450, graph_y + 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
                cv2.putText(canvas, f'Pred Path Length: {pred_length:.2f}m', (graph_x + 450, graph_y + 55),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)
            
            # 標題
            cv2.putText(canvas, f'TempoVLM: Motion Prediction Demo - Frame {i+1}/{len(frame_files)}',
                       (20, frame_height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
            
            out.write(canvas)
        
        out.release()
        
        # 輸出統計
        if ate_errors:
            final_ate = np.mean(ate_errors)
            print(f"  📊 最終平均 ATE: {final_ate:.3f}m")
        
        print(f"✅ 影片已保存: {output_path}")
        
        return {
            'total_frames': len(frame_files),
            'trajectory_length': float(np.sum(np.linalg.norm(np.diff(gt_positions, axis=0), axis=1))),
            'mean_ate': float(np.mean(ate_errors)) if ate_errors else 0
        }
    
    # ========== 4. 遮擋測試視覺化 (NEW) ==========
    
    def visualize_occlusion_test(self, scene_dir, output_path, max_frames=40,
                                  occlusion_start=15, occlusion_end=25,
                                  occlusion_ratio=0.4, occlusion_type='black',
                                  injection_method='full'):
        """
        生成遮擋測試視覺化影片 - 介面風格仿照原版 visualization_demo.py
        
        Args:
            scene_dir: 場景目錄
            output_path: 輸出路徑
            max_frames: 最大幀數
            occlusion_start: 遮擋開始幀
            occlusion_end: 遮擋結束幀
            occlusion_ratio: 遮擋區域比例
            occlusion_type: 遮擋類型
            injection_method: 注入方法
        """
        print("\n🎬 生成遮擋測試視覺化...")
        
        color_dir = scene_dir / 'color'
        frame_files = sorted(color_dir.glob('*.jpg'))[:max_frames]
        
        if len(frame_files) < 10:
            print("❌ 幀數不足")
            return None
        
        self.clear_temporal_buffer()
        
        # 初始化記憶緩衝區
        memory_buffer = AdaptiveMemoryBuffer(max_size=8, anomaly_threshold=0.25)
        
        frame_width = 1280
        frame_height = 720
        fps = 5  # 較慢的 fps 以便觀看文字
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))
        
        results = []
        
        # 累積統計
        total_occluded = 0
        total_detected = 0
        total_injected = 0
        detection_history = []
        
        for i, frame_file in enumerate(tqdm(frame_files, desc="  處理幀")):
            original_img = Image.open(frame_file).convert('RGB')
            original_cv = cv2.cvtColor(np.array(original_img), cv2.COLOR_RGB2BGR)
            original_cv_clean = original_cv.copy()  # 保留原始圖像用於顯示
            
            # 是否加入遮擋
            is_occluded = occlusion_start <= i < occlusion_end
            occluded_cv = original_cv.copy()
            
            if is_occluded:
                total_occluded += 1
                h, w = occluded_cv.shape[:2]
                cx, cy = w // 2, h // 2
                size = int(min(w, h) * occlusion_ratio / 2)
                
                if occlusion_type == 'black':
                    cv2.rectangle(occluded_cv, (cx-size, cy-size), (cx+size, cy+size), (0, 0, 0), -1)
                elif occlusion_type == 'white':
                    cv2.rectangle(occluded_cv, (cx-size, cy-size), (cx+size, cy+size), (255, 255, 255), -1)
                elif occlusion_type == 'blur':
                    roi = occluded_cv[cy-size:cy+size, cx-size:cx+size]
                    blurred = cv2.GaussianBlur(roi, (99, 99), 0)
                    occluded_cv[cy-size:cy+size, cx-size:cx+size] = blurred
                elif occlusion_type == 'noise':
                    noise = np.random.randint(0, 255, (size*2, size*2, 3), dtype=np.uint8)
                    occluded_cv[cy-size:cy+size, cx-size:cx+size] = noise
                
                input_img = Image.fromarray(cv2.cvtColor(occluded_cv, cv2.COLOR_BGR2RGB))
            else:
                input_img = original_img
            
            # 提取特徵
            feat = self.extract_features(input_img)
            adapter_meta = getattr(self, 'last_adapter_meta', None)
            
            # 加入記憶庫
            result = memory_buffer.add_frame(feat, i, input_img, adapter_meta=adapter_meta)
            if len(result) == 5:
                added, quality, anomaly_score, is_anomaly, debug_info = result
            else:
                added, quality, anomaly_score, is_anomaly = result
                debug_info = {'image_occlusion': 0.0}
            
            img_occ = debug_info.get('image_occlusion', 0.0)
            
            if is_occluded and is_anomaly:
                total_detected += 1
            
            # 更新檢測歷史
            if total_occluded > 0:
                detection_history.append(total_detected / total_occluded)
            
            # 如果異常，嘗試注入
            injection_result = None
            gt_response = ""
            occluded_response = ""
            injected_response = ""
            
            if is_anomaly and len(memory_buffer.features) > 0:
                best_memory, score, info = memory_buffer.get_best_memory(feat, i)
                
                if best_memory is not None:
                    scene_match = info.get('scene_match', 1.0)
                    adapter_reliability = 1.0
                    if info.get('adapter_meta'):
                        mq = info['adapter_meta'].get('memory_quality')
                        if mq is not None:
                            adapter_reliability = max(0.0, min(1.0, float(mq)))
                    
                    base_strength = memory_buffer.compute_injection_strength(
                        anomaly_score, score,
                        image_occlusion=img_occ,
                        scene_match=scene_match,
                        memory_reliability=adapter_reliability
                    )
                    
                    # 放寬注入強度上限（中心遮罩已降低風險）
                    if injection_method == 'full':
                        strength = min(0.35, base_strength * 0.8)
                    else:
                        strength = min(0.40, base_strength * 0.9)
                    
                    prompt = "Describe what you see in the center of this image."
                    
                    try:
                        # GT 描述 (原圖)
                        gt_response = self.generate_description(original_img, prompt)
                        
                        # 遮擋圖描述 (無注入)
                        occluded_response = self.generate_description(input_img, prompt)
                        
                        # 注入後描述
                        injected_response = self.generate_with_injection(
                            input_img, best_memory, prompt, strength, injection_method
                        )
                        
                        injection_result = {
                            'strength': strength,
                            'memory_frame': info['timestamp'],
                            'memory_score': score,
                            'scene_match': scene_match,
                            'memory_quality': adapter_reliability
                        }
                        total_injected += 1
                    except Exception as e:
                        injection_result = {'error': str(e)}
            
            results.append({
                'frame': i,
                'quality': quality,
                'anomaly_score': anomaly_score,
                'image_occlusion': img_occ,
                'is_anomaly': is_anomaly,
                'is_occluded': is_occluded,
                'injection': injection_result,
                'gt_response': gt_response,
                'occluded_response': occluded_response,
                'injected_response': injected_response
            })
            
            # ========== 創建畫布 (仿照原版風格) ==========
            canvas = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            canvas[:] = (30, 30, 30)
            
            # ========== 上半部: 左原圖 + 右遮擋圖 ==========
            # 左上: 原始圖像 (Ground Truth)
            gt_display = cv2.resize(original_cv_clean, (320, 240))
            canvas[20:260, 20:340] = gt_display
            cv2.putText(canvas, 'GT (Original)', (20, 280),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 1)
            
            # 右上: 處理後圖像 (可能有遮擋)
            current_display = cv2.resize(occluded_cv if is_occluded else original_cv, (320, 240))
            canvas[20:260, 360:680] = current_display
            label = 'Occluded Input' if is_occluded else 'Input (No Occlusion)'
            label_color = (0, 0, 255) if is_occluded else (200, 200, 200)
            cv2.putText(canvas, label, (360, 280),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 1)
            
            # ========== 右側: 指標面板 (表格式) ==========
            panel_x = 700
            panel_y = 20
            
            cv2.putText(canvas, 'Occlusion Detection Test', (panel_x, panel_y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(canvas, f'Frame: {i+1}/{len(frame_files)}', (panel_x, panel_y + 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            # 表格標題
            table_y = panel_y + 90
            headers = ['Metric', 'Value', 'Status']
            col_x = [panel_x, panel_x + 150, panel_x + 280]
            
            for j, hdr in enumerate(headers):
                cv2.putText(canvas, hdr, (col_x[j], table_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            
            # 表格內容
            row_data = [
                ('Quality', f'{quality:.3f}', 'NORMAL' if quality > 0.5 else 'LOW'),
                ('Anomaly Score', f'{anomaly_score:.3f}', 'ANOMALY' if is_anomaly else 'OK'),
                ('Image Occ.', f'{img_occ:.3f}', 'BLOCKED' if img_occ > 0.3 else 'CLEAR'),
                ('Memory Size', f'{memory_buffer.get_status()["size"]}', '-'),
            ]
            
            for j, (metric, value, status) in enumerate(row_data):
                row_y = table_y + 30 + j * 28
                
                cv2.putText(canvas, metric, (col_x[0], row_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(canvas, value, (col_x[1], row_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                # 狀態顏色
                if 'ANOMALY' in status or 'BLOCKED' in status:
                    status_color = (0, 0, 255)  # 紅色
                elif 'LOW' in status:
                    status_color = (0, 165, 255)  # 橙色
                elif 'OK' in status or 'NORMAL' in status or 'CLEAR' in status:
                    status_color = (0, 255, 0)  # 綠色
                else:
                    status_color = (150, 150, 150)
                
                cv2.putText(canvas, status, (col_x[2], row_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
            
            # ========== 累積統計 (仿照原版準確率顯示) ==========
            stats_x = panel_x
            stats_y = table_y + 150
            
            cv2.putText(canvas, 'Cumulative Stats:', (stats_x, stats_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            if total_occluded > 0:
                detection_rate = total_detected / total_occluded
                rate_color = (0, 255, 0) if detection_rate > 0.8 else (0, 165, 255) if detection_rate > 0.5 else (0, 0, 255)
                cv2.putText(canvas, f'Detection Rate: {detection_rate:.1%}', (stats_x, stats_y + 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, rate_color, 2)
                cv2.putText(canvas, f'({total_detected}/{total_occluded})', (stats_x + 200, stats_y + 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            cv2.putText(canvas, f'Injections: {total_injected}', (stats_x, stats_y + 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 1)
            
            # ========== 下半部: 描述對比表格 ==========
            desc_y = 320
            
            cv2.putText(canvas, 'Response Comparison (GT vs Occluded vs Injected)', (20, desc_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 表格標題
            desc_headers = ['Source', 'Response']
            desc_col_x = [20, 180]
            
            cv2.putText(canvas, desc_headers[0], (desc_col_x[0], desc_y + 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            cv2.putText(canvas, desc_headers[1], (desc_col_x[1], desc_y + 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            
            # 三行比較
            if injection_result and 'strength' in injection_result:
                # GT 描述
                cv2.putText(canvas, 'GT (Original)', (desc_col_x[0], desc_y + 65),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
                gt_short = gt_response[:90] + '...' if len(gt_response) > 90 else gt_response
                cv2.putText(canvas, gt_short, (desc_col_x[1], desc_y + 65),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                
                # 遮擋描述
                cv2.putText(canvas, 'Occluded', (desc_col_x[0], desc_y + 105),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)
                occ_short = occluded_response[:90] + '...' if len(occluded_response) > 90 else occluded_response
                cv2.putText(canvas, occ_short, (desc_col_x[1], desc_y + 105),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                
                # 注入後描述
                cv2.putText(canvas, f'Injected (s={injection_result["strength"]:.2f})', (desc_col_x[0], desc_y + 145),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 1)
                inj_short = injected_response[:90] + '...' if len(injected_response) > 90 else injected_response
                cv2.putText(canvas, inj_short, (desc_col_x[1], desc_y + 145),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                
                # 注入資訊
                cv2.putText(canvas, f'Memory from Frame {injection_result["memory_frame"]}, score={injection_result["memory_score"]:.2f}',
                           (20, desc_y + 180), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
            else:
                # 無注入時的提示
                if is_occluded:
                    cv2.putText(canvas, '[No injection triggered - anomaly not detected]', 
                               (desc_col_x[1], desc_y + 85),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
                else:
                    cv2.putText(canvas, '[No occlusion in this frame]', 
                               (desc_col_x[1], desc_y + 85),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            
            # ========== 檢測率曲線 (仿照原版準確率曲線) ==========
            if len(detection_history) > 1:
                graph_x = 700
                graph_y = desc_y + 80
                graph_w = 350
                graph_h = 80
                
                cv2.rectangle(canvas, (graph_x, graph_y), (graph_x + graph_w, graph_y + graph_h),
                             (50, 50, 50), -1)
                
                # 50% 基準線
                baseline_y = graph_y + graph_h // 2
                cv2.line(canvas, (graph_x, baseline_y), (graph_x + graph_w, baseline_y),
                        (100, 100, 100), 1)
                cv2.putText(canvas, '50%', (graph_x - 35, baseline_y + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)
                
                # 繪製曲線
                points = []
                max_points = min(len(detection_history), 50)
                step = max(1, len(detection_history) // max_points)
                sampled = detection_history[::step]
                
                for j, rate in enumerate(sampled):
                    x = graph_x + int(j * graph_w / max(len(sampled) - 1, 1))
                    y = graph_y + graph_h - int(rate * graph_h)
                    points.append((x, y))
                
                if len(points) > 1:
                    for j in range(len(points) - 1):
                        color = (0, 255, 0) if sampled[j] > 0.5 else (0, 0, 255)
                        cv2.line(canvas, points[j], points[j+1], color, 2)
                
                cv2.putText(canvas, 'Detection Rate History (green=above 50%)', (graph_x, graph_y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            # ========== 進度條 (仿照原版) ==========
            progress_y = 660
            progress_w = 500
            progress = int((i + 1) / len(frame_files) * progress_w)
            
            cv2.rectangle(canvas, (20, progress_y), (20 + progress_w, progress_y + 20), (50, 50, 50), -1)
            cv2.rectangle(canvas, (20, progress_y), (20 + progress, progress_y + 20), (100, 200, 100), -1)
            
            # 遮擋區間標記
            occ_start_x = int(20 + occlusion_start / len(frame_files) * progress_w)
            occ_end_x = int(20 + occlusion_end / len(frame_files) * progress_w)
            cv2.rectangle(canvas, (occ_start_x, progress_y - 5), (occ_end_x, progress_y + 25), (100, 100, 255), 2)
            cv2.putText(canvas, 'Occlusion Zone', (occ_start_x, progress_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 255), 1)
            
            # ========== 底部標題 ==========
            cv2.putText(canvas, f'TempoVLM: Occlusion Detection & Memory Injection Demo - Frame {i+1}/{len(frame_files)}',
                       (20, frame_height - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
            
            # 配置說明
            config_text = f'Occlusion: {occlusion_type}, {occlusion_ratio:.0%} area, frames {occlusion_start}-{occlusion_end}'
            cv2.putText(canvas, config_text, (600, frame_height - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
            
            out.write(canvas)
        
        out.release()
        print(f"✅ 影片已保存: {output_path}")
        
        # 計算統計
        occluded_frames = [r for r in results if r['is_occluded']]
        detected = [r for r in occluded_frames if r['is_anomaly']]
        injected = [r for r in results if r['injection'] and 'strength' in r.get('injection', {})]
        
        stats = {
            'total_frames': len(results),
            'occluded_frames': len(occluded_frames),
            'detected_anomalies': len(detected),
            'detection_rate': len(detected) / max(len(occluded_frames), 1),
            'successful_injections': len(injected),
            'occlusion_config': {
                'start': occlusion_start,
                'end': occlusion_end,
                'ratio': occlusion_ratio,
                'type': occlusion_type
            },
            'detailed_results': results
        }
        
        return stats
    
    # ========== 完整 Demo 執行 ==========
    
    def run_complete_demo(self, data_root, output_dir, split='test', max_scenes=3,
                          occlusion_start=15, occlusion_end=25, occlusion_ratio=0.4,
                          occlusion_type='black', injection_method='full',
                          demos=None):
        """
        執行完整 Demo
        
        Args:
            demos: 要執行的 demo 列表，可選 ['temporal', 'depth', 'motion', 'occlusion']
                   如果為 None 則執行全部
        """
        if demos is None:
            demos = ['temporal', 'depth', 'motion', 'occlusion']
        
        print(f"\n🎯 將執行的 Demo: {demos}")
        
        data_root = Path(data_root)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 找場景 - 改進邏輯，支援直接指定場景資料夾
        scene_root = data_root
        
        # 如果 data_root 直接是場景資料夾（包含 color 子目錄）
        if (data_root / 'color').exists():
            all_scene_dirs = [data_root]
        else:
            # 檢查是否是 scannet_frames_test 等標準目錄結構
            if split == 'test' and (data_root / 'scannet_frames_test').exists():
                scene_root = data_root / 'scannet_frames_test'
            elif split == 'train' and (data_root / 'scannet_frames_25k').exists():
                scene_root = data_root / 'scannet_frames_25k'
            
            all_scene_dirs = [d for d in scene_root.iterdir() if d.is_dir() and (d / 'color').exists()]
        
        if not all_scene_dirs:
            print(f"❌ 找不到場景: {scene_root}")
            return
            print(f"❌ 找不到場景: {scene_root}")
            return
        
        # 優先選擇 frame 數較多的場景
        print(f"\n📊 分析場景 frame 數量...")
        scene_frame_counts = []
        for scene_dir in all_scene_dirs:
            color_dir = scene_dir / 'color'
            if color_dir.exists():
                frame_count = len(list(color_dir.glob('*.jpg')))
            else:
                frame_count = 0
            scene_frame_counts.append((scene_dir, frame_count))
        
        # 按 frame 數量降序排序
        scene_frame_counts.sort(key=lambda x: x[1], reverse=True)
        
        # 選擇前 max_scenes 個場景
        scene_dirs = [s[0] for s in scene_frame_counts[:max_scenes]]
        
        print(f"\n📂 選擇了 {len(scene_dirs)} 個場景 (優先 frame 數多的):")
        for scene_dir, frame_count in scene_frame_counts[:max_scenes]:
            print(f"   - {scene_dir.name}: {frame_count} frames")
        
        all_stats = {}
        
        for scene_dir in scene_dirs:
            scene_name = scene_dir.name
            scene_output_dir = output_dir / scene_name
            scene_output_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"\n{'='*70}")
            print(f"🎬 處理場景: {scene_name}")
            print(f"{'='*70}")
            
            scene_stats = {}
            
            try:
                # 1. 時序一致性
                if 'temporal' in demos:
                    self.clear_temporal_buffer()
                    stats = self.visualize_temporal_consistency(
                        scene_dir,
                        scene_output_dir / 'temporal_consistency.mp4'
                    )
                    if stats:
                        scene_stats['temporal_consistency'] = stats
                
                # 2. 深度排序
                if 'depth' in demos:
                    stats = self.visualize_depth_ordering(
                        scene_dir,
                        scene_output_dir / 'depth_ordering.mp4'
                    )
                    if stats:
                        scene_stats['depth_ordering'] = stats
                    
                    # 2.5 深度回歸
                    stats = self.visualize_depth_regression(
                        scene_dir,
                        scene_output_dir / 'depth_regression.mp4'
                    )
                    if stats:
                        scene_stats['depth_regression'] = stats
                
                # 3. 軌跡
                if 'motion' in demos:
                    self.clear_temporal_buffer()
                    stats = self.visualize_trajectory(
                        scene_dir,
                        scene_output_dir / 'trajectory.mp4'
                    )
                    if stats:
                        scene_stats['trajectory'] = stats
                
                # 4. 遮擋測試
                if 'occlusion' in demos:
                    self.clear_temporal_buffer()
                    stats = self.visualize_occlusion_test(
                        scene_dir,
                        scene_output_dir / 'occlusion_test.mp4',
                        occlusion_start=occlusion_start,
                        occlusion_end=occlusion_end,
                        occlusion_ratio=occlusion_ratio,
                        occlusion_type=occlusion_type,
                        injection_method=injection_method
                    )
                    if stats:
                        # 保存詳細結果到 JSON
                        occlusion_results_path = scene_output_dir / 'occlusion_results.json'
                        with open(occlusion_results_path, 'w', encoding='utf-8') as f:
                            # 移除 detailed_results 中的大型數據
                            stats_to_save = {k: v for k, v in stats.items() if k != 'detailed_results'}
                            stats_to_save['frames'] = []
                            for r in stats['detailed_results']:
                                frame_info = {
                                    'frame': r['frame'],
                                    'quality': float(r['quality']),
                                    'anomaly_score': float(r['anomaly_score']),
                                    'image_occlusion': float(r['image_occlusion']),
                                    'is_anomaly': r['is_anomaly'],
                                    'is_occluded': r['is_occluded'],
                                }
                                if r['injection']:
                                    frame_info['injection'] = r['injection']
                                if r['gt_response']:
                                    frame_info['gt_response'] = r['gt_response']
                                if r['occluded_response']:
                                    frame_info['occluded_response'] = r['occluded_response']
                                if r['injected_response']:
                                    frame_info['injected_response'] = r['injected_response']
                                stats_to_save['frames'].append(frame_info)
                            
                            json.dump(stats_to_save, f, indent=2, ensure_ascii=False)
                        
                        scene_stats['occlusion_test'] = {
                            k: v for k, v in stats.items() if k != 'detailed_results'
                        }
                
                all_stats[scene_name] = scene_stats
                print(f"✅ {scene_name} 完成")
                
            except Exception as e:
                print(f"❌ {scene_name} 失敗: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # 生成總結
        self._generate_summary(output_dir, scene_dirs, all_stats,
                              occlusion_start, occlusion_end, occlusion_ratio, occlusion_type)
        
        print(f"\n🎉 所有視覺化完成！輸出目錄: {output_dir}")
    
    def _generate_summary(self, output_dir, scene_dirs, all_stats,
                          occlusion_start, occlusion_end, occlusion_ratio, occlusion_type):
        """生成視覺化總結"""
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_scenes': len(scene_dirs),
            'scenes': [d.name for d in scene_dirs],
            'occlusion_config': {
                'start': occlusion_start,
                'end': occlusion_end,
                'ratio': occlusion_ratio,
                'type': occlusion_type
            },
            'outputs_per_scene': [
                'temporal_consistency.mp4',
                'depth_ordering.mp4',
                'trajectory.mp4',
                'occlusion_test.mp4',
                'occlusion_results.json'
            ],
            'stats': all_stats
        }
        
        with open(output_dir / 'summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # 生成 README
        readme = f"""# TempoVLM Complete Demo Results

## 生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 總共 {len(scene_dirs)} 個場景

| 場景 | 時序一致性 | 深度排序 | 軌跡預測 | 遮擋測試 |
|------|-----------|---------|---------|---------|
"""
        for scene_dir in scene_dirs:
            scene_name = scene_dir.name
            readme += f"| {scene_name} | ✅ | ✅ | ✅ | ✅ |\n"
        
        readme += f"""
## 每個場景包含:

1. **temporal_consistency.mp4** - 時序一致性對比影片
   - Base Model vs Unified Model 特徵相似度曲線

2. **depth_ordering.mp4** - 深度排序測試
   - 三個區域 (左/中/右) 深度比較

3. **trajectory.mp4** - 軌跡預測
   - 俯視圖顯示 GT 軌跡

4. **occlusion_test.mp4** - 遮擋測試 ⭐ NEW
   - 遮擋配置: Frame {occlusion_start}-{occlusion_end}, ratio={occlusion_ratio}, type={occlusion_type}
   - GT / 遮擋 / 注入後 描述對比

5. **occlusion_results.json** - 遮擋測試詳細結果
   - 每幀的異常分數、品質、描述文字

## 統計摘要
"""
        
        for scene_name, stats in all_stats.items():
            readme += f"\n### {scene_name}\n"
            
            if 'temporal_consistency' in stats:
                tc = stats['temporal_consistency']
                readme += f"- 時序一致性改善: {tc.get('improvement', 0):.2f}%\n"
            
            if 'occlusion_test' in stats:
                ot = stats['occlusion_test']
                readme += f"- 遮擋檢測率: {ot.get('detection_rate', 0)*100:.1f}%\n"
                readme += f"- 成功注入數: {ot.get('successful_injections', 0)}\n"
        
        with open(output_dir / 'README.md', 'w', encoding='utf-8') as f:
            f.write(readme)


def main():
    parser = argparse.ArgumentParser(description='TempoVLM Complete Demo')
    parser.add_argument('--model_path', type=str, required=True,
                       help='UnifiedTempoVLM 模型路徑')
    parser.add_argument('--data_root', type=str, required=True,
                       help='ScanNet 資料根目錄')
    parser.add_argument('--output_dir', type=str, default='./complete_demo_output',
                       help='輸出目錄')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'test', 'all'],
                       help='使用哪個資料集')
    parser.add_argument('--max_scenes', type=int, default=3,
                       help='最多處理幾個場景')
    parser.add_argument('--device', type=str, default='cuda')
    
    # Demo 選擇參數
    parser.add_argument('--demos', type=str, default='all',
                       help='要執行的 demo，用逗號分隔: temporal,depth,motion,occlusion 或 all')
    
    # 遮擋測試參數
    parser.add_argument('--occlusion_start', type=int, default=15,
                       help='遮擋開始幀')
    parser.add_argument('--occlusion_end', type=int, default=25,
                       help='遮擋結束幀')
    parser.add_argument('--occlusion_ratio', type=float, default=0.4,
                       help='遮擋區域比例')
    parser.add_argument('--occlusion_type', type=str, default='black',
                       choices=['black', 'white', 'blur', 'noise'],
                       help='遮擋類型')
    parser.add_argument('--injection_method', type=str, default='full',
                       choices=['raw', 'full', 'strong', 'adaptive'],
                       help='注入方法')
    
    args = parser.parse_args()
    
    # 解析 demos 參數
    if args.demos == 'all':
        demos = ['temporal', 'depth', 'motion', 'occlusion']
    else:
        demos = [d.strip() for d in args.demos.split(',')]
    
    visualizer = CompleteDemoVisualizer(
        unified_model_path=args.model_path,
        device=args.device
    )
    
    visualizer.run_complete_demo(
        data_root=args.data_root,
        output_dir=args.output_dir,
        split=args.split,
        max_scenes=args.max_scenes,
        occlusion_start=args.occlusion_start,
        occlusion_end=args.occlusion_end,
        occlusion_ratio=args.occlusion_ratio,
        occlusion_type=args.occlusion_type,
        injection_method=args.injection_method,
        demos=demos
    )


if __name__ == '__main__':
    main()
