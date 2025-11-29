#!/usr/bin/env python3
"""
TempoVLM 視覺化展示腳本
======================

生成三種視覺化展示：
1. 時序穩定性 - Split Screen 遮擋測試影片
2. 深度感知 - Depth Radar 儀表板
3. 運動感知 - Real-time Trajectory Plot

輸出：
- 對比影片
- 儀表板截圖
- 軌跡動畫
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

# Qwen2-VL
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


class TempoVLMVisualizer:
    """TempoVLM 視覺化器"""
    
    def __init__(self, unified_model_path, device='cuda'):
        self.device = device
        
        print("=" * 70)
        print("🎨 TempoVLM Visualizer")
        print("=" * 70)
        
        # 載入模型
        print("\n📦 載入模型...")
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
        print("✅ 模型載入完成！")
    
    def _load_unified_model(self, model_path):
        """載入 Unified Model"""
        from models_unified import UnifiedTempoVLM
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        
        if 'shared_encoder.0.weight' in state_dict:
            hidden_dim = state_dict['shared_encoder.0.weight'].shape[0]
        else:
            hidden_dim = 768
        
        self.unified_model = UnifiedTempoVLM(hidden_dim=hidden_dim).to(self.device)
        
        if 'model_state_dict' in checkpoint:
            self.unified_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.unified_model.load_state_dict(checkpoint)
        
        self.unified_model.eval()
        self.unified_model.half()
    
    def extract_features(self, image):
        """提取特徵"""
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
            features = hidden_states.mean(dim=1)
        
        return features.half()
    
    # ========== 1. 時序穩定性視覺化 ==========
    
    def visualize_temporal_consistency(self, scene_dir, output_path, max_frames=60):
        """
        生成時序一致性對比影片
        - 左側: Base Model 特徵相似度曲線
        - 右側: Unified Model 特徵相似度曲線
        - 底部: 相似度隨時間變化的圖
        """
        print("\n🎬 生成時序一致性對比影片...")
        
        color_dir = scene_dir / 'color'
        frame_files = sorted(color_dir.glob('*.jpg'))[:max_frames]
        
        if len(frame_files) < 10:
            print("❌ 幀數不足")
            return
        
        # 提取特徵
        print("  提取特徵...")
        base_features = []
        for f in tqdm(frame_files, desc="  Base"):
            img = Image.open(f).convert('RGB')
            feat = self.extract_features(img)
            base_features.append(feat)
        
        unified_features = [base_features[0]]
        for i in range(1, len(base_features)):
            with torch.no_grad():
                outputs = self.unified_model(base_features[i], base_features[i-1], tasks=['temporal'])
                unified_features.append(outputs['temporal'])
        
        # 計算相似度
        base_sims = [1.0]  # 第一幀與自己的相似度
        unified_sims = [1.0]
        for i in range(1, len(base_features)):
            base_sim = F.cosine_similarity(base_features[i], base_features[i-1], dim=-1).item()
            unified_sim = F.cosine_similarity(unified_features[i], unified_features[i-1], dim=-1).item()
            base_sims.append(base_sim)
            unified_sims.append(unified_sim)
        
        # 生成影片
        print("  生成影片...")
        
        # 影片參數
        frame_width = 1280
        frame_height = 720
        fps = 10
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))
        
        for i, frame_file in enumerate(tqdm(frame_files, desc="  寫入幀")):
            # 讀取原始幀
            img = cv2.imread(str(frame_file))
            img = cv2.resize(img, (640, 480))
            
            # 創建畫布
            canvas = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            canvas[:] = (30, 30, 30)  # 深灰背景
            
            # 放置原始圖片
            canvas[20:500, 20:660] = img
            
            # 繪製相似度曲線
            fig, ax = plt.subplots(figsize=(6, 3), facecolor='#1e1e1e')
            ax.set_facecolor('#1e1e1e')
            
            # 繪製到當前幀為止的曲線
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
            
            # 保存圖表為圖片
            fig.tight_layout()
            fig.canvas.draw()
            
            # 使用新的 API
            plot_img = np.array(fig.canvas.buffer_rgba())[:, :, :3]  # RGBA -> RGB
            plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR)
            plot_img = cv2.resize(plot_img, (600, 200))
            plt.close(fig)
            
            # 放置圖表
            canvas[510:710, 20:620] = plot_img
            
            # 繪製當前幀的指標
            current_base_sim = base_sims[i]
            current_unified_sim = unified_sims[i]
            
            # 右側面板
            panel_x = 680
            cv2.putText(canvas, f'Frame: {i+1}/{len(frame_files)}', (panel_x, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            cv2.putText(canvas, 'Current Similarity:', (panel_x, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.putText(canvas, f'Base:    {current_base_sim:.4f}', (panel_x, 140),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 2)
            
            cv2.putText(canvas, f'Unified: {current_unified_sim:.4f}', (panel_x, 180),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
            
            # 改善指示
            improve = (current_unified_sim - current_base_sim) / current_base_sim * 100
            color = (100, 255, 100) if improve > 0 else (100, 100, 255)
            cv2.putText(canvas, f'Improvement: {improve:+.2f}%', (panel_x, 230),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # 平均統計
            cv2.putText(canvas, 'Average (so far):', (panel_x, 300),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(canvas, f'Base:    {np.mean(base_sims[:i+1]):.4f}', (panel_x, 340),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 2)
            cv2.putText(canvas, f'Unified: {np.mean(unified_sims[:i+1]):.4f}', (panel_x, 380),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
            
            # 繪製相似度條
            bar_y = 450
            bar_width = 400
            bar_height = 30
            
            # Base Model 條
            cv2.putText(canvas, 'Base', (panel_x, bar_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.rectangle(canvas, (panel_x, bar_y), (panel_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
            base_width = int(bar_width * (current_base_sim - 0.8) / 0.2)  # 0.8-1.0 範圍
            cv2.rectangle(canvas, (panel_x, bar_y), (panel_x + base_width, bar_y + bar_height), (100, 100, 255), -1)
            
            # Unified Model 條
            bar_y2 = bar_y + 50
            cv2.putText(canvas, 'Unified', (panel_x, bar_y2 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.rectangle(canvas, (panel_x, bar_y2), (panel_x + bar_width, bar_y2 + bar_height), (50, 50, 50), -1)
            unified_width = int(bar_width * (current_unified_sim - 0.8) / 0.2)
            cv2.rectangle(canvas, (panel_x, bar_y2), (panel_x + unified_width, bar_y2 + bar_height), (100, 255, 100), -1)
            
            # 標題
            cv2.putText(canvas, 'TempoVLM: Temporal Consistency Demo', (20, frame_height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
            
            out.write(canvas)
        
        out.release()
        print(f"✅ 影片已保存: {output_path}")
    
    # ========== 2. 深度感知視覺化 ==========
    
    def visualize_depth_radar(self, scene_dir, output_path, max_frames=60):
        """
        生成深度排序能力展示影片
        
        正確的測試方式：
        1. 裁切畫面中的不同區域
        2. 對每個區域分別提取 Qwen2-VL 特徵
        3. 用真實的區域特徵做深度排序預測
        """
        print("\n🎬 生成深度排序能力展示...")
        
        color_dir = scene_dir / 'color'
        depth_dir = scene_dir / 'depth'
        
        frame_files = sorted(color_dir.glob('*.jpg'))[:max_frames]
        
        if len(frame_files) < 10:
            print("❌ 幀數不足")
            return
        
        # 影片參數
        frame_width = 1280
        frame_height = 720
        fps = 10
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))
        
        # 統計
        correct_predictions = 0
        total_predictions = 0
        accuracy_history = deque(maxlen=30)
        
        for i, frame_file in enumerate(tqdm(frame_files, desc="  生成幀")):
            # 讀取圖像和深度
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
            
            # 定義三個區域（在原圖上裁切）
            img_w, img_h = img_pil.size
            region_width = img_w // 3
            region_height = img_h // 2  # 取中間一半高度
            y_start = img_h // 4
            
            # 裁切三個區域的圖片
            region_crops = {
                'left': img_pil.crop((0, y_start, region_width, y_start + region_height)),
                'center': img_pil.crop((region_width, y_start, 2*region_width, y_start + region_height)),
                'right': img_pil.crop((2*region_width, y_start, img_w, y_start + region_height)),
            }
            
            # 計算三個區域的 GT 深度
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
            
            # 提取三個區域的特徵
            test_results = []
            
            if self.unified_model is not None:
                with torch.no_grad():
                    # 對每個區域分別提取特徵
                    region_features = {}
                    for name, crop in region_crops.items():
                        # 調整大小確保模型能處理
                        crop_resized = crop.resize((224, 224))
                        feat = self.extract_features(crop_resized)
                        region_features[name] = feat
                    
                    # 編碼特徵
                    left_enc = self.unified_model.shared_encoder(region_features['left'].half())
                    center_enc = self.unified_model.shared_encoder(region_features['center'].half())
                    right_enc = self.unified_model.shared_encoder(region_features['right'].half())
                    
                    # GT 答案
                    gt_order_lc = 0 if gt_depths['left'] < gt_depths['center'] else 1
                    gt_order_cr = 0 if gt_depths['center'] < gt_depths['right'] else 1
                    gt_order_lr = 0 if gt_depths['left'] < gt_depths['right'] else 1
                    
                    # 預測 Left vs Center
                    combined_lc = torch.cat([left_enc, center_enc], dim=-1)
                    logits_lc = self.unified_model.depth_order_head(combined_lc)
                    pred_lc = torch.argmax(logits_lc, dim=-1).item()
                    conf_lc = torch.softmax(logits_lc, dim=-1).max().item()
                    correct_lc = (pred_lc == gt_order_lc)
                    
                    # 預測 Center vs Right
                    combined_cr = torch.cat([center_enc, right_enc], dim=-1)
                    logits_cr = self.unified_model.depth_order_head(combined_cr)
                    pred_cr = torch.argmax(logits_cr, dim=-1).item()
                    conf_cr = torch.softmax(logits_cr, dim=-1).max().item()
                    correct_cr = (pred_cr == gt_order_cr)
                    
                    # 預測 Left vs Right
                    combined_lr = torch.cat([left_enc, right_enc], dim=-1)
                    logits_lr = self.unified_model.depth_order_head(combined_lr)
                    pred_lr = torch.argmax(logits_lr, dim=-1).item()
                    conf_lr = torch.softmax(logits_lr, dim=-1).max().item()
                    correct_lr = (pred_lr == gt_order_lr)
                    
                    test_results = [
                        ('L vs C', gt_order_lc, pred_lc, correct_lc, conf_lc, gt_depths['left'], gt_depths['center']),
                        ('C vs R', gt_order_cr, pred_cr, correct_cr, conf_cr, gt_depths['center'], gt_depths['right']),
                        ('L vs R', gt_order_lr, pred_lr, correct_lr, conf_lr, gt_depths['left'], gt_depths['right']),
                    ]
                    
                    # 更新統計
                    for result in test_results:
                        total_predictions += 1
                        if result[3]:  # correct
                            correct_predictions += 1
                    
                    if total_predictions > 0:
                        accuracy_history.append(correct_predictions / total_predictions)
            
            # 創建畫布
            canvas = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            canvas[:] = (30, 30, 30)
            
            # ========== 左上: 原始圖 + 區域標記 ==========
            img_with_regions = img_resized.copy()
            
            # 畫區域分割線和框
            h_vis, w_vis = img_with_regions.shape[:2]
            y1, y2 = h_vis//4, 3*h_vis//4
            
            # 左區域框 (紅)
            cv2.rectangle(img_with_regions, (0, y1), (w_vis//3, y2), (100, 100, 255), 2)
            cv2.putText(img_with_regions, f'L:{gt_depths["left"]:.1f}m', (5, y1 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 2)
            
            # 中區域框 (綠)
            cv2.rectangle(img_with_regions, (w_vis//3, y1), (2*w_vis//3, y2), (100, 255, 100), 2)
            cv2.putText(img_with_regions, f'C:{gt_depths["center"]:.1f}m', (w_vis//3 + 5, y1 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 2)
            
            # 右區域框 (藍)
            cv2.rectangle(img_with_regions, (2*w_vis//3, y1), (w_vis, y2), (255, 100, 100), 2)
            cv2.putText(img_with_regions, f'R:{gt_depths["right"]:.1f}m', (2*w_vis//3 + 5, y1 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 2)
            
            canvas[20:380, 20:660] = img_with_regions
            cv2.putText(canvas, 'RGB + Region Crops (for feature extraction)', (25, 400),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # ========== 右上: 深度圖視覺化 ==========
            depth_vis = cv2.applyColorMap(
                (np.clip(depth_resized / 5.0, 0, 1) * 255).astype(np.uint8),
                cv2.COLORMAP_JET
            )
            # 也畫上區域框
            cv2.rectangle(depth_vis, (0, h//4), (w//3, 3*h//4), (255, 255, 255), 1)
            cv2.rectangle(depth_vis, (w//3, h//4), (2*w//3, 3*h//4), (255, 255, 255), 1)
            cv2.rectangle(depth_vis, (2*w//3, h//4), (w, 3*h//4), (255, 255, 255), 1)
            
            canvas[20:380, 680:1260] = cv2.resize(depth_vis, (580, 360))
            cv2.putText(canvas, 'GT Depth Map (white boxes = compared regions)', (685, 400),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # ========== 下半部: 深度排序測試結果 ==========
            panel_y = 430
            
            cv2.putText(canvas, 'Depth Ordering Test (Real Region Features)', (20, panel_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(canvas, f'Frame: {i+1}/{len(frame_files)}', (550, panel_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            # 測試結果表格
            table_y = panel_y + 40
            headers = ['Test', 'Depths', 'GT', 'Pred', 'Result', 'Conf']
            col_x = [20, 100, 220, 320, 420, 490]
            
            for j, hdr in enumerate(headers):
                cv2.putText(canvas, hdr, (col_x[j], table_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            
            for j, result in enumerate(test_results):
                test_name, gt, pred, correct, conf, depth_a, depth_b = result
                row_y = table_y + 30 + j * 35
                
                # Test name
                cv2.putText(canvas, test_name, (col_x[0], row_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                # Actual depths
                cv2.putText(canvas, f'{depth_a:.1f}m vs {depth_b:.1f}m', (col_x[1], row_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
                
                # GT (A closer / B closer)
                gt_text = 'A closer' if gt == 0 else 'B closer'
                cv2.putText(canvas, gt_text, (col_x[2], row_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
                
                # Pred
                pred_text = 'A closer' if pred == 0 else 'B closer'
                cv2.putText(canvas, pred_text, (col_x[3], row_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)
                
                # Result
                result_text = 'OK' if correct else 'X'
                result_color = (0, 255, 0) if correct else (0, 0, 255)
                cv2.putText(canvas, result_text, (col_x[4], row_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, result_color, 2)
                
                # Confidence
                cv2.putText(canvas, f'{conf:.2f}', (col_x[5], row_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # ========== 累積準確率 ==========
            stats_x = 600
            cv2.putText(canvas, 'Cumulative Stats:', (stats_x, panel_y + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            if total_predictions > 0:
                acc = correct_predictions / total_predictions
                acc_color = (0, 255, 0) if acc > 0.6 else (0, 165, 255) if acc > 0.4 else (0, 0, 255)
                cv2.putText(canvas, f'Accuracy: {acc:.1%}', (stats_x, panel_y + 75),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, acc_color, 2)
                cv2.putText(canvas, f'({correct_predictions}/{total_predictions})', (stats_x + 150, panel_y + 75),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                # 與隨機基線比較
                random_baseline = 0.5
                improvement = acc - random_baseline
                imp_color = (0, 255, 0) if improvement > 0 else (0, 0, 255)
                cv2.putText(canvas, f'vs Random: {improvement:+.1%}', (stats_x, panel_y + 105),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, imp_color, 1)
            
            # ========== 準確率曲線 ==========
            if len(accuracy_history) > 1:
                graph_x = 600
                graph_y = panel_y + 130
                graph_w = 350
                graph_h = 80
                
                cv2.rectangle(canvas, (graph_x, graph_y), (graph_x + graph_w, graph_y + graph_h),
                             (50, 50, 50), -1)
                
                # 50% 基準線
                baseline_y = graph_y + graph_h // 2
                cv2.line(canvas, (graph_x, baseline_y), (graph_x + graph_w, baseline_y),
                        (100, 100, 100), 1)
                cv2.putText(canvas, '50%', (graph_x - 30, baseline_y + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)
                
                # 繪製曲線
                points = []
                for j, acc in enumerate(accuracy_history):
                    x = graph_x + int(j * graph_w / 30)
                    y = graph_y + graph_h - int(acc * graph_h)
                    points.append((x, y))
                
                if len(points) > 1:
                    for j in range(len(points) - 1):
                        color = (0, 255, 0) if list(accuracy_history)[j] > 0.5 else (0, 0, 255)
                        cv2.line(canvas, points[j], points[j+1], color, 2)
                
                cv2.putText(canvas, 'Accuracy History (green=better than random)', (graph_x, graph_y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            # ========== 說明文字 ==========
            note_y = 680
            cv2.putText(canvas, 'Method: Crop L/C/R regions -> Extract Qwen2-VL features -> depth_order_head predicts which is closer',
                       (20, note_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
            
            # 標題
            cv2.putText(canvas, f'TempoVLM: Depth Ordering Demo - Frame {i+1}/{len(frame_files)}',
                       (20, frame_height - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
            
            out.write(canvas)
        
        out.release()
        
        # 輸出統計
        if total_predictions > 0:
            final_acc = correct_predictions / total_predictions
            print(f"  📊 最終深度排序準確率: {final_acc:.1%} ({correct_predictions}/{total_predictions})")
        
        print(f"✅ 影片已保存: {output_path}")
    
    # ========== 2.5 深度回歸視覺化 (需要 depth_regression 訓練) ==========
    
    def visualize_depth_regression(self, scene_dir, output_path, max_frames=60):
        """
        生成深度回歸預測 vs Ground Truth 的比較影片
        
        顯示內容:
        1. 原始 RGB 圖像
        2. GT 深度圖 (熱力圖)
        3. 三個區域 (左/中/右) 的預測深度 vs GT 深度
        4. 預測誤差曲線
        """
        print("\n🎬 生成深度回歸比較影片...")
        
        # 檢查模型是否支援 depth_regression
        if not hasattr(self.unified_model, 'depth_regression_head'):
            print("❌ 模型未包含 depth_regression_head，需要用 depth_regression 任務訓練")
            return
        
        color_dir = scene_dir / 'color'
        depth_dir = scene_dir / 'depth'
        
        frame_files = sorted(color_dir.glob('*.jpg'))[:max_frames]
        
        if len(frame_files) < 10:
            print("❌ 幀數不足")
            return
        
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
            with torch.no_grad():
                for name, crop in regions.items():
                    crop_resized = crop.resize((224, 224))
                    feat = self.extract_features(crop_resized)
                    feat_enc = self.unified_model.shared_encoder(feat.half())
                    
                    # 深度回歸預測 (輸出 0-1，需要反正規化)
                    pred_norm = self.unified_model.depth_regression_head(feat_enc)
                    pred_norm = pred_norm.squeeze().item()
                    
                    # 反正規化: normalized = (depth - 0.5) / 4.5
                    # depth = normalized * 4.5 + 0.5
                    pred_depth = pred_norm * 4.5 + 0.5
                    pred_depth = max(0.5, min(5.0, pred_depth))  # Clamp to valid range
                    
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
                for errors, color, name in [
                    (errors_left, (100, 100, 255), 'L'),
                    (errors_center, (100, 255, 100), 'C'),
                    (errors_right, (255, 100, 100), 'R'),
                ]:
                    points = []
                    for k, err in enumerate(errors[-50:]):  # 最近 50 幀
                        x = graph_x + int(k * graph_w / 50)
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
    
    # ========== 3. 運動感知視覺化 ==========
    
    def visualize_trajectory(self, scene_dir, output_path, max_frames=100):
        """
        生成即時軌跡影片
        - 主畫面: 原始影像
        - 右下角: 俯視軌跡圖
        """
        print("\n🎬 生成軌跡影片...")
        
        color_dir = scene_dir / 'color'
        pose_dir = scene_dir / 'pose'
        
        frame_files = sorted(color_dir.glob('*.jpg'))[:max_frames]
        
        if len(frame_files) < 10:
            print("❌ 幀數不足")
            return
        
        # 載入 GT 軌跡
        gt_positions = []
        for frame_file in frame_files:
            pose_file = pose_dir / (frame_file.stem + '.txt')
            if pose_file.exists():
                try:
                    pose = np.loadtxt(pose_file).reshape(4, 4)
                    gt_positions.append(pose[:3, 3])
                except:
                    gt_positions.append(gt_positions[-1] if gt_positions else np.zeros(3))
            else:
                gt_positions.append(gt_positions[-1] if gt_positions else np.zeros(3))
        
        gt_positions = np.array(gt_positions)
        
        # 預測軌跡
        print("  預測運動...")
        pred_positions = [np.zeros(3)]
        prev_feat = None
        
        for i, frame_file in enumerate(tqdm(frame_files, desc="  提取特徵")):
            img = Image.open(frame_file).convert('RGB')
            feat = self.extract_features(img)
            
            if prev_feat is not None:
                with torch.no_grad():
                    outputs = self.unified_model(feat, prev_feat, tasks=['motion'])
                    pred_motion = outputs['motion'].cpu().numpy()[0]
                    pred_positions.append(pred_positions[-1] + pred_motion[:3])
            
            prev_feat = feat
        
        pred_positions = np.array(pred_positions)
        
        # 對齊預測軌跡
        pred_centered = pred_positions - pred_positions.mean(axis=0)
        gt_centered = gt_positions - gt_positions.mean(axis=0)
        
        pred_scale = np.linalg.norm(pred_centered)
        gt_scale = np.linalg.norm(gt_centered)
        
        if pred_scale > 1e-6:
            pred_scaled = pred_centered * (gt_scale / pred_scale)
        else:
            pred_scaled = pred_centered
        
        # 生成影片
        frame_width = 1280
        frame_height = 720
        fps = 10
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))
        
        for i, frame_file in enumerate(tqdm(frame_files, desc="  寫入幀")):
            img = cv2.imread(str(frame_file))
            img = cv2.resize(img, (800, 600))
            
            # 創建畫布
            canvas = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            canvas[:] = (30, 30, 30)
            
            # 放置主圖
            canvas[20:620, 20:820] = img
            
            # 繪製俯視圖
            traj_size = 400
            traj_x = 850
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
            all_pos = np.vstack([gt_centered[:i+1], pred_scaled[:i+1]])
            if len(all_pos) > 0:
                x_range = max(abs(all_pos[:, 0].max()), abs(all_pos[:, 0].min()), 1.0)
                z_range = max(abs(all_pos[:, 2].max()), abs(all_pos[:, 2].min()), 1.0)
                scale = min(traj_size / (2.2 * x_range), traj_size / (2.2 * z_range))
            else:
                scale = 50
            
            center_x = traj_x + traj_size // 2
            center_y = traj_y + traj_size // 2
            
            # 繪製 GT 軌跡
            gt_points = []
            for j in range(i + 1):
                px = int(center_x + gt_centered[j, 0] * scale)
                py = int(center_y - gt_centered[j, 2] * scale)  # z 軸向上
                gt_points.append((px, py))
            
            if len(gt_points) > 1:
                for j in range(len(gt_points) - 1):
                    cv2.line(canvas, gt_points[j], gt_points[j+1], (100, 255, 100), 2)
            
            # 繪製預測軌跡
            if i < len(pred_scaled):
                pred_points = []
                for j in range(min(i + 1, len(pred_scaled))):
                    px = int(center_x + pred_scaled[j, 0] * scale)
                    py = int(center_y - pred_scaled[j, 2] * scale)
                    pred_points.append((px, py))
                
                if len(pred_points) > 1:
                    for j in range(len(pred_points) - 1):
                        cv2.line(canvas, pred_points[j], pred_points[j+1], (255, 100, 100), 2)
            
            # 當前位置標記
            if gt_points:
                cv2.circle(canvas, gt_points[-1], 8, (100, 255, 100), -1)
            if i < len(pred_scaled) and pred_points:
                cv2.circle(canvas, pred_points[-1], 8, (255, 100, 100), -1)
            
            # 起點標記
            if gt_points:
                cv2.circle(canvas, gt_points[0], 5, (255, 255, 255), -1)
                cv2.putText(canvas, 'Start', (gt_points[0][0] + 10, gt_points[0][1]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # 圖例
            cv2.putText(canvas, 'Top-down Trajectory View', (traj_x, traj_y + traj_size + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            legend_y = traj_y + traj_size + 60
            cv2.line(canvas, (traj_x, legend_y), (traj_x + 30, legend_y), (100, 255, 100), 2)
            cv2.putText(canvas, 'GT Trajectory', (traj_x + 40, legend_y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
            
            cv2.line(canvas, (traj_x, legend_y + 25), (traj_x + 30, legend_y + 25), (255, 100, 100), 2)
            cv2.putText(canvas, 'Predicted', (traj_x + 40, legend_y + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 1)
            
            # 誤差統計
            if i > 0 and i < len(pred_scaled):
                error = np.sqrt(((gt_centered[:i+1] - pred_scaled[:i+1]) ** 2).sum(axis=1).mean())
                cv2.putText(canvas, f'ATE: {error:.3f}m', (traj_x, legend_y + 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # 標題
            cv2.putText(canvas, f'TempoVLM: Motion Prediction Demo - Frame {i+1}/{len(frame_files)}',
                       (20, frame_height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
            
            out.write(canvas)
        
        out.release()
        print(f"✅ 影片已保存: {output_path}")
    
    # ========== 主函數 ==========
    
    def run_all_visualizations(self, data_root, output_dir, split='test', max_scenes=5):
        """執行所有視覺化"""
        data_root = Path(data_root)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 根據 split 選擇資料夾
        if split == 'test':
            search_dirs = ['scannet_frames_test']
            print("\n🧪 使用測試集 (scannet_frames_test)")
        elif split == 'train':
            search_dirs = ['scannet_frames_25k']
            print("\n📚 使用訓練集 (scannet_frames_25k)")
        else:
            search_dirs = ['scannet_frames_test', 'scannet_frames_25k', '']
        
        # 找場景
        scene_dirs = []
        for subdir in search_dirs:
            subpath = data_root / subdir if subdir else data_root
            if subpath.exists():
                scenes = [d for d in subpath.iterdir() if d.is_dir() and d.name.startswith('scene')]
                scene_dirs.extend(scenes)
        
        if not scene_dirs:
            print("❌ 找不到場景")
            return
        
        # 選擇多個場景
        scene_dirs = sorted(scene_dirs)[:max_scenes]
        print(f"\n📊 將處理 {len(scene_dirs)} 個場景")
        
        # 為每個場景生成視覺化
        for idx, scene_dir in enumerate(scene_dirs):
            print(f"\n{'='*60}")
            print(f"🎬 場景 {idx+1}/{len(scene_dirs)}: {scene_dir.name}")
            print(f"{'='*60}")
            
            # 創建場景專屬輸出目錄
            scene_output_dir = output_dir / scene_dir.name
            scene_output_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                # 1. 時序一致性
                self.visualize_temporal_consistency(
                    scene_dir, 
                    scene_output_dir / 'temporal_consistency.mp4'
                )
                
                # 2. 深度排序 (depth_order)
                self.visualize_depth_radar(
                    scene_dir,
                    scene_output_dir / 'depth_ordering.mp4'
                )
                
                # 3. 深度回歸 (depth_regression) - 需要有訓練過的模型
                if hasattr(self.unified_model, 'depth_regression_head'):
                    self.visualize_depth_regression(
                        scene_dir,
                        scene_output_dir / 'depth_regression.mp4'
                    )
                
                # 4. 軌跡
                self.visualize_trajectory(
                    scene_dir,
                    scene_output_dir / 'trajectory.mp4'
                )
                
                print(f"✅ {scene_dir.name} 完成")
                
            except Exception as e:
                print(f"❌ {scene_dir.name} 失敗: {e}")
                continue
        
        # 生成總結
        self._generate_summary(output_dir, scene_dirs)
        
        print(f"\n🎉 所有視覺化完成！輸出目錄: {output_dir}")
    
    def _generate_summary(self, output_dir, scene_dirs):
        """生成視覺化總結"""
        has_depth_regression = hasattr(self.unified_model, 'depth_regression_head')
        
        outputs = [
            'temporal_consistency.mp4',
            'depth_ordering.mp4',
            'trajectory.mp4'
        ]
        if has_depth_regression:
            outputs.insert(2, 'depth_regression.mp4')
        
        summary = {
            'total_scenes': len(scene_dirs),
            'scenes': [d.name for d in scene_dirs],
            'has_depth_regression': has_depth_regression,
            'outputs_per_scene': outputs
        }
        
        import json
        with open(output_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        # 生成 README
        depth_reg_col = "| 深度回歸 " if has_depth_regression else ""
        depth_reg_header = "|---------|" if has_depth_regression else ""
        
        readme = f"""# TempoVLM Visualization Results

## 總共 {len(scene_dirs)} 個場景

| 場景 | 時序一致性 | 深度排序 {depth_reg_col}| 軌跡預測 |
|------|-----------|---------|{depth_reg_header}---------|
"""
        for scene_dir in scene_dirs:
            scene_name = scene_dir.name
            depth_reg_check = "| ✅ " if has_depth_regression else ""
            readme += f"| {scene_name} | ✅ | ✅ {depth_reg_check}| ✅ |\n"
        
        readme += f"""
## 每個場景包含:

1. **temporal_consistency.mp4** - 時序一致性對比影片
   - 左側: 原始影像
   - 右側: Base vs Unified 相似度曲線

2. **depth_ordering.mp4** - 深度排序測試
   - 預測三個區域 (左/中/右) 哪個更近
   - 與 Ground Truth 深度圖比較
"""
        
        if has_depth_regression:
            readme += """
3. **depth_regression.mp4** - 深度數值預測 ⭐ NEW
   - 預測每個區域的絕對深度值 (米)
   - 柱狀圖比較: GT vs Prediction
   - 即時顯示預測誤差曲線
"""
            readme += """
4. **trajectory.mp4** - 軌跡預測
"""
        else:
            readme += """
3. **trajectory.mp4** - 軌跡預測
"""
        
        readme += """   - 俯視圖顯示 GT 軌跡 vs 預測軌跡
   - ATE 誤差即時更新
"""
        
        with open(output_dir / 'README.md', 'w') as f:
            f.write(readme)


def main():
    parser = argparse.ArgumentParser(description='TempoVLM Visualization')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./viz_demo')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'test', 'all'],
                        help='使用哪個資料集: train=scannet_frames_25k, test=scannet_frames_test')
    parser.add_argument('--max_scenes', type=int, default=5,
                        help='最多處理幾個場景')
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    visualizer = TempoVLMVisualizer(
        unified_model_path=args.model_path,
        device=args.device
    )
    
    visualizer.run_all_visualizations(
        data_root=args.data_root,
        output_dir=args.output_dir,
        split=args.split,
        max_scenes=args.max_scenes
    )


if __name__ == '__main__':
    main()
