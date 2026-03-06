#!/usr/bin/env python3
"""
Complete visualization and evaluation script for TempoDepth-VLM.

Features:
1. Multi-dataset support: ScanNet and NYU Depth V2
2. Scientific validation for depth and trajectory tasks
3. Occlusion recovery with optional YOLO-based masking and memory injection
4. Aggregated report export to final_report.json

Example:
python complete_demo.py --model_path checkpoints/model.pt --data_root data/scannet --dataset scannet --demos depth,motion
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
import cv2
import argparse
from collections import deque
from datetime import datetime
import random

# Qwen2-VL
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from utils.memory_utils import AdaptiveMemoryBuffer

# Optional YOLO-based occlusion
try:
    from utils.yolo_occlusion import YOLOOccluder
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    YOLOOccluder = None
    print("YOLO is not installed; object occlusion is unavailable")


class CompleteDemoVisualizer:
    """Complete demo visualizer"""
    
    def __init__(self, unified_model_path, device='cuda'):
        self.device = device
        self.checkpoint_path = unified_model_path
        
        print("=" * 70)
        print("TempoVLM Complete Demo Visualizer (Scientific Validation)")
        print("=" * 70)
        print(f"\n Using checkpoint: {unified_model_path}")
        
        # Load models
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
        
        # Temporal buffer
        self.temporal_buffer = deque(maxlen=5)
        
        # Feature projector for injection dimension matching
        self.feature_projector = None
        
        print("model loaded.\n")
    
    def _load_unified_model(self, model_path):
        from models_unified import UnifiedTempoVLM
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        
        # Auto-detect model dimensions
        if 'shared_encoder.0.weight' in state_dict:
            hidden_dim = state_dict['shared_encoder.0.weight'].shape[0]
        else:
            hidden_dim = 768
        
        # Detect whether GRU is enabled
        use_gru = 'temporal_gru.weight_ih' in state_dict or 'temporal_gru.weight_hh' in state_dict
        
        self.unified_model = UnifiedTempoVLM(
            hidden_dim=hidden_dim,
            use_gru_memory=use_gru
        ).to(self.device)
        
        if 'model_state_dict' in checkpoint:
            state_dict_to_load = checkpoint['model_state_dict']
        else:
            state_dict_to_load = checkpoint
        
        # Backward compatibility for legacy memory_quality_gate checkpoints
        if 'memory_quality_gate.2.weight' in state_dict_to_load and \
           'memory_quality_gate.3.weight' not in state_dict_to_load:
            state_dict_to_load['memory_quality_gate.3.weight'] = state_dict_to_load.pop('memory_quality_gate.2.weight')
            state_dict_to_load['memory_quality_gate.3.bias'] = state_dict_to_load.pop('memory_quality_gate.2.bias')
        
        self.unified_model.load_state_dict(state_dict_to_load, strict=False)
        
        self.unified_model.eval()
        self.unified_model.float()
        self.hidden_dim = hidden_dim
        self.use_gru = use_gru
        
        self.gru_hidden_state = None
    
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

    def extract_edge_features(self, image):
        """
        Extract edge-focused features for scene change detection
        """
        import numpy as np
        from PIL import Image as PILImage
        
        if isinstance(image, PILImage.Image):
            img_array = np.array(image).copy()
        else:
            img_array = image.copy()
            
        h, w = img_array.shape[:2]
        margin_h = int(h * 0.2)
        margin_w = int(w * 0.2)
        img_array[margin_h:h-margin_h, margin_w:w-margin_w] = 0
        
        masked_image = PILImage.fromarray(img_array)
        return self.extract_features(masked_image, use_adapter=False)
    
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
    
    def detect_occlusion_regions(self, image):
        """
         Automatically detect occlusion regions in an image
        """
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
        
        if img_array.shape[-1] == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        h, w = gray.shape
        black_mask = (gray < 10).astype(np.uint8)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        low_texture_mask = (np.abs(laplacian) < 5).astype(np.uint8)
        combined_mask = np.logical_and(black_mask, low_texture_mask).astype(np.uint8)
        kernel = np.ones((5, 5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(combined_mask, connectivity=8)
        
        filtered_mask = np.zeros_like(combined_mask)
        min_area = (h * w) * 0.01
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area > min_area:
                filtered_mask[labels == i] = 1
        
        return filtered_mask
    
    def generate_with_injection(self, image, memory_feat, prompt, injection_strength=0.5, injection_method='full', 
                               occlusion_info=None, max_new_tokens=400):
        """
         Direct feature injection hook
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
        enhanced_feat_copy = memory_feat.clone().detach()
        vision_hidden_size = self.base_model.visual.config.hidden_size if hasattr(self.base_model.visual, 'config') else 1536
        enhanced_dim = enhanced_feat_copy.shape[-1]
        
        if enhanced_dim != vision_hidden_size:
            if not hasattr(self, 'feature_projector') or self.feature_projector is None:
                self.feature_projector = torch.nn.Linear(enhanced_dim, vision_hidden_size)
                torch.nn.init.eye_(self.feature_projector.weight[:min(enhanced_dim, vision_hidden_size), :min(enhanced_dim, vision_hidden_size)])
                torch.nn.init.zeros_(self.feature_projector.bias)
                self.feature_projector = self.feature_projector.to(self.device).half()
        occlusion_mask_2d = None
        if occlusion_info and 'objects' in occlusion_info:
            img_array = np.array(image) if isinstance(image, Image.Image) else image
            h, w = img_array.shape[:2]
            occlusion_mask_2d = np.zeros((h, w), dtype=np.float32)
            
            for obj in occlusion_info['objects']:
                x1, y1, x2, y2 = obj['bbox']
                margin = 20
                x1 = max(0, x1 - margin)
                y1 = max(0, y1 - margin)
                x2 = min(w, x2 + margin)
                y2 = min(h, y2 + margin)
                occlusion_mask_2d[y1:y2, x1:x2] = 1.0
        else:
            occlusion_mask_2d = self.detect_occlusion_regions(image).astype(np.float32)
        occlusion_mask_tensor = torch.from_numpy(occlusion_mask_2d).to(self.device)
        
        def create_injection_hook(method, strength, occl_mask):
            def injection_hook(module, input, output):
                nonlocal enhanced_feat_copy
                
                with torch.no_grad():
                    if enhanced_dim != vision_hidden_size:
                        projected = self.feature_projector(enhanced_feat_copy.float()).half()
                    else:
                        projected = enhanced_feat_copy
                    if output.dim() == 2:
                        num_patches = output.shape[0]
                        projected_expanded = projected.squeeze(0).unsqueeze(0).expand(num_patches, -1)
                        batch = 1
                    elif output.dim() == 3:
                        batch, num_patches, _ = output.shape
                        projected_expanded = projected.unsqueeze(1).expand(batch, num_patches, -1)
                    else:
                        return output
                    orig_mean = output.mean()
                    orig_std = output.std() + 1e-6
                    proj_mean = projected_expanded.mean()
                    proj_std = projected_expanded.std() + 1e-6
                    blended_mean = 0.4 * proj_mean + 0.6 * orig_mean
                    blended_std = torch.max(0.8 * proj_std + 0.2 * orig_std, 0.6 * proj_std)
                    projected_normalized = (projected_expanded - proj_mean) / proj_std * blended_std + blended_mean
                    if num_patches > 4:
                        side = int(num_patches ** 0.5)
                        if side * side == num_patches and occl_mask is not None:
                            h, w = occl_mask.shape
                            patch_h = h // side
                            patch_w = w // side
                            injection_mask = torch.zeros((1, num_patches, 1), device=output.device)
                            for i in range(side):
                                for j in range(side):
                                    patch_idx = i * side + j
                                    y_start = i * patch_h
                                    y_end = (i + 1) * patch_h if i < side - 1 else h
                                    x_start = j * patch_w
                                    x_end = (j + 1) * patch_w if j < side - 1 else w
                                    patch_region = occl_mask[y_start:y_end, x_start:x_end]
                                    occlusion_ratio = patch_region.mean().item()
                                    injection_mask[0, patch_idx, 0] = occlusion_ratio
                            if side >= 3:
                                mask_2d = injection_mask.view(1, side, side, 1)
                                kernel_size = 3
                                padding = kernel_size // 2
                                mask_2d_padded = F.pad(mask_2d.permute(0, 3, 1, 2), 
                                                       (padding, padding, padding, padding), 
                                                       mode='replicate')
                                smoothed = F.avg_pool2d(mask_2d_padded, kernel_size, stride=1, padding=0)
                                injection_mask = smoothed.permute(0, 2, 3, 1).reshape(1, num_patches, 1)
                            injection_mask = torch.clamp(injection_mask * 2.5, max=1.0)
                            high_occlusion = (injection_mask > 0.5).float()
                            injection_mask = injection_mask + high_occlusion * 0.2
                            injection_mask = torch.clamp(injection_mask, max=1.0)
                        else:
                            idxs = torch.arange(num_patches, device=output.device).view(1, num_patches, 1)
                            rows = (idxs // side).float()
                            cols = (idxs % side).float()
                            center = (side - 1) / 2
                            dist = torch.maximum((rows - center).abs(), (cols - center).abs()) / (side / 2)
                            injection_mask = (dist < 0.8).float()
                            injection_mask = torch.clamp(injection_mask * 1.5, max=1.0)
                    else:
                        injection_mask = torch.ones((1, num_patches, 1), device=output.device)
                    
                    # ============================================================
                    # ============================================================
                    
                    if method == 'full':
                        mix = strength * injection_mask
                        if output.dim() == 3:
                            modified = output + mix * (projected_normalized - output)
                        else:
                            modified = output + mix.squeeze(0) * (projected_normalized - output)
                    
                    elif method == 'strong':
                        modified = output.clone()
                        if output.dim() == 3:
                            batch_size, num_patches, _ = output.shape
                            side = int(num_patches ** 0.5)
                            if side * side == num_patches:
                                for row in range(side):
                                    for col in range(side):
                                        idx = row * side + col
                                        local_strength = strength * injection_mask[:, idx, :].squeeze(-1)
                                        modified[:, idx] = (1 - local_strength) * output[:, idx] + local_strength * projected_normalized[:, idx]
                            else:
                                mix = strength * injection_mask
                                modified = output + mix * (projected_normalized - output)
                        else:
                            mix = strength * injection_mask
                            modified = output + mix.squeeze(0) * (projected_normalized - output)
                    
                    elif method == 'adaptive':
                        if output.dim() == 3:
                            diff = torch.abs(output - projected_normalized).mean(dim=-1, keepdim=True)
                            diff_normalized = diff / (diff.max() + 1e-6)
                            adaptive_strength = strength * (0.3 + 0.7 * diff_normalized) * injection_mask
                            occlusion_boost = injection_mask * 0.2
                            adaptive_strength = torch.clamp(adaptive_strength + occlusion_boost, max=1.0)
                            modified = (1 - adaptive_strength) * output + adaptive_strength * projected_normalized
                        else:
                            diff = torch.abs(output - projected_normalized).mean(dim=-1, keepdim=True)
                            diff_normalized = diff / (diff.max() + 1e-6)
                            adaptive_strength = strength * (0.3 + 0.7 * diff_normalized) * injection_mask.squeeze(0)
                            occlusion_boost = injection_mask.squeeze(0) * 0.2
                            adaptive_strength = torch.clamp(adaptive_strength + occlusion_boost, max=1.0)
                            modified = (1 - adaptive_strength) * output + adaptive_strength * projected_normalized
                    
                    else:  # 'raw' or fallback
                        mix = strength * injection_mask
                        if output.dim() == 3:
                            modified = output + mix * (projected_normalized - output)
                        else:
                            modified = output + mix.squeeze(0) * (projected_normalized - output)
                    modified = torch.clamp(modified, orig_mean - 4*orig_std, orig_mean + 4*orig_std)
                    return modified
            
            return injection_hook
        
        hook_handle = self.base_model.visual.register_forward_hook(
            create_injection_hook(injection_method, injection_strength, occlusion_mask_tensor)
        )
        
        try:
            with torch.no_grad():
                generated = self.base_model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False
                )
        finally:
            hook_handle.remove()
        
        full_response = self.processor.decode(generated[0], skip_special_tokens=True)
        response = full_response
        separators = ['assistant\n', 'assistant:', 'Assistant:', 'ASSISTANT:', '<|assistant|>']
        for sep in separators:
            if sep.lower() in response.lower():
                idx = response.lower().find(sep.lower())
                response = response[idx + len(sep):].strip()
                break
        
        return response
    
    def generate_with_injection_two_stage(self, image, memory_feat, injection_strength=0.5, 
                                     injection_method='full', occlusion_info=None,
                                     stage1_max_new_tokens=400, stage2_max_new_tokens=400):
        """
        Two-stage memory-guided generation
        """
        if occlusion_info and 'objects' in occlusion_info:
            occluded_classes = [obj['class_name'] for obj in occlusion_info['objects']]
            occluded_classes_str = ', '.join(set(occluded_classes))
            
            stage1_prompt = (
                f"There are black rectangular masks in this image hiding some objects. "
                f"Answer the following questions step by step:\n"
                f"Q1: What objects are hidden under the black masks?\n"
                f"Q2: Based on your memory from previous frames, what was in those locations?\n"
                f"Q3: Given the context (desk/table scene), what specific items like '{occluded_classes_str}' are being occluded?\n"
                f"Please answer each question clearly and concisely."
            )
        else:
            stage1_prompt = (
                "There are black rectangular regions hiding some objects in this image. "
                "Answer the following questions:\n"
                "Q1: What objects are being hidden under the black masks?\n"
                "Q2: Based on your visual memory from previous frames, what should be in those locations?\n"
                "Please identify the specific occluded objects (e.g., laptop, keyboard, mouse, book)."
            )
        stage1_response = self.generate_with_injection(
            image, memory_feat, stage1_prompt, injection_strength, injection_method, occlusion_info,
            max_new_tokens=stage1_max_new_tokens
        )
        stage2_prompt = (
            f"Based on your analysis that identified the following occluded objects: {stage1_response[:200]}...\n\n"
            f"Now, provide a complete and natural description of the entire scene. "
            f"Describe the room, furniture, and all objects present (including those that are currently occluded). "
            f"Write as if you can see the complete scene without any black masks."
        )
        stage2_response = self.generate_with_injection(
            image, memory_feat, stage2_prompt, injection_strength, injection_method, occlusion_info,
            max_new_tokens=stage2_max_new_tokens
        )
        
        combined_response = (
            f"[Analysis] {stage1_response}\n\n"
            f"[Complete Scene Description] {stage2_response}"
        )
        
        return {
            'stage1_analysis': stage1_response,
            'stage2_description': stage2_response,
            'combined': combined_response
        }

    def clear_temporal_buffer(self):
        self.temporal_buffer.clear()
        if hasattr(self, 'gru_hidden_state'):
            self.gru_hidden_state = None

    # =========================================================================
    # 1. Temporal consistency visualization
    # =========================================================================
    def visualize_temporal_consistency(self, scene_dir, output_path, max_frames=60):
        print("\nCreating Temporal Consistency Video...")
        
        color_dir = scene_dir / 'color'
        frame_files = sorted(list(color_dir.glob('*.jpg')) + list(color_dir.glob('*.png')))[:max_frames]
        
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
        
        # Render video
        print("  Render video...")
        
        frame_width = 1280
        frame_height = 720
        fps = 10
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))
        
        for i, frame_file in enumerate(tqdm(frame_files, desc="  Writing frames")):
            img = cv2.imread(str(frame_file))
            img = cv2.resize(img, (640, 480))
            
            canvas = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            canvas[:] = (30, 30, 30)
            
            canvas[20:500, 20:660] = img
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
        print(f" Video saved: {output_path}")
        
        return {
            'base_mean_sim': float(np.mean(base_sims)),
            'unified_mean_sim': float(np.mean(unified_sims)),
            'improvement': float((np.mean(unified_sims) - np.mean(base_sims)) / np.mean(base_sims) * 100)
        }

    # =========================================================================
    # 2.5 Depth regression visualization
    # =========================================================================
    def visualize_depth_regression(self, scene_dir, output_path, max_frames=60, calibration_frames=10):
        print(f"\n Processing Depth Regression (Calib: {calibration_frames} frames)...")
        
        color_dir = scene_dir / 'color'
        depth_dir = scene_dir / 'depth'
        
        frame_files = sorted(list(color_dir.glob('*.jpg')) + list(color_dir.glob('*.png')))[:max_frames]
        if len(frame_files) == 0: return None
        
        frame_width, frame_height = 1280, 720
        out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), 5, (frame_width, frame_height))
        
        stats = {'abs_diffs': [], 'sq_diffs': [], 'rels': [], 'ratios': [], 'region_rels': {'left':[], 'center':[], 'right':[]}}
        calibration_data = {'pred_medians': [], 'gt_medians': []}
        locked_scale = 1.0
        is_calibrated = False

        for i, frame_file in enumerate(tqdm(frame_files, desc="  Frames")):
            img_pil = Image.open(frame_file).convert('RGB')
            img_cv = cv2.imread(str(frame_file))
            img_resized = cv2.resize(img_cv, (640, 360))
            depth_file = depth_dir / (frame_file.stem + '.png')
            if not depth_file.exists(): depth_file = depth_dir / (frame_file.stem + '.jpg')
            
            if depth_file.exists():
                depth = cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
            else:
                depth = np.ones((480, 640)) * 3.0
            depth_resized = cv2.resize(depth, (640, 360))
            
            # GT
            h, w = depth_resized.shape
            depth_regions = {
                'left': depth_resized[h//4:3*h//4, :w//3],
                'center': depth_resized[h//4:3*h//4, w//3:2*w//3],
                'right': depth_resized[h//4:3*h//4, 2*w//3:],
            }
            gt_depths = {k: v[(v > 0.1) & (v < 10)].mean() if len(v[(v > 0.1) & (v < 10)]) > 0 else 3.0 
                        for k, v in depth_regions.items()}

            # Inference
            with torch.no_grad():
                feat = self.extract_features(img_pil.resize((224, 224)), use_adapter=False)
                outputs, _ = self.unified_model(feat.float(), tasks=['depth_regression'])
                pred_depth_raw = outputs['depth_regression'].squeeze()
            curr_pred_med = float(np.median(pred_depth_raw.detach().cpu().numpy()))
            curr_gt_med = float(np.median(list(gt_depths.values())))
            
            if calibration_frames > 0:
                if i < calibration_frames:
                    calibration_data['pred_medians'].append(curr_pred_med)
                    calibration_data['gt_medians'].append(curr_gt_med)
                    current_scale = curr_gt_med / max(curr_pred_med, 1e-3)
                    status_text = f"Calibrating... ({i+1}/{calibration_frames})"
                    status_color = (0, 255, 255)
                else:
                    if not is_calibrated:
                        if len(calibration_data['pred_medians']) > 0:
                            avg_pred = np.median(calibration_data['pred_medians'])
                            avg_gt = np.median(calibration_data['gt_medians'])
                            locked_scale = avg_gt / max(avg_pred, 1e-3)
                        else:
                            locked_scale = 1.0
                        is_calibrated = True
                    
                    current_scale = locked_scale
                    status_text = "Testing (Scale Locked)"
                    status_color = (0, 255, 0)
            else:
                current_scale = curr_gt_med / max(curr_pred_med, 1e-3)
                status_text = "Testing (Per-Frame Scaling)"
                status_color = (0, 255, 0)
            pred_depths = {
                'left': max(0.1, min(10.0, pred_depth_raw[0].item() * current_scale)),
                'center': max(0.1, min(10.0, pred_depth_raw[1].item() * current_scale)),
                'right': max(0.1, min(10.0, pred_depth_raw[2].item() * current_scale))
            }
            
            # Metrics
            for name in ['left', 'center', 'right']:
                p, g = pred_depths[name], gt_depths[name]
                abs_diff = abs(p - g)
                stats['abs_diffs'].append(abs_diff)
                stats['sq_diffs'].append((p - g) ** 2)
                stats['rels'].append(abs_diff / max(g, 0.1))
                ratio = max(p / max(g, 0.1), g / max(p, 0.1))
                stats['ratios'].append(ratio)
                stats['region_rels'][name].append(abs_diff / max(g, 0.1))

            # Visualization
            canvas = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            canvas[:] = (30, 30, 30)
            canvas[20:380, 20:660] = img_resized
            depth_vis = cv2.applyColorMap((np.clip(depth_resized/5.0, 0, 1)*255).astype(np.uint8), cv2.COLORMAP_JET)
            canvas[20:380, 680:1260] = cv2.resize(depth_vis, (580, 360))
            cv2.putText(canvas, status_text, (20, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            for j, name in enumerate(['left', 'center', 'right']):
                x, y = 100 + j * 150, 600
                h_gt = int(min(gt_depths[name], 5.0) / 5.0 * 100)
                h_pred = int(min(pred_depths[name], 5.0) / 5.0 * 100)
                cv2.rectangle(canvas, (x, y-h_gt), (x+40, y), (255, 150, 50), -1)
                cv2.rectangle(canvas, (x+50, y-h_pred), (x+90, y), (50, 255, 50), -1)
                cv2.putText(canvas, f"{name}: {abs(gt_depths[name]-pred_depths[name]):.2f}m", (x, y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
            out.write(canvas)
        out.release()
        
        if stats['rels']:
            metrics = {
                'absrel': float(np.mean(stats['rels'])),
                'mae': float(np.mean(stats['abs_diffs'])),
                'rmse': float(np.sqrt(np.mean(stats['sq_diffs']))),
                'delta1': float(np.mean(np.array(stats['ratios']) < 1.25) * 100),
                'delta2': float(np.mean(np.array(stats['ratios']) < 1.25**2) * 100),
                'delta3': float(np.mean(np.array(stats['ratios']) < 1.25**3) * 100)
            }
            print("  [Depth Metrics]")
            print(f"  AbsRel: {metrics['absrel']:.4f} | delta1: {metrics['delta1']:.2f}% | RMSE: {metrics['rmse']:.4f}m")
            return {'metrics': metrics}
        return None


    # =========================================================================
    # 3. Trajectory visualization
    # =========================================================================
    def visualize_trajectory(self, scene_dir, output_path, max_frames=60, calibration_frames=5):
        print(f"\n Rendering trajectory prediction video (auto-axis alignment + advanced metrics)...")
        
        color_dir = scene_dir / 'color'
        pose_dir = scene_dir / 'pose'
        frame_files = sorted(list(color_dir.glob('*.jpg')) + list(color_dir.glob('*.png')))[:max_frames]
        if len(frame_files) < 10: return None
        gt_positions = []
        for f in frame_files:
            p_file = pose_dir / (f.stem + '.txt')
            if p_file.exists():
                try: 
                    pose = np.loadtxt(p_file).reshape(4, 4)
                    gt_positions.append(pose[:3, 3])
                except: 
                    gt_positions.append(gt_positions[-1] if gt_positions else np.zeros(3))
            else: 
                gt_positions.append(gt_positions[-1] if gt_positions else np.zeros(3))
        gt_positions = np.array(gt_positions)
        self.clear_temporal_buffer()
        raw_predictions = [] # [N, 6]
        prev_feat = None
        
        print(f"  Processing {len(frame_files)} frames (Inference)...")
        for i, frame_file in enumerate(tqdm(frame_files, desc="  Inference")):
            img = Image.open(frame_file).convert('RGB')
            feat = self.extract_features(img)
            
            if prev_feat is not None:
                with torch.no_grad():
                    outputs, _ = self.unified_model(feat, prev_feat, tasks=['motion'])
                    raw_motion = outputs['motion'].cpu().numpy()[0]
                    raw_predictions.append(raw_motion)
            prev_feat = feat
            
        if not raw_predictions: return None
        raw_predictions = np.array(raw_predictions)
        
        # ============================================================
        # ============================================================
        import itertools
        def calc_trajectory_error(perm, signs, raw_preds, gt_pos, calib_len):
            trans = raw_preds[:, :3]
            aligned_trans = np.stack([
                trans[:, perm[0]] * signs[0],
                trans[:, perm[1]] * signs[1],
                trans[:, perm[2]] * signs[2]
            ], axis=1)
            gt_deltas = gt_pos[1:] - gt_pos[:-1]
            gt_dist_sum = np.sum(np.linalg.norm(gt_deltas[:calib_len], axis=1))
            pred_dist_sum = np.sum(np.linalg.norm(aligned_trans[:calib_len], axis=1))
            
            scale = gt_dist_sum / max(pred_dist_sum, 1e-6)
            scale = np.clip(scale, 0.1, 50.0)
            traj = [gt_pos[0]]
            curr = gt_pos[0].copy()
            
            check_len = min(len(aligned_trans), calib_len * 3)
            gt_d = gt_deltas[:check_len]
            pr_d = aligned_trans[:check_len] * scale
            delta_error = np.mean(np.linalg.norm(gt_d - pr_d, axis=1))
            return delta_error, scale
        permutations = list(itertools.permutations([0, 1, 2]))
        signs = list(itertools.product([1, -1], repeat=3))
        
        best_score = float('inf')
        best_config = None # (perm, sign, scale)
        
        print("   Searching best axis alignment...")
        for p in permutations:
            for s in signs:
                score, scale = calc_trajectory_error(p, s, raw_predictions, gt_positions, calibration_frames)
                if score < best_score:
                    best_score = score
                    best_config = (p, s, scale)
        
        best_perm, best_sign, best_scale = best_config
        axis_map = ['X', 'Y', 'Z']
        print(f"   Best alignment: [{best_sign[0]}{axis_map[best_perm[0]]}, {best_sign[1]}{axis_map[best_perm[1]]}, {best_sign[2]}{axis_map[best_perm[2]]}]")
        print(f"    Best scale: {best_scale:.4f}")

        # ============================================================
        # ============================================================
        final_trans = np.stack([
            raw_predictions[:, 0+best_perm[0]] * best_sign[0],
            raw_predictions[:, 0+best_perm[1]] * best_sign[1],
            raw_predictions[:, 0+best_perm[2]] * best_sign[2]
        ], axis=1)
        final_rot = np.stack([
            raw_predictions[:, 3+best_perm[0]] * best_sign[0],
            raw_predictions[:, 3+best_perm[1]] * best_sign[1],
            raw_predictions[:, 3+best_perm[2]] * best_sign[2]
        ], axis=1)

        current_pos = gt_positions[0].copy()
        pred_positions = [current_pos.copy()]
        first_pose_file = pose_dir / (frame_files[0].stem + '.txt')
        if first_pose_file.exists():
            current_R = np.loadtxt(first_pose_file).reshape(4, 4)[:3, :3]
        else:
            current_R = np.eye(3)

        frame_width, frame_height = 1280, 720
        out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), 10, (frame_width, frame_height))
        def euler_to_matrix(rx, ry, rz):
            cx, sx = np.cos(rx), np.sin(rx)
            cy, sy = np.cos(ry), np.sin(ry)
            cz, sz = np.cos(rz), np.sin(rz)
            Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
            Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
            return Rz @ Ry @ Rx

        for i in range(len(frame_files)):
            if i > 0 and i-1 < len(final_trans):
                step_local = final_trans[i-1] * best_scale
                step_world = current_R @ step_local
                current_pos = current_pos + step_world
                pred_positions.append(current_pos.copy())
                delta_R = euler_to_matrix(*final_rot[i-1])
                current_R = current_R @ delta_R
            canvas = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            img_cv = cv2.resize(cv2.imread(str(frame_files[i])), (640, 360))
            canvas[20:380, 20:660] = img_cv
            
            center_x, center_y = 960, 360
            viz_scale = 50 
            ax1, ax2 = 0, 1 
            
            # Draw GT
            for j in range(1, i+1):
                p1, p2 = gt_positions[j-1], gt_positions[j]
                cv2.line(canvas, 
                        (int(center_x + (p1[ax1]-gt_positions[0][ax1])*viz_scale), int(center_y - (p1[ax2]-gt_positions[0][ax2])*viz_scale)),
                        (int(center_x + (p2[ax1]-gt_positions[0][ax1])*viz_scale), int(center_y - (p2[ax2]-gt_positions[0][ax2])*viz_scale)),
                        (100, 255, 100), 2)
            
            # Draw Pred
            curr_pred_len = len(pred_positions)
            for j in range(1, curr_pred_len):
                p1, p2 = pred_positions[j-1], pred_positions[j]
                cv2.line(canvas, 
                        (int(center_x + (p1[ax1]-gt_positions[0][ax1])*viz_scale), int(center_y - (p1[ax2]-gt_positions[0][ax2])*viz_scale)),
                        (int(center_x + (p2[ax1]-gt_positions[0][ax1])*viz_scale), int(center_y - (p2[ax2]-gt_positions[0][ax2])*viz_scale)),
                        (100, 100, 255), 2)

            curr_ate = 0.0
            if i < len(pred_positions):
                curr_ate = np.linalg.norm(gt_positions[i] - pred_positions[i])
            
            cv2.putText(canvas, f"Axis: [{best_sign[0]}{axis_map[best_perm[0]]}, {best_sign[1]}{axis_map[best_perm[1]]}, {best_sign[2]}{axis_map[best_perm[2]]}]", 
                       (20, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            cv2.putText(canvas, f"Scale: {best_scale:.1f}x", (20, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(canvas, f"ATE: {curr_ate:.3f}m", (20, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            out.write(canvas)
            
        out.release()
        
        # ============================================================
        # ============================================================
        min_len = min(len(gt_positions), len(pred_positions))
        if min_len > 1:
            ate_scores = [np.linalg.norm(gt_positions[k] - pred_positions[k]) for k in range(min_len)]
            ate_array = np.array(ate_scores)
            mae = float(np.mean(ate_array))
            rmse = float(np.sqrt(np.mean(ate_array**2)))
            max_ate = float(np.max(ate_array))
            final_error = float(ate_array[-1])
            gt_steps = np.linalg.norm(gt_positions[1:] - gt_positions[:-1], axis=1)
            total_distance = np.sum(gt_steps)
            drift_ratio = (final_error / total_distance) * 100 if total_distance > 0 else 0.0
            pred_steps = np.linalg.norm(np.array(pred_positions)[1:] - np.array(pred_positions)[:-1], axis=1)
            min_step_len = min(len(gt_steps), len(pred_steps))
            rpe_trans = np.mean(np.abs(gt_steps[:min_step_len] - pred_steps[:min_step_len]))

            metrics = {
                'ate_mean': mae,
                'ate_rmse': rmse,
                'ate_max': max_ate,
                'final_drift': final_error,
                'drift_ratio': drift_ratio,
                'rpe_trans': float(rpe_trans)
            }
            
            print("  [Advanced Metrics]")
            print(f"  RMSE (ATE):  {metrics['ate_rmse']:.4f} m")
            print(f"  Max ATE:     {metrics['ate_max']:.4f} m")
            print(f"  RPE (Local): {metrics['rpe_trans']:.4f} m/frame")
            print(f"  Drift Ratio: {metrics['drift_ratio']:.2f} %")
            
            return {'metrics': metrics}
        
        return None

    # =========================================================================
    # 4. Occlusion test visualization
    # =========================================================================
    def visualize_occlusion_test(self, scene_dir, output_path, max_frames=60,
                                 mode='continuous', occlusion_type='black',
                                 injection_method='full', anomaly_threshold=0.25):
        """
        Args:
            mode: 'continuous', 'interval', or 'random'
        """
        print(f"\n Rendering occlusion test visualization (mode: {mode})...")
        self.clear_temporal_buffer()
        memory_buffer = AdaptiveMemoryBuffer(max_size=8, anomaly_threshold=anomaly_threshold)
        
        color_dir = scene_dir / 'color'
        frame_files = sorted(list(color_dir.glob('*.jpg')) + list(color_dir.glob('*.png')))[:max_frames]
        if len(frame_files) < 10: return None
        occlusion_frame_list = []
        if mode == 'continuous':
            start_frame = int(max_frames * 0.2)
            duration = int(max_frames * 0.4)
            occlusion_frame_list = list(range(start_frame, start_frame + duration))
            print(f"   Mode: continuous occlusion (Frames {start_frame} -> {start_frame + duration})")
        elif mode == 'interval':
            period, block = 8, 3
            for i in range(max_frames):
                if (i % period) < block and i > 5:
                    occlusion_frame_list.append(i)
            print(f"   Mode: interval occlusion (every {period} frames, occlude {block} frames)")
        elif mode == 'random':
            rng = random.Random(42)
            for i in range(5, max_frames):
                if rng.random() < 0.3: occlusion_frame_list.append(i)
            print(f"   Mode: random occlusion")

        frame_width, frame_height = 1280, 720
        out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), 5, (frame_width, frame_height))
        results = []
        yolo_occluder = None
        if occlusion_type.startswith('yolo_') and YOLO_AVAILABLE:
            yolo_occluder = YOLOOccluder(model_size='n', confidence_threshold=0.15)
        
        for i, frame_file in enumerate(tqdm(frame_files, desc="  Processing frames")):
            original_img = Image.open(frame_file).convert('RGB')
            original_cv = cv2.cvtColor(np.array(original_img), cv2.COLOR_RGB2BGR)
            
            is_occluded = i in occlusion_frame_list
            occluded_cv = original_cv.copy()
            
            occluded_object_info = None
            
            if is_occluded:
                h, w = occluded_cv.shape[:2]
                cx, cy = w // 2, h // 2
                if occlusion_type.startswith('yolo_') and yolo_occluder:
                    occluded_cv, selected, _ = yolo_occluder.occlude_multiple_objects(
                        occluded_cv, max_objects=2, min_area=1000
                    )
                    if selected:
                        occluded_object_info = {'objects': selected}
                    else:
                        # Fallback
                        size = int(min(w, h) * 0.4 / 2)
                        cv2.rectangle(occluded_cv, (cx-size, cy-size), (cx+size, cy+size), (0, 0, 0), -1)
                else:
                    size = int(min(w, h) * 0.4 / 2)
                    cv2.rectangle(occluded_cv, (cx-size, cy-size), (cx+size, cy+size), (0, 0, 0), -1)
                
                input_img = Image.fromarray(cv2.cvtColor(occluded_cv, cv2.COLOR_BGR2RGB))
            else:
                input_img = original_img
            feat = self.extract_features(input_img)
            edge_feat = self.extract_edge_features(input_img)
            
            result = memory_buffer.add_frame(feat, i, input_img, edge_feat=edge_feat)
            if len(result) == 5:
                added, quality, anomaly_score, is_anomaly, debug = result
            else:
                added, quality, anomaly_score, is_anomaly = result
            injected_response = ""
            if is_anomaly and is_occluded and len(memory_buffer.features) > 0:
                best_memory, score, info = memory_buffer.get_best_memory(feat, i, edge_feat=edge_feat)
                if best_memory is not None:
                    res = self.generate_with_injection_two_stage(
                        input_img, best_memory, 
                        injection_strength=0.5, 
                        injection_method=injection_method,
                        occlusion_info=occluded_object_info
                    )
                    injected_response = res['combined']
            canvas = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            canvas[:] = (30, 30, 30)
            canvas[20:300, 20:480] = cv2.resize(original_cv, (460, 280)) # GT
            canvas[20:300, 500:960] = cv2.resize(occluded_cv, (460, 280)) # Input
            
            cv2.putText(canvas, "GT", (20, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            cv2.putText(canvas, f"Input (Occluded: {is_occluded})", (500, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                       (0,0,255) if is_occluded else (0,255,0), 1)
            status_color = (0, 0, 255) if is_anomaly else (0, 255, 0)
            cv2.putText(canvas, f"Anomaly Detected: {is_anomaly}", (20, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            if injected_response:
                cv2.putText(canvas, "Memory Injection Active", (20, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 100), 2)
                lines = injected_response.split('\n')
                for k, line in enumerate(lines[:5]):
                    cv2.putText(canvas, line[:80], (20, 480 + k*25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            
            out.write(canvas)
            
            results.append({
                'frame': i, 
                'is_occluded': is_occluded, 
                'is_anomaly': is_anomaly,
                'injected_response': injected_response
            })

        out.release()
        occluded_frames = [r for r in results if r['is_occluded']]
        detected = [r for r in occluded_frames if r['is_anomaly']]
        detection_rate = len(detected) / max(len(occluded_frames), 1)
        
        return {'detection_rate': detection_rate, 'detailed_results': results}

    def _print_global_averages(self, all_stats):
        """Compute and print average metrics across scenes, then return a summary"""
        print(f"\n{'='*70}")
        print(f" FINAL REPORT: GLOBAL AVERAGE METRICS (Across {len(all_stats)} Scenes)")
        print(f"{'='*70}")
        
        global_summary = {}
        depth_metrics = {'absrel': [], 'delta1': [], 'rmse': [], 'mae': []}
        for s in all_stats.values():
            if 'depth' in s and s['depth'] and 'metrics' in s['depth']:
                m = s['depth']['metrics']
                for k in depth_metrics:
                    if k in m: depth_metrics[k].append(m[k])
        
        if depth_metrics['absrel']:
            avg_absrel = np.mean(depth_metrics['absrel'])
            avg_rmse = np.mean(depth_metrics['rmse'])
            avg_mae = np.mean(depth_metrics['mae'])
            avg_d1 = np.mean(depth_metrics['delta1'])
            
            print("\n  [Depth Regression]")
            print(f"  Avg AbsRel: {avg_absrel:.4f}")
            print(f"  Avg RMSE:   {avg_rmse:.4f} m")
            print(f"  Avg MAE:    {avg_mae:.4f} m")
            print(f"  Avg delta1: {avg_d1:.2f} %")
            
            global_summary['depth_regression'] = {
                'avg_absrel': avg_absrel,
                'avg_rmse': avg_rmse,
                'avg_mae': avg_mae,
                'avg_delta1': avg_d1
            }
        traj_metrics = {'ate_rmse': [], 'rpe_trans': [], 'drift_ratio': [], 'ate_mean': []}
        
        for s in all_stats.values():
            if 'motion' in s and s['motion'] and 'metrics' in s['motion']:
                m = s['motion']['metrics']
                if 'ate_rmse' in m:
                    traj_metrics['ate_rmse'].append(m['ate_rmse'])
                    traj_metrics['rpe_trans'].append(m['rpe_trans'])
                    traj_metrics['drift_ratio'].append(m['drift_ratio'])
                    traj_metrics['ate_mean'].append(m['ate_mean'])
                elif 'ate' in m:
                    traj_metrics['ate_mean'].append(m['ate'])
        
        if traj_metrics['ate_mean']:
            avg_ate_mean = np.mean(traj_metrics['ate_mean'])
            
            print("\n  [Trajectory Prediction]")
            print(f"  Avg ATE (Mean): {avg_ate_mean:.4f} m")
            
            global_summary['trajectory'] = {'avg_ate_mean': avg_ate_mean}
            if traj_metrics['ate_rmse']:
                avg_rmse = np.mean(traj_metrics['ate_rmse'])
                avg_rpe = np.mean(traj_metrics['rpe_trans'])
                avg_drift = np.mean(traj_metrics['drift_ratio'])
                
                print(f"  Avg ATE (RMSE): {avg_rmse:.4f} m")
                print(f"  Avg RPE:        {avg_rpe:.4f} m/frame")
                print(f"  Avg Drift:      {avg_drift:.2f} %")
                
                global_summary['trajectory'].update({
                    'avg_ate_rmse': avg_rmse,
                    'avg_rpe': avg_rpe,
                    'avg_drift': avg_drift
                })
                
            
        print(f"\n{'='*70}\n")
        return global_summary

    def run_complete_demo(self, data_root, output_dir, dataset='scannet', max_scenes=3,
                          occlusion_mode='continuous', calibration_frames=10,
                          demos=None, split='test',
                          occlusion_type='black',
                          injection_method='full',
                          anomaly_threshold=0.25):
        
        if demos is None: demos = ['temporal', 'depth', 'motion', 'occlusion']
        
        data_root = Path(data_root)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        scene_dirs = []
        if dataset == 'scannet':
            print(f" Searching ScanNet scenes in: {data_root}")
            if (data_root / 'color').exists():
                scene_dirs = [data_root]
            else:
                scene_dirs = [d for d in data_root.iterdir() if d.is_dir() and (d/'color').exists()]
        
        elif dataset == 'nyu':
            print(f" Searching NYU v2 scenes in: {data_root}")
            scene_dirs = [d for d in data_root.iterdir() if d.is_dir()]
            scene_dirs = [d for d in scene_dirs if (d/'color').exists()]
        
        if not scene_dirs:
            print(f" No valid scenes found for dataset '{dataset}' in {data_root}")
            return
        scene_dirs = sorted(scene_dirs)[:max_scenes]
        print(f" Selected {len(scene_dirs)} scenes for testing.")

        all_stats = {}
        for scene_dir in scene_dirs:
            scene_name = scene_dir.name
            scene_out = output_dir / scene_name
            scene_out.mkdir(exist_ok=True)
            scene_stats = {}
            
            print(f"\n{'='*70}")
            print(f" Processing scene: {scene_name}")
            print(f"{'='*70}")
            
            try:
                if 'temporal' in demos:
                    scene_stats['temporal'] = self.visualize_temporal_consistency(
                        scene_dir, scene_out / 'temporal.mp4')

                if 'depth' in demos:
                    scene_stats['depth'] = self.visualize_depth_regression(
                        scene_dir, scene_out / 'depth_regr.mp4', calibration_frames=calibration_frames)
                
                if 'motion' in demos:
                    scene_stats['motion'] = self.visualize_trajectory(
                        scene_dir, scene_out / 'traj.mp4', calibration_frames=5) 
                
                if 'occlusion' in demos:
                    scene_stats['occlusion'] = self.visualize_occlusion_test(
                        scene_dir, 
                        scene_out / 'occ.mp4', 
                        mode=occlusion_mode,
                        occlusion_type=occlusion_type,
                        injection_method=injection_method,
                        anomaly_threshold=anomaly_threshold
                    )
                
                all_stats[scene_name] = scene_stats
                print(f" {scene_name} completed")
            except Exception as e:
                print(f" {scene_name} failed: {e}")
                import traceback
                traceback.print_exc()
        with open(output_dir / 'summary.json', 'w') as f:
            def convert(o):
                if isinstance(o, np.float32): return float(o)
                return o
            json.dump(all_stats, f, indent=2, default=convert)
        global_metrics = self._print_global_averages(all_stats)
        final_report = {
            "meta": {
                "timestamp": datetime.now().isoformat(),
                "dataset": dataset,
                "total_scenes": len(scene_dirs),
                "model_checkpoint": str(self.checkpoint_path)
            },
            "global_metrics": global_metrics,
            "scene_details": {}
        }
        for s_name, s_data in all_stats.items():
            s_summary = {}
            if 'depth' in s_data and s_data['depth']:
                s_summary['depth'] = s_data['depth']['metrics']
            if 'motion' in s_data and s_data['motion']:
                s_summary['motion'] = s_data['motion']['metrics']
            final_report["scene_details"][s_name] = s_summary
            
        with open(output_dir / 'final_report.json', 'w', encoding='utf-8') as f:
            json.dump(final_report, f, indent=2, ensure_ascii=False, default=convert)
            
        print(f" Final report saved: {output_dir / 'final_report.json'}")

def main():
    parser = argparse.ArgumentParser(description='TempoVLM Complete Demo')
    
    # Core path arguments
    parser.add_argument('--model_path', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--data_root', type=str, required=True, help='Dataset root directory')
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory')
    
    # Dataset and evaluation settings
    parser.add_argument('--dataset', type=str, default='scannet', choices=['scannet', 'nyu'], help='Dataset type')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'test', 'all'], help='Dataset split')
    parser.add_argument('--max_scenes', type=int, default=3, help='Maximum number of test scenes')
    parser.add_argument('--device', type=str, default='cuda', help='Compute device')
    
    # Demo selection
    parser.add_argument('--demos', type=str, default='all', 
                        help='Demos to run (comma-separated): temporal,depth,motion,occlusion, or all')
    
    # Validation parameters
    parser.add_argument('--occlusion_mode', type=str, default='continuous', 
                        choices=['continuous', 'interval', 'random'], 
                        help='Occlusion test mode')
    parser.add_argument('--calibration_frames', type=int, default=0, 
                        help='Number of initial frames for scale calibration (then lock)')
    
    # Advanced parameters
    parser.add_argument('--occlusion_type', type=str, default='black', help='Occlusion type')
    parser.add_argument('--injection_method', type=str, default='full', help='Memory injection method')
    parser.add_argument('--anomaly_threshold', type=float, default=0.25, help='Anomaly detection threshold')

    args = parser.parse_args()
    
    # Parse demo list
    if args.demos == 'all':
        demos = ['temporal', 'depth', 'motion', 'occlusion']
    else:
        demos = [d.strip() for d in args.demos.split(',')]
    
    # Initialize and run
    vis = CompleteDemoVisualizer(args.model_path, device=args.device)
    vis.run_complete_demo(
        data_root=args.data_root,
        output_dir=args.output_dir,
        dataset=args.dataset,
        split=args.split,
        max_scenes=args.max_scenes,
        occlusion_mode=args.occlusion_mode,
        calibration_frames=args.calibration_frames,
        demos=demos,
        # Pass advanced parameters
        occlusion_type=args.occlusion_type,
        injection_method=args.injection_method,
        anomaly_threshold=args.anomaly_threshold
    )

if __name__ == '__main__':
    main()
