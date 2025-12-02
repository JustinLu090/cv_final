#!/usr/bin/env python3
"""
occlusion_tester.py - 遮擋測試器核心類
======================================

包含 OcclusionTester 類，提供:
- 特徵提取
- 場景描述生成
- Direct Feature Injection
- 遮擋應用

使用方式:
    from occlusion_tester import OcclusionTester
    
    tester = OcclusionTester(unified_model_path='path/to/model.pt')
    
    # 提取特徵
    feat = tester.extract_features(image)
    
    # 生成描述
    desc = tester.generate_description(image, prompt="Describe.")
    
    # Direct Injection
    response = tester.generate_with_direct_injection(
        current_image=occluded_img,
        enhanced_feat=memory_feat,
        prompt="Describe the center.",
        injection_method='raw',
        injection_strength=0.4
    )
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


class OcclusionTester:
    """遮擋測試器 - 核心類"""
    
    def __init__(self, unified_model_path=None, device='cuda'):
        self.device = device
        
        print("=" * 70)
        print("🧪 TempoVLM Occlusion Tester")
        print("=" * 70)
        
        # 載入 Qwen2-VL
        print("\n📦 載入 Qwen2-VL...")
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

        self.unified_model = None
        
        self.temporal_buffer = []
        self.temporal_buffer_size = 8
        
        if unified_model_path and os.path.exists(unified_model_path):
            self._load_unified_model(unified_model_path)
        else:
            print(f"⚠️ 未載入 UnifiedTempoVLM (將使用原始 VLM 特徵)")
            if unified_model_path:
                print(f"   路徑不存在: {unified_model_path}")
        
        # Feature Injection 元件
        self.feature_projector = None
        self._vision_hidden_size = None
        # 最新一幀的 Adapter 輸出品質 (GRU 版本才有)
        self.last_adapter_meta = None
        
        print("✅ 模型載入完成！")
    
    def _load_unified_model(self, model_path):
        """載入 Unified Model，自動相容新舊版本"""
        print(f"📦 載入 UnifiedTempoVLM: {model_path}")
        
        from models_unified import UnifiedTempoVLM
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # 檢測 hidden_dim
        if 'shared_encoder.0.weight' in state_dict:
            hidden_dim = state_dict['shared_encoder.0.weight'].shape[0]
        else:
            hidden_dim = 768
        
        # 檢測是否為新版 checkpoint（有 GRU）
        has_gru = any('temporal_gru' in k for k in state_dict.keys())
        
        # 檢測深度輸出維度（新版=3, 舊版=1）
        if 'depth_regression_head.2.bias' in state_dict:
            depth_dim = state_dict['depth_regression_head.2.bias'].shape[0]
        else:
            depth_dim = 3
        
        print(f"   Checkpoint 類型: {'新版 (GRU)' if has_gru else '舊版 (無 GRU)'}")
        print(f"   hidden_dim: {hidden_dim}")
        print(f"   深度輸出維度: {depth_dim}")
        
        # 建立對應架構的模型
        self.unified_model = UnifiedTempoVLM(
            hidden_dim=hidden_dim,
            use_gru_memory=has_gru
        ).to(self.device)
        
        # 處理 size mismatch 的問題（手動過濾）
        model_state = self.unified_model.state_dict()
        filtered_state_dict = {}
        skipped_keys = []
        
        for k, v in state_dict.items():
            if k in model_state:
                if v.shape == model_state[k].shape:
                    filtered_state_dict[k] = v
                else:
                    skipped_keys.append(f"{k}: checkpoint {v.shape} vs model {model_state[k].shape}")
            else:
                # key 不存在於模型中，跳過
                pass
        
        if skipped_keys:
            print(f"   ⚠️ 跳過 size 不匹配的權重:")
            for sk in skipped_keys[:5]:
                print(f"      - {sk}")
        
        # 載入過濾後的 state_dict
        missing, unexpected = self.unified_model.load_state_dict(filtered_state_dict, strict=False)
        
        if missing:
            # 過濾掉預期內的缺失（新模組）
            expected_missing = ['motion_scale', 'temporal_gru', 'memory_quality_gate', 
                               'memory_output_gate', 'motion_uncertainty_head', 
                               'velocity_smoothing', 'global_scale_head', 
                               'motion_quality_head', 'place_embedding',
                               'depth_regression_head']  # 舊版深度也算預期內
            real_missing = [k for k in missing if not any(exp in k for exp in expected_missing)]
            if real_missing:
                print(f"   ⚠️ 缺少非預期的權重: {real_missing[:5]}...")
        
        self.unified_model.eval()
        self.unified_model.float()  # 使用 float32 避免精度問題
        
        # 記錄模型類型
        self.use_gru = has_gru
        self.gru_hidden_state = None  # GRU 隱藏狀態
        
        print(f"✅ UnifiedTempoVLM 載入完成 (GRU記憶: {'啟用' if has_gru else '停用'})")
    
    def extract_features(self, image, use_adapter=True):
        """
        提取特徵 (Qwen2-VL hidden states)
        
        Args:
            image: 輸入圖像
            use_adapter: 是否使用 Adapter 增強特徵（預設 True）
        
        Returns:
            features: 如果 use_adapter=True 且有 Adapter，返回增強後的特徵
                     否則返回原始 VLM 特徵
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
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
            raw_features = outputs.hidden_states[-1].mean(dim=1)
        
        # 如果有 Adapter 且啟用，使用 Adapter 增強特徵
        # 重設上一幀的 Adapter 品質資訊
        self.last_adapter_meta = None
        if use_adapter and self.unified_model is not None:
            enhanced_features = self._enhance_with_adapter(raw_features)
            return enhanced_features
        
        return raw_features
    
    def _enhance_with_adapter(self, current_feat):
        """
        使用 UnifiedTempoVLM Adapter 增強特徵
        
        整合時序資訊，讓特徵包含過去幀的上下文
        支援新版 (GRU) 和舊版模型
        
        Args:
            current_feat: 當前幀的原始特徵 [1, feat_dim]
        
        Returns:
            enhanced_feat: 時序增強後的特徵 [1, feat_dim]
        """
        # 維護緩衝區大小
        if len(self.temporal_buffer) > self.temporal_buffer_size:
            self.temporal_buffer.pop(0)
        
        # 如果緩衝區幀數不足，直接返回原始特徵
        if len(self.temporal_buffer) < 1:
            # 第一幀，沒有前一幀可用
            self.temporal_buffer.append(current_feat.clone())
            return current_feat
        
        try:
            with torch.no_grad():
                # UnifiedTempoVLM.forward() 需要 curr_feat 和 prev_feat
                # curr_feat: [B, feat_dim], prev_feat: [B, feat_dim]
                prev_feat = self.temporal_buffer[-1]  # 最後一幀作為 prev
                
                # 確保維度正確 [1, feat_dim]
                if current_feat.dim() == 1:
                    current_feat = current_feat.unsqueeze(0)
                if prev_feat.dim() == 1:
                    prev_feat = prev_feat.unsqueeze(0)
                
                # 確保使用 float32
                current_feat = current_feat.float()
                prev_feat = prev_feat.float()
                
                adapter_meta = {}
                if hasattr(self, 'use_gru') and self.use_gru:
                    outputs, self.gru_hidden_state = self.unified_model(
                        curr_feat=current_feat,
                        prev_feat=prev_feat,
                        hidden_state=self.gru_hidden_state,
                        tasks=['temporal']
                    )
                else:
                    result = self.unified_model(
                        curr_feat=current_feat,
                        prev_feat=prev_feat,
                        tasks=['temporal']
                    )
                    if isinstance(result, tuple):
                        outputs, _ = result
                    else:
                        outputs = result
                
                if isinstance(outputs, dict) and 'temporal' in outputs and outputs['temporal'] is not None:
                    enhanced_current = outputs['temporal']  # [1, feat_dim]
                    # 捕捉新架構的品質門控訊息（如果有）
                    if 'memory_quality' in outputs and outputs['memory_quality'] is not None:
                        adapter_meta['memory_quality'] = float(outputs['memory_quality'])
                    if 'temporal_gate' in outputs and outputs['temporal_gate'] is not None:
                        adapter_meta['temporal_gate'] = float(outputs['temporal_gate'])
                else:
                    # 沒有 temporal 輸出，返回原始特徵
                    enhanced_current = current_feat
                
                # 更新緩衝區（用原始特徵，不是增強後的）
                self.temporal_buffer.append(current_feat.clone())
                self.last_adapter_meta = adapter_meta if adapter_meta else None
                
                return enhanced_current
                
        except Exception as e:
            print(f"⚠️ Adapter 增強失敗: {e}")
            # 仍然更新緩衝區
            self.temporal_buffer.append(current_feat.clone())
            self.last_adapter_meta = None
            return current_feat
    
    def extract_features_raw(self, image):
        """
        提取原始特徵（不經過 Adapter）
        
        用於需要原始特徵的場景，如異常檢測
        """
        return self.extract_features(image, use_adapter=False)
    
    def extract_edge_features(self, image, edge_ratio=0.25):
        """
        提取圖像邊緣區域的特徵
        
        用於場景變化檢測 - 邊緣通常不受中心遮擋影響
        
        Args:
            image: 輸入圖像
            edge_ratio: 邊緣區域佔比（從每邊取這個比例）
        
        Returns:
            edge_feat: 邊緣區域的特徵
        """
        if isinstance(image, np.ndarray):
            img_array = image
        else:
            img_array = np.array(image)
        
        h, w = img_array.shape[:2]
        edge_h = int(h * edge_ratio)
        edge_w = int(w * edge_ratio)
        
        # 提取四個邊緣區域
        top = img_array[:edge_h, :, :]
        bottom = img_array[-edge_h:, :, :]
        left = img_array[edge_h:-edge_h, :edge_w, :]
        right = img_array[edge_h:-edge_h, -edge_w:, :]
        
        # 組合邊緣區域成一張圖
        # 使用上邊緣和下邊緣的平均特徵（最穩定）
        edge_combined = np.vstack([top, bottom])
        
        edge_img = Image.fromarray(edge_combined)
        
        # 提取特徵（不使用 Adapter，避免時序干擾）
        return self.extract_features(edge_img, use_adapter=False)
    
    def clear_temporal_buffer(self):
        """清空時序緩衝區和 GRU 隱藏狀態（場景切換時使用）"""
        self.temporal_buffer = []
        if hasattr(self, 'gru_hidden_state'):
            self.gru_hidden_state = None
        self.last_adapter_meta = None
    
    def compute_similarity(self, feat1, feat2):
        """計算餘弦相似度"""
        return F.cosine_similarity(
            feat1.flatten().unsqueeze(0),
            feat2.flatten().unsqueeze(0)
        ).item()
    
    def generate_description(self, image, prompt=None):
        """生成場景描述"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if prompt is None:
            prompt = "Describe what you see in this image in one sentence."
        
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
                max_new_tokens=100,
                do_sample=False
            )
        
        response = self.processor.decode(generated[0], skip_special_tokens=True)
        if "assistant" in response.lower():
            response = response.split("assistant")[-1].strip()
        return response
    
    def _get_vision_hidden_size(self):
        """獲取視覺特徵維度"""
        if self._vision_hidden_size is None:
            try:
                self._vision_hidden_size = self.base_model.config.vision_config.hidden_size
            except:
                self._vision_hidden_size = 1536
        return self._vision_hidden_size
    
    def _init_feature_projector(self, source_dim, target_dim):
        """初始化特徵投影層"""
        if self.feature_projector is None or \
           self.feature_projector.in_features != source_dim or \
           self.feature_projector.out_features != target_dim:
            
            self.feature_projector = torch.nn.Linear(source_dim, target_dim)
            with torch.no_grad():
                if source_dim == target_dim:
                    torch.nn.init.eye_(self.feature_projector.weight)
                else:
                    torch.nn.init.xavier_uniform_(self.feature_projector.weight)
                torch.nn.init.zeros_(self.feature_projector.bias)
            
            self.feature_projector = self.feature_projector.to(self.device).half()
    
    def generate_with_direct_injection(
        self, 
        current_image, 
        enhanced_feat,
        prompt=None,
        injection_method='raw',
        injection_strength=0.4
    ):
        """
        🧠 Direct Feature Injection
        
        將記憶特徵注入到視覺編碼器輸出中
        
        Args:
            current_image: 當前幀（可能被遮擋）
            enhanced_feat: 要注入的記憶特徵
            prompt: 提問
            injection_method: 'raw', 'full', 'strong', 'adaptive'
            injection_strength: 注入強度 (0-1)
        
        Returns:
            str: 生成的回答
        """
        if isinstance(current_image, np.ndarray):
            current_image = Image.fromarray(cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB))
        
        if prompt is None:
            prompt = "What objects are in the center of this image?"
        
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": current_image},
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
        
        enhanced_feat_copy = enhanced_feat.clone().detach()
        vision_hidden_size = self._get_vision_hidden_size()
        enhanced_dim = enhanced_feat_copy.shape[-1]
        
        if enhanced_dim != vision_hidden_size:
            self._init_feature_projector(enhanced_dim, vision_hidden_size)
        
        def create_injection_hook(method, strength):
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
                    
                    # 正規化：避免被遮擋圖的低方差把記憶壓扁，改用記憶統計為主、原圖為輔
                    proj_mean = projected_expanded.mean()
                    proj_std = projected_expanded.std() + 1e-6
                    blended_mean = 0.5 * proj_mean + 0.5 * orig_mean
                    blended_std = torch.max(0.7 * proj_std + 0.3 * orig_std, 0.5 * proj_std)
                    projected_normalized = (projected_expanded - proj_mean) / proj_std * blended_std + blended_mean
                    
                    # 生成中心遮罩，只在中心區域注入，減少對周邊上下文的干擾
                    if num_patches > 4:
                        side = int(num_patches ** 0.5)
                        if side * side == num_patches:
                            # 以 Chebyshev 距離衡量中心性，中心 60% 內注入
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
                    # 中心加權提升，讓遮擋區域的注入更明顯
                    center_mask = torch.clamp(center_mask * 1.5, max=1.0)
                    
                    # ============================================================
                    # 注入方法選擇
                    # ============================================================
                    
                    if method == 'full':
                        # 方法1: 全圖注入 (但僅中心區域)
                        mix = strength * center_mask
                        if output.dim() == 3:
                            modified = output + mix * (projected_normalized - output)
                        else:
                            modified = output + mix.squeeze(0) * (projected_normalized - output)
                    
                    elif method == 'strong':
                        # 方法2: 強力中心注入 - 中心區域高強度，邊緣低強度
                        modified = output.clone()
                        if output.dim() == 2:
                            num_patches = output.shape[0]
                            side = int(num_patches ** 0.5)
                            if side * side == num_patches:
                                for row in range(side):
                                    for col in range(side):
                                        idx = row * side + col
                                        # 計算到中心的距離
                                        dist_to_center = max(abs(row - side/2), abs(col - side/2)) / (side/2)
                                        # 中心強度高，邊緣強度低
                                        local_strength = strength * (1 - dist_to_center * 0.5)
                                        local_strength *= center_mask.view(-1)[idx].item()
                                        modified[idx] = (1 - local_strength) * output[idx] + local_strength * projected_normalized[idx]
                        elif output.dim() == 3:
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
                    
                    elif method == 'adaptive':
                        # 方法3: 自適應注入 - 根據特徵差異決定注入強度
                        # 差異大的 patch 注入更多
                        if output.dim() == 3:
                            # [batch, num_patches, hidden]
                            diff = torch.abs(output - projected_normalized).mean(dim=-1, keepdim=True)
                            diff_normalized = diff / (diff.max() + 1e-6)
                            # 差異越大，注入越強（因為可能是遮擋區域）
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
                    
                    # 限制數值範圍
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
                    max_new_tokens=100,
                    do_sample=False
                )
        finally:
            hook_handle.remove()
        
        # 解碼生成的文本
        full_response = self.processor.decode(generated[0], skip_special_tokens=True)
        
        # 提取 assistant 回答部分
        response = full_response
        
        # 嘗試多種分隔方式
        separators = ['assistant\n', 'assistant:', 'Assistant:', 'ASSISTANT:', '<|assistant|>']
        for sep in separators:
            if sep.lower() in response.lower():
                idx = response.lower().find(sep.lower())
                response = response[idx + len(sep):].strip()
                break
        
        # 如果回應仍然包含 prompt，嘗試移除
        if prompt and prompt in response:
            response = response.split(prompt)[-1].strip()
        
        # 如果回應為空，返回原始輸出（用於 debug）
        if not response.strip():
            # 可能是輸入和輸出混在一起，取最後一段
            lines = full_response.strip().split('\n')
            if lines:
                response = lines[-1].strip()
        
        return response
    
    def apply_occlusion(self, image, occ_type='box', ratio=0.3):

        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image.copy()
        
        h, w = img_array.shape[:2]
        cx, cy = w // 2, h // 2
        
        occ_w = int(w * ratio)
        occ_h = int(h * ratio)
        x1, y1 = cx - occ_w // 2, cy - occ_h // 2
        x2, y2 = cx + occ_w // 2, cy + occ_h // 2
        
        if occ_type == 'box':
            img_array[y1:y2, x1:x2] = 0
        elif occ_type == 'noise':
            noise = np.random.randint(0, 255, (y2-y1, x2-x1, 3), dtype=np.uint8)
            img_array[y1:y2, x1:x2] = noise
        elif occ_type == 'blur':
            roi = img_array[y1:y2, x1:x2]
            blurred = cv2.GaussianBlur(roi, (99, 99), 0)
            img_array[y1:y2, x1:x2] = blurred
        
        return Image.fromarray(img_array), (x1, y1, x2, y2)
