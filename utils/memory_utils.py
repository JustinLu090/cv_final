#!/usr/bin/env python3
"""
memory_utils.py - 記憶緩衝區和自適應注入工具
=============================================

包含:
- AdaptiveMemoryBuffer: 自適應記憶緩衝區，用於任意影片的記憶管理

使用方式:
    from memory_utils import AdaptiveMemoryBuffer
    
    buffer = AdaptiveMemoryBuffer(max_size=10, anomaly_threshold=0.3)
    
    # 逐幀處理
    for i, frame in enumerate(frames):
        feat = extract_features(frame)
        added, quality, anomaly_score, is_anomaly = buffer.add_frame(feat, i)
        
        if is_anomaly:
            best_memory, score, info = buffer.get_best_memory(feat, i)
            strength = buffer.compute_injection_strength(anomaly_score, score)
            # 執行注入...
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from collections import deque


class AdaptiveMemoryBuffer:
    """
    自適應記憶緩衝區
    
    用於在任意影片中維護高品質幀的記憶，並在檢測到異常時進行特徵注入。
    不需要預先知道遮擋發生的時間點。
    
    工作流程:
    1. 每幀計算品質分數（清晰度、特徵豐富度）
    2. 高品質幀加入記憶庫
    3. 檢測當前幀是否異常（可能被遮擋）
    4. 如果異常，從記憶庫選擇最佳特徵進行注入
    
    v2 更新:
    - 使用 Z-Score 動態閾值取代固定閾值
    - 自適應不同場景的變動程度
    """
    
    def __init__(self, max_size=10, anomaly_threshold=0.3, z_score_threshold=3.0):
        self.max_size = max_size
        self.anomaly_threshold = anomaly_threshold  # 冷啟動時的固定閾值
        self.z_score_threshold = z_score_threshold  # Z-Score 閾值（預設 3 個標準差）
        
        # 記憶儲存
        self.features = []      # 特徵張量
        self.edge_features = [] # NEW: 邊緣特徵 (用於場景與遮擋比對)
        self.qualities = []     # 品質分數 (0-1)
        self.timestamps = []    # 幀索引
        self.images = []        # 原始圖片 (可選)
        self.adapter_metas = [] # Adapter 輸出品質（GRU 版本）
        
        # 統計資訊
        self.feature_mean = None
        self.feature_std = None
        self.prev_feat = None
        self.last_valid_feat = None  # NEW: 上一個有效(未被遮擋)的特徵
        
        # 圖像統計（用於檢測遮擋）
        self.image_brightness_history = []
        
        # Z-Score 動態閾值所需的歷史紀錄
        self.anomaly_score_history = deque(maxlen=30)  # 記錄正常幀的異常分數
        self.feature_anomaly_history = deque(maxlen=30)  # 特徵層面異常分數
        self.image_anomaly_history = deque(maxlen=30)    # 圖像層面異常分數
        
    def compute_quality_score(self, feat, image=None):
        """
        計算幀的品質分數
        
        考慮因素:
        1. 特徵方差（豐富度）- 高方差 = 特徵豐富
        2. 特徵熵 - 均勻分佈 = 正常場景
        3. 與歷史均值的偏差 - 太偏離可能是異常
        """
        score = 0.0
        
        # 1. 特徵方差（正規化到 0-1）
        feat_var = feat.var().item()
        var_score = min(feat_var / 0.5, 1.0)
        score += 0.4 * var_score
        
        # 2. 特徵非零比例（避免大面積黑色/單色）
        nonzero_ratio = (feat.abs() > 0.01).float().mean().item()
        score += 0.3 * nonzero_ratio
        
        # 3. 與歷史均值的一致性
        if self.feature_mean is not None:
            consistency = F.cosine_similarity(
                feat.flatten().unsqueeze(0),
                self.feature_mean.flatten().unsqueeze(0)
            ).item()
            consistency_score = max(0, (consistency - 0.3) / 0.7)
            score += 0.3 * consistency_score
        else:
            score += 0.3
        
        return min(score, 1.0)
    
    def detect_image_occlusion(self, image):
        """
        圖像層面檢測遮擋（支援 YOLO 不規則遮擋和傳統中央遮擋）
        
        改進：不再限於中央區域，而是檢測整張圖片中的黑色/異常區域
        
        Returns:
            occlusion_score: 0-1, 越高越可能被遮擋
        """
        if image is None:
            return 0.0
        
        import numpy as np
        from PIL import Image as PILImage
        
        # 轉換為 numpy
        if isinstance(image, PILImage.Image):
            img_array = np.array(image)
        else:
            img_array = image
            
        h, w = img_array.shape[:2]
        
        occlusion_score = 0.0
        
        # ============================================================
        # 方法 1: 全圖黑色區域檢測（支援 YOLO bounding box 遮擋）
        # ============================================================
        if len(img_array.shape) == 3:
            brightness = img_array.mean(axis=2)
        else:
            brightness = img_array
        
        # 黑色像素比例（整張圖，不只中央）
        black_pixels = (brightness < 15).sum()
        black_ratio = black_pixels / (h * w)
        
        # 白色像素比例
        white_pixels = (brightness > 240).sum()
        white_ratio = white_pixels / (h * w)
        
        # 極端顏色總比例
        extreme_ratio = black_ratio + white_ratio
        
        # 任何位置的大面積黑色/白色都算遮擋
        if extreme_ratio > 0.15:  # 降低閾值，因為 YOLO 遮擋可能不到 25%
            occlusion_score += 0.8 * min(extreme_ratio / 0.4, 1.0)  # 40% 極端色 = 滿分
        
        # ============================================================
        # 方法 2: 連通區域檢測（檢測 YOLO bounding box）
        # ============================================================
        # 找出黑色連通區域
        black_mask = (brightness < 15).astype(np.uint8) * 255
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(black_mask, connectivity=8)
        
        # 檢查是否有大的黑色連通區域（可能是 YOLO 遮擋）
        large_black_regions = 0
        total_black_area = 0
        
        for i in range(1, num_labels):  # 跳過背景 (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            if area > 500:  # 大於 500 像素的黑色區域
                large_black_regions += 1
                total_black_area += area
        
        # 如果有多個大黑色區域（YOLO 遮擋多個物體）或單個超大區域
        if large_black_regions > 0:
            region_score = min(large_black_regions * 0.2, 0.5)  # 多個物件遮擋
            area_score = min(total_black_area / (h * w * 0.5), 0.5)  # 總面積
            occlusion_score += max(region_score, area_score)
        
        # ============================================================
        # 方法 3: 低紋理檢測（在有遮擋的區域）
        # ============================================================
        # 只檢測黑色/白色區域的紋理
        if extreme_ratio > 0.1:
            extreme_mask = ((brightness < 15) | (brightness > 240)).astype(np.uint8)
            masked_brightness = brightness * extreme_mask
            
            if extreme_mask.sum() > 100:  # 至少 100 個像素
                # 計算遮擋區域的 Laplacian 方差
                masked_uint8 = masked_brightness.astype(np.uint8)
                laplacian = cv2.Laplacian(masked_uint8, cv2.CV_64F)
                laplacian_var = laplacian.var()
                
                # 遮擋區域應該非常平滑
                if laplacian_var < 50:
                    occlusion_score += 0.2
        
        # ============================================================
        # 方法 4: 中央區域加權（傳統中央遮擋仍然重要）
        # ============================================================
        # 檢測中央區域，給予額外加權
        cx, cy = w // 2, h // 2
        margin = min(w, h) // 4
        center = img_array[cy-margin:cy+margin, cx-margin:cx+margin]
        
        if len(center.shape) == 3:
            center_brightness = center.mean(axis=2)
        else:
            center_brightness = center
        
        center_black_ratio = (center_brightness < 15).mean()
        center_white_ratio = (center_brightness > 240).mean()
        center_extreme = center_black_ratio + center_white_ratio
        
        # 中央區域遮擋給予額外分數
        if center_extreme > 0.3:
            occlusion_score += 0.3 * min(center_extreme, 1.0)
        
        return min(occlusion_score, 1.0)
    
    def detect_anomaly(self, current_feat, image=None):
        """
        檢測當前幀是否異常（可能被遮擋或損壞）
        
        結合特徵層面和圖像層面的檢測
        使用 Z-Score 動態閾值適應不同場景
        
        返回:
            anomaly_score: 0-1，越高越異常
            is_anomaly: bool
        """
        if len(self.features) < 3:
            return 0.0, False
        
        feature_anomaly = 0.0
        
        # === 特徵層面檢測 ===
        # 1. 與歷史均值的偏差
        if self.feature_mean is not None:
            mean_sim = F.cosine_similarity(
                current_feat.flatten().unsqueeze(0),
                self.feature_mean.flatten().unsqueeze(0)
            ).item()
            deviation = 1 - mean_sim
            feature_anomaly += 0.4 * deviation
        
        # 2. 與前一幀的突變程度
        if self.prev_feat is not None:
            temporal_sim = F.cosine_similarity(
                current_feat.flatten().unsqueeze(0),
                self.prev_feat.flatten().unsqueeze(0)
            ).item()
            temporal_diff = 1 - temporal_sim
            if temporal_diff > 0.3:
                feature_anomaly += 0.4 * min(temporal_diff, 1.0)
        
        # 3. 特徵方差異常低（大面積單色）
        feat_var = current_feat.var().item()
        if self.feature_std is not None:
            expected_var = self.feature_std.item() ** 2
            if feat_var < expected_var * 0.3:
                feature_anomaly += 0.2
        
        # === 圖像層面檢測 ===
        image_occlusion = self.detect_image_occlusion(image)
        
        # === 組合異常分數 ===
        # 特徵層面權重較高 (0.7)，因為它比較能反映語意變化
        anomaly_score = 0.7 * feature_anomaly + 0.3 * image_occlusion
        
        # === Z-Score 動態閾值判斷 ===
        is_anomaly = self._check_anomaly_zscore(anomaly_score, feature_anomaly, image_occlusion)
        
        return anomaly_score, is_anomaly
    
    def _check_anomaly_zscore(self, anomaly_score, feature_anomaly, image_occlusion):
        """
        使用 Z-Score 動態閾值判斷是否異常
        
        核心原則：
        - 我們主要關心的是「視覺遮擋」，而非「場景變化」
        - 圖像層面的遮擋檢測是最可靠的信號
        - 特徵層面的變化可能是正常的場景/視角變化
        
        判斷邏輯（以圖像遮擋為主，閾值更嚴格）：
        - 圖像遮擋分數 > 0.50 → 直接異常（明確遮擋）
        - 圖像遮擋分數 > 0.25 且 img_z_score > 閾值 → 異常
        - 其他情況 → 視為正常
        """
        min_history = 5  # 至少需要 5 幀歷史才啟用 Z-Score
        
        # 冷啟動：使用固定閾值，但以圖像遮擋為主
        if len(self.anomaly_score_history) < min_history:
            is_anomaly = image_occlusion > 0.60  # 冷啟動時閾值更高 (0.45 -> 0.60)
            
            # 如果不是異常，加入歷史
            if not is_anomaly:
                self.anomaly_score_history.append(anomaly_score)
                self.feature_anomaly_history.append(feature_anomaly)
                self.image_anomaly_history.append(image_occlusion)
            
            return is_anomaly
        
        # === Z-Score 計算 ===
        # 圖像層面的 Z-Score（最重要！）
        img_mu = np.mean(self.image_anomaly_history)
        img_sigma = np.std(self.image_anomaly_history) + 1e-6
        img_z_score = (image_occlusion - img_mu) / img_sigma
        
        # === 以圖像遮擋為主的判斷邏輯（更嚴格）===
        is_anomaly = False
        
        # 條件 1: 圖像遮擋分數絕對值很高（明確遮擋）
        # 條件 1: 圖像遮擋分數絕對值很高（明確遮擋）
        # 提高閾值從 0.30 到 0.50 (v4: 回調到 0.50, v5: 用戶要求提高到 0.60)
        if image_occlusion > 0.60:
            is_anomaly = True
        
        # 條件 2: 圖像有遮擋跡象 + 圖像 Z-Score 很高
        # 提高 img 閾值從 0.25 到 0.35
        elif image_occlusion > 0.35 and img_z_score > self.z_score_threshold * 1.2:
            is_anomaly = True
        
        # 條件 3: 圖像 Z-Score 極端高（很罕見的情況）
        elif img_z_score > self.z_score_threshold * 2.5 and image_occlusion > 0.20:
            is_anomaly = True
        
        # 只有正常幀才更新歷史（避免汙染統計數據）
        if not is_anomaly:
            self.anomaly_score_history.append(anomaly_score)
            self.feature_anomaly_history.append(feature_anomaly)
            self.image_anomaly_history.append(image_occlusion)
        
        return is_anomaly
    
    def add_frame(self, feat, timestamp, image=None, force=False, adapter_meta=None, edge_feat=None):
        """
        嘗試將幀加入記憶庫
        
        Args:
            edge_feat: 邊緣特徵 (可選)，如果未提供但有 image，可嘗試自動計算
            
        Returns:
            added: bool, 是否成功加入
            quality: float, 品質分數
            anomaly_score: float, 異常分數
            is_anomaly: bool, 是否異常
            debug_info: dict, 調試信息（包含圖像遮擋分數等）
        """
        quality = self.compute_quality_score(feat, image)
        anomaly_score, is_anomaly = self.detect_anomaly(feat, image)  # 傳入 image
        
        # 獲取圖像遮擋分數（用於調試）
        image_occlusion = self.detect_image_occlusion(image) if image is not None else 0.0
        debug_info = {
            'image_occlusion': image_occlusion,
            'feature_anomaly': anomaly_score - 0.5 * image_occlusion,  # 近似特徵異常
        }
        
        # 異常幀不加入記憶庫
        if is_anomaly and not force:
            self.prev_feat = feat.clone()
            return False, quality, anomaly_score, True, debug_info
        
        # 品質太低不加入
        if quality < 0.5 and not force:
            self.prev_feat = feat.clone()
            return False, quality, anomaly_score, False, debug_info
        
        # 加入記憶庫
        self.features.append(feat.clone())
        
        # 儲存邊緣特徵
        if edge_feat is not None:
             self.edge_features.append(edge_feat.clone())
        else:
             # 如果沒提供，暫時存主特徵 (雖然不理想，但比沒有好)
             self.edge_features.append(feat.clone())
             
        self.qualities.append(quality)
        self.timestamps.append(timestamp)
        if image is not None:
            self.images.append(image)
        self.adapter_metas.append(adapter_meta)
        
        # 維護大小限制
        while len(self.features) > self.max_size:
            if len(self.features) > 3:
                min_idx = min(range(len(self.features) - 3), 
                            key=lambda i: self.qualities[i])
                self.features.pop(min_idx)
                self.qualities.pop(min_idx)
                self.timestamps.pop(min_idx)
                if self.images:
                    self.images.pop(min_idx)
                if self.adapter_metas:
                    self.adapter_metas.pop(min_idx)
                if len(self.edge_features) > min_idx:
                    self.edge_features.pop(min_idx)
        
        self._update_statistics()
        self.prev_feat = feat.clone()
        
        # 如果這一幀是正常的(非異常)，它就是我們的"最後目擊證人"
        if not is_anomaly:
            self.last_valid_feat = feat.clone()
        
        return True, quality, anomaly_score, False, debug_info
    
    def _update_statistics(self):
        """更新特徵統計資訊"""
        if len(self.features) > 0:
            stacked = torch.stack(self.features)
            self.feature_mean = stacked.mean(dim=0)
            self.feature_std = stacked.std(dim=0, unbiased=False).mean()
    
    def get_best_memory(self, current_feat, current_timestamp, edge_feat=None):
        """
        選擇最佳記憶進行注入
        
        Args:
            current_feat: 當前幀完整特徵
            current_timestamp: 當前時間戳
            edge_feat: 邊緣區域特徵（可選，用於場景變化檢測）
        
        Returns:
            best_feat: 最佳記憶特徵
            best_score: 綜合分數
            best_info: 詳細資訊 dict（包含 scene_match 場景匹配度）
        """
        if len(self.features) == 0:
            return None, 0.0, {}
        
        best_score = -1
        best_feat = None
        best_info = {}
        
        for i, (feat, quality, timestamp) in enumerate(
            zip(self.features, self.qualities, self.timestamps)
        ):
            time_diff = current_timestamp - timestamp
            time_weight = np.exp(-0.1 * time_diff)
            
            spatial_sim = F.cosine_similarity(
                current_feat.flatten().unsqueeze(0),
                feat.flatten().unsqueeze(0)
            ).item()
            
            # === 新增：場景匹配度檢測 (v6.1 Edge-Based Logic) ===
            scene_match = 1.0
            
            # 使用邊緣特徵比對 (Edge vs Edge)
            # 這是最準確的方法：比較當前幀的邊緣 vs 記憶幀的邊緣
            if edge_feat is not None and i < len(self.edge_features):
                stored_edge = self.edge_features[i]
                edge_sim = F.cosine_similarity(
                    edge_feat.flatten().unsqueeze(0),
                    stored_edge.flatten().unsqueeze(0)
                ).item()
                scene_match = max(0.0, edge_sim)
                
            # 備用 1: 使用 last_valid_feat (如果有) - 較不準確，可能有過時風險
            elif self.last_valid_feat is not None:
                scene_sim = F.cosine_similarity(
                    self.last_valid_feat.flatten().unsqueeze(0),
                    feat.flatten().unsqueeze(0)
                ).item()
                scene_match = max(0.0, scene_sim)
            
            # 綜合分數（加入場景匹配權重）
            score = time_weight * 0.2 + spatial_sim * 0.2 + quality * 0.3 + scene_match * 0.3
            
            if score > best_score:
                best_score = score
                best_feat = feat
                best_info = {
                    'timestamp': timestamp,
                    'quality': quality,
                    'time_diff': time_diff,
                    'spatial_sim': spatial_sim,
                    'scene_match': scene_match,
                    'adapter_meta': self.adapter_metas[i] if i < len(self.adapter_metas) else None
                }
        
        return best_feat, best_score, best_info
    
    def compute_injection_strength(self, anomaly_score, memory_similarity, image_occlusion=0.0, scene_match=1.0, memory_reliability=1.0):
        """
        動態計算注入強度
        
        Args:
            anomaly_score: 綜合異常分數 (0-1)
            memory_similarity: 記憶與當前的相似度 (0-1)
            image_occlusion: 圖像遮擋程度 (0-1)
            scene_match: 場景匹配度 (0-1)，越低表示場景可能已變化
            memory_reliability: 來自 Adapter 的記憶可靠度 (0-1)
        """
        # 基礎強度
        base_strength = 0.4
        
        # === 場景匹配度門控 ===
        # 如果場景匹配度低，大幅降低注入強度
        if scene_match < 0.5:
            # 場景明顯變化，幾乎不注入
            return 0.10
        elif scene_match < 0.7:
            # 場景可能變化，保守注入
            base_strength = 0.25
        elif scene_match < 0.85:
            # 場景略有變化
            base_strength = 0.35
        
        # === 根據圖像遮擋程度調整 ===
        if image_occlusion > 0.70:
            occlusion_boost = 0.15
        elif image_occlusion > 0.50:
            occlusion_boost = 0.10
        elif image_occlusion > 0.30:
            occlusion_boost = 0.05
        else:
            occlusion_boost = 0.0
        
        # 異常越高，注入越強
        anomaly_factor = 0.7 + anomaly_score * 0.5
        
        # 記憶品質因子
        safety_factor = max(0.7, memory_similarity)
        
        # 場景匹配度作為額外乘數
        scene_factor = 0.5 + 0.5 * scene_match  # 範圍 0.5-1.0
        
        # 來自 Adapter 的可靠度（GRU 版本輸出 memory_quality）
        reliability = max(0.0, min(1.0, memory_reliability))
        reliability_factor = 0.4 + 0.6 * reliability  # 品質越低，注入越弱
        
        final_strength = (base_strength + occlusion_boost) * anomaly_factor * safety_factor * scene_factor * reliability_factor
        
        # 限制範圍
        return max(0.10, min(0.55, final_strength))
    
    def get_status(self):
        """返回記憶庫狀態"""
        status = {
            'size': len(self.features),
            'max_size': self.max_size,
            'avg_quality': np.mean(self.qualities) if self.qualities else 0,
            'timestamps': list(self.timestamps),
        }
        
        # Z-Score 統計
        if len(self.anomaly_score_history) > 0:
            status['zscore_stats'] = {
                'history_size': len(self.anomaly_score_history),
                'anomaly_mu': float(np.mean(self.anomaly_score_history)),
                'anomaly_sigma': float(np.std(self.anomaly_score_history)),
                'dynamic_threshold': float(np.mean(self.anomaly_score_history) + 
                                          self.z_score_threshold * np.std(self.anomaly_score_history))
            }
        
        return status
    
    def clear(self):
        """清空記憶庫和歷史統計"""
        self.features = []
        self.edge_features = []
        self.qualities = []
        self.timestamps = []
        self.images = []
        self.feature_mean = None
        self.feature_std = None
        self.prev_feat = None
        self.last_valid_feat = None
        
        # 清空圖像亮度歷史
        self.image_brightness_history = []
        
        # 清空 Z-Score 歷史
        self.anomaly_score_history.clear()
        self.feature_anomaly_history.clear()
        self.image_anomaly_history.clear()
