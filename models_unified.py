

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple


class UnifiedTempoVLM(nn.Module):
    """
    Unified TempoVLM multi-task model with GRU Long-term Memory

    Tasks:
    - temporal: 時序一致性 (使用 GRU 長期記憶)
    - depth_order: 深度排序 (A vs B 誰更近)
    - depth_regression: 相對深度值預測
    - motion: 相機運動預測 (6DoF)
    
    GRU 記憶功能:
    - 維護長期隱藏狀態，即使連續多幀被遮擋也能保留之前的資訊
    - 自動學習何時更新/遺忘記憶
    """
    
    def __init__(
        self,
        feat_dim: int = 1536,
        hidden_dim: int = 768,
        num_scene_classes: int = 20,
        dropout: float = 0.1,
        use_gru_memory: bool = True,  # 是否使用 GRU 記憶
    ):
        super().__init__()
        
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.use_gru_memory = use_gru_memory
        
        # ============================================================
        # shared encoder
        # ============================================================
        self.shared_encoder = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # ============================================================
        # GRU Long-term Memory (NEW)
        # ============================================================
        if use_gru_memory:
            # GRU Cell: 輸入當前觀測，輸出更新後的記憶
            self.temporal_gru = nn.GRUCell(hidden_dim, hidden_dim)
            
            # 記憶品質評估器：評估當前幀是否可信（用於決定是否更新記憶）
            self.memory_quality_gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
            
            # 記憶融合門：決定輸出時使用多少記憶 vs 當前觀測
            self.memory_output_gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid()
            )
        
        # ============================================================
        # temporal consistency branch (保留原有結構作為備用)
        # ============================================================
        self.temporal_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        self.temporal_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        self.temporal_output = nn.Sequential(
            nn.Linear(hidden_dim, feat_dim),
        )
        
        # ============================================================
        # depth order branch
        # ============================================================
        self.depth_order_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),  # [A較近, B較近]
        )
        
        # ============================================================
        # depth regression branch (predict absolute depth values for 3 regions)
        # ============================================================
        self.depth_regression_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 3),  # 輸出 3 個區域: [left, center, right]
        )
        # 最大深度範圍 (用於 sigmoid 映射)
        self.max_depth = 10.0  # 10 meters
        
        # ============================================================
        # camera motion prediction branch
        # ============================================================
        self.motion_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        self.motion_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 6),  # [tx, ty, tz, rx, ry, rz]
        )
        # 運動尺度因子 (可學習的參數) - 分離平移和旋轉的 scale
        # 初始化接近 ScanNet 的典型運動範圍：平移 ~0.01-0.1m，旋轉 ~0.01-0.1rad
        self.motion_scale = nn.Parameter(torch.tensor([0.05, 0.05, 0.05, 0.02, 0.02, 0.02]))
        
        # ============================================================
        # 軌跡累積誤差修正模組 (NEW)
        # ============================================================
        # 1. Motion Uncertainty Head - 預測每幀運動的不確定性
        self.motion_uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 6),  # 每個維度的 log variance
        )
        
        # 2. Velocity Consistency - 用於平滑軌跡
        self.velocity_smoothing = nn.Sequential(
            nn.Linear(hidden_dim + 6, hidden_dim // 2),  # 當前特徵 + 前一幀運動
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 6),  # 修正項
        )
        
        # 3. Global Scale Predictor - 預測全局尺度因子（解決 scale 不一致問題）
        self.global_scale_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Softplus(),  # 確保 scale > 0
        )
        
        # 4. Motion Quality Detector - 檢測快速運動/模糊幀
        self.motion_quality_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),  # 0 = 低品質, 1 = 高品質
        )
        
        # 5. Place Recognition - 簡化版 Loop Closure（檢測是否回到相似位置）
        self.place_embedding = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
        )
       
        self.scene_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_scene_classes),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GRUCell):
                # GRU 特殊初始化
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def init_hidden_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """初始化 GRU 隱藏狀態"""
        return torch.zeros(batch_size, self.hidden_dim, device=device)
    
    def load_pretrained_temporal(self, checkpoint_path: str, strict: bool = False):

        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        print(f"📦 原始 checkpoint 包含的 keys:")
        for k, v in state_dict.items():
            print(f"   {k}: {v.shape}")
        
        compatible_keys = []
        incompatible_keys = []
        
        
        if 'gate.0.weight' in state_dict:
            old_weight = state_dict['gate.0.weight']  # [768, 3072]
            new_weight_shape = self.temporal_gate[0].weight.shape  # [768, 1536]
            
            if old_weight.shape[0] == new_weight_shape[0]:
            
                self.temporal_gate[0].weight.data = old_weight[:, :new_weight_shape[1]].clone()
                if 'gate.0.bias' in state_dict:
                    self.temporal_gate[0].bias.data = state_dict['gate.0.bias'].clone()
                compatible_keys.append('gate.0 (partial)')
        
        if 'refine.0.weight' in state_dict:
            old_shape = state_dict['refine.0.weight'].shape
            new_shape = self.temporal_output[0].weight.shape
            
            if old_shape == new_shape:
                self.temporal_output[0].weight.data = state_dict['refine.0.weight'].clone()
                self.temporal_output[0].bias.data = state_dict['refine.0.bias'].clone()
                compatible_keys.append('refine.0')
        
        print(f"\n✅ 預訓練權重載入結果:")
        print(f"   - 部分相容: {compatible_keys}")
        print(f"   - 結構不同，需重新訓練: shared_encoder, temporal_fusion")
        print(f"   ⚠️ 由於架構差異，建議重新訓練或使用 --no_pretrained")
        
        return compatible_keys
    
    def forward(
        self,
        curr_feat: torch.Tensor,
        prev_feat: Optional[torch.Tensor] = None,
        hidden_state: Optional[torch.Tensor] = None,  # GRU 隱藏狀態 (NEW)
        region_a_feat: Optional[torch.Tensor] = None,
        region_b_feat: Optional[torch.Tensor] = None,
        tasks: List[str] = ['temporal'],
    ) -> Tuple[Dict[str, torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass with optional GRU memory
        
        Args:
            curr_feat: 當前幀特徵 [B, feat_dim]
            prev_feat: 前一幀特徵 [B, feat_dim] (temporal/motion 用)
            hidden_state: GRU 隱藏狀態 [B, hidden_dim] (長期記憶)
            region_a_feat: 區域 A 特徵 [B, feat_dim] (depth_order 用)
            region_b_feat: 區域 B 特徵 [B, feat_dim] (depth_order 用)
            tasks: 要執行的任務列表
        
        Returns:
            outputs: 包含各任務輸出的字典
            next_hidden_state: 更新後的 GRU 隱藏狀態 (如果使用 GRU)
        """
        outputs = {}
        next_hidden_state = None
        
        # 編碼當前幀
        curr_enc = self.shared_encoder(curr_feat)  # [B, hidden_dim]
        batch_size = curr_feat.shape[0]
        device = curr_feat.device
        
        # ============================================================
        # Task 1 : temporal consistency (with GRU Long-term Memory)
        # ============================================================
        if 'temporal' in tasks:
            if self.use_gru_memory:
                # ========== GRU 長期記憶模式 ==========
                
                # 初始化隱藏狀態（如果是第一幀或新場景）
                if hidden_state is None:
                    hidden_state = self.init_hidden_state(batch_size, device)
                
                # 1. 評估當前幀的品質（是否被遮擋）
                #    比較當前觀測和長期記憶的差異
                combined_for_quality = torch.cat([curr_enc, hidden_state], dim=-1)
                quality_score = self.memory_quality_gate(combined_for_quality)  # [B, 1]
                
                # 2. GRU 更新記憶
                #    quality_score 高 = 當前幀可信，多更新記憶
                #    quality_score 低 = 當前幀可能被遮擋，少更新記憶
                gru_input = curr_enc * quality_score + hidden_state * (1 - quality_score)
                new_memory = self.temporal_gru(gru_input, hidden_state)
                
                # 3. 決定輸出時使用多少記憶
                combined_for_output = torch.cat([curr_enc, new_memory], dim=-1)
                output_gate = self.memory_output_gate(combined_for_output)  # [B, hidden_dim]
                
                # 4. 融合當前觀測和長期記憶
                fused_enc = output_gate * new_memory + (1 - output_gate) * curr_enc
                
                # 5. 輸出精煉後的特徵
                refined = self.temporal_output(fused_enc)
                
                # 6. 殘差連接
                outputs['temporal'] = curr_feat + refined
                outputs['temporal_gate'] = output_gate.mean()
                outputs['memory_quality'] = quality_score.mean()  # 用於監控
                
                # 更新隱藏狀態
                next_hidden_state = new_memory
                
            elif prev_feat is not None:
                # ========== 原始模式（無 GRU）==========
                prev_enc = self.shared_encoder(prev_feat)
                
                # fusion
                combined = torch.cat([curr_enc, prev_enc], dim=-1)
                fused = self.temporal_fusion(combined)
                
                # gate
                gate = self.temporal_gate(combined)
                gated = fused * gate + curr_enc * (1 - gate)

                # output refined features
                refined = self.temporal_output(gated)
                
                # residual connection
                outputs['temporal'] = curr_feat + refined
                outputs['temporal_gate'] = gate.mean() 
        
        # ============================================================
        # Task 2 : depth order
        # ============================================================
        if 'depth_order' in tasks:
            if region_a_feat is not None and region_b_feat is not None:
                region_a_enc = self.shared_encoder(region_a_feat)
                region_b_enc = self.shared_encoder(region_b_feat)
                combined = torch.cat([region_a_enc, region_b_enc], dim=-1)
                outputs['depth_order'] = self.depth_order_head(combined)  # [B, 2]
            else:
                # 使用全圖特徵的不同區域（簡化版）
                outputs['depth_order'] = None
        
        # ============================================================
        # Task 3 : depth regression (輸出 3 個區域的絕對深度)
        # ============================================================
        if 'depth_regression' in tasks:
            raw_depth = self.depth_regression_head(curr_enc)  # [B, 3]
            # 使用 softplus 確保輸出為正數，並限制在合理範圍內
            # softplus(x) = log(1 + exp(x))，平滑的 ReLU
            depth = F.softplus(raw_depth) * (self.max_depth / 5.0)  # scale to ~0-10m range
            # 或者用 sigmoid: depth = torch.sigmoid(raw_depth) * self.max_depth
            outputs['depth_regression'] = depth  # [B, 3] = [left, center, right]
        
        # ============================================================
        # Task 4 : camera motion prediction (with quality & scale correction)
        # ============================================================
        if 'motion' in tasks and prev_feat is not None:
            prev_enc = self.shared_encoder(prev_feat)
            combined = torch.cat([curr_enc, prev_enc], dim=-1)
            fused = self.motion_fusion(combined)
            raw_motion = self.motion_head(fused)  # [B, 6]
            
            # 1. 基礎運動預測 + scale 參數
            motion = raw_motion * self.motion_scale.unsqueeze(0)  # [B, 6]
            
            # 2. 預測運動不確定性 (用於加權 loss)
            motion_log_var = self.motion_uncertainty_head(fused)  # [B, 6]
            motion_uncertainty = torch.exp(motion_log_var)  # [B, 6]
            
            # 3. 預測全局 scale factor (用於校正累積誤差)
            global_scale = self.global_scale_head(curr_enc)  # [B, 1]
            # 將 global_scale 限制在合理範圍 [0.5, 2.0]
            global_scale = 0.5 + 1.5 * torch.sigmoid(global_scale - 1)
            
            # 4. 檢測運動品質（快速運動/模糊檢測）
            motion_quality = self.motion_quality_head(combined)  # [B, 1]
            
            # 5. Place Recognition embedding (用於 Loop Closure)
            place_emb = self.place_embedding(curr_enc)  # [B, hidden_dim//2]
            
            outputs['motion'] = motion
            outputs['motion_raw'] = raw_motion  # 原始預測（用於分析）
            outputs['motion_uncertainty'] = motion_uncertainty
            outputs['motion_log_var'] = motion_log_var
            outputs['motion_global_scale'] = global_scale
            outputs['motion_quality'] = motion_quality
            outputs['place_embedding'] = place_emb
        
        if 'scene_class' in tasks:
            outputs['scene_class'] = self.scene_classifier(curr_enc)  # [B, num_classes]
        
        # 返回 outputs 和 next_hidden_state（如果使用 GRU 記憶）
        if self.use_gru_memory and 'temporal' in tasks:
            return outputs, next_hidden_state
        else:
            return outputs, None
    
    def forward_temporal(
        self, 
        curr_feat: torch.Tensor, 
        prev_feat: torch.Tensor = None,
        hidden_state: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """便利方法：只執行 temporal 任務"""
        outputs, next_hidden = self.forward(
            curr_feat, prev_feat, 
            hidden_state=hidden_state,
            tasks=['temporal']
        )
        return outputs['temporal'], next_hidden
    
    def forward_depth_order(
        self, 
        region_a_feat: torch.Tensor, 
        region_b_feat: torch.Tensor
    ) -> torch.Tensor:
        outputs, _ = self.forward(
            region_a_feat, 
            region_a_feat=region_a_feat,
            region_b_feat=region_b_feat,
            tasks=['depth_order']
        )
        return outputs['depth_order']
    
    def forward_motion(
        self, 
        curr_feat: torch.Tensor, 
        prev_feat: torch.Tensor
    ) -> torch.Tensor:
        outputs, _ = self.forward(curr_feat, prev_feat, tasks=['motion'])
        return outputs['motion']


class UnifiedLoss(nn.Module):
    """
    Unified multi-task loss for UnifiedTempoVLM
    使用 Kendall et al. (CVPR 2018) 的自動加權方法
    """
    def __init__(
        self,
        num_tasks: int = 5,
        use_uncertainty_weighting: bool = True,
    ):
        super().__init__()
        
        self.use_uncertainty_weighting = use_uncertainty_weighting
        
        # 可學習的 log variance 參數 (用於自動 Loss 平衡)
        # 任務順序: [temporal, depth_order, depth_regression, motion, scene_class]
        if use_uncertainty_weighting:
            self.log_vars = nn.Parameter(torch.zeros(num_tasks))
        else:
            # 固定權重 (備用)
            self.register_buffer('fixed_weights', torch.ones(num_tasks))
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
    
    def _weighted_loss(self, loss, task_idx):
        """
        根據不確定性自動加權 loss
        公式: L_weighted = L / (2 * σ²) + log(σ) = L * exp(-log_var) / 2 + log_var / 2
        """
        if self.use_uncertainty_weighting:
            # Clamp log_vars 避免數值不穩定
            log_var = torch.clamp(self.log_vars[task_idx], min=-4, max=4)
            precision = torch.exp(-log_var)
            return precision * loss + 0.5 * log_var
        else:
            return loss * self.fixed_weights[task_idx]
    
    def scale_invariant_depth_loss(self, pred, target):
        """
        Scale-Invariant Loss for depth prediction
        對深度預測更穩定，不受絕對尺度影響
        """
        # 創建有效深度的 mask（排除無效的 0 值區域）
        valid_mask = target > 0.1  # 只考慮深度 > 0.1m 的區域
        
        if valid_mask.sum() == 0:
            # 沒有有效深度，返回 0 loss
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        # 只計算有效區域
        pred_valid = pred[valid_mask].clamp(min=1e-6)
        target_valid = target[valid_mask].clamp(min=1e-6)
        
        # 對數空間的差異
        log_diff = torch.log(pred_valid) - torch.log(target_valid)
        
        # Scale-invariant loss
        n = log_diff.numel()
        if n == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
            
        si_loss = torch.sum(log_diff ** 2) / n - (torch.sum(log_diff) ** 2) / (n ** 2)
        
        # 加上 L1 loss 作為正則化
        l1_loss = F.l1_loss(pred_valid, target_valid)
        
        return si_loss + 0.5 * l1_loss
    
    def motion_loss(self, pred, target, log_var=None):
        """
        運動預測的 loss，分別處理平移和旋轉
        支援不確定性加權 (Learned Uncertainty)
        
        Args:
            pred: 預測運動 [B, 6]
            target: GT 運動 [B, 6]
            log_var: 預測的 log variance [B, 6] (可選)
        """
        # 分離平移 [tx, ty, tz] 和旋轉 [rx, ry, rz]
        pred_trans = pred[:, :3]
        pred_rot = pred[:, 3:]
        target_trans = target[:, :3]
        target_rot = target[:, 3:]
        
        if log_var is not None:
            # 方案：使用軟性不確定性加權（不會產生負 loss）
            # 使用 softplus 確保 variance > 0
            # L = |pred - target|^2 / (2 * variance) + 0.5 * log(variance)
            # 其中 variance = softplus(log_var) + 1e-6
            
            # 將 log_var 轉為正的 variance
            variance = F.softplus(log_var) + 1e-6  # [B, 6], 確保 > 0
            
            var_trans = variance[:, :3]
            var_rot = variance[:, 3:]
            
            # 平移 loss（誤差 / variance + log(variance)）
            trans_error = (pred_trans - target_trans) ** 2
            trans_loss = (trans_error / (2 * var_trans)).mean() + 0.5 * torch.log(var_trans).mean()
            
            # 旋轉 loss
            rot_error = (pred_rot - target_rot) ** 2
            rot_loss = (rot_error / (2 * var_rot)).mean() + 0.5 * torch.log(var_rot).mean()
            
            # 加一個 baseline loss 確保有梯度
            baseline_trans = F.smooth_l1_loss(pred_trans, target_trans)
            baseline_rot = F.mse_loss(pred_rot, target_rot)
            
            # 混合：50% 不確定性加權 + 50% baseline
            trans_loss = 0.5 * trans_loss + 0.5 * baseline_trans
            rot_loss = 0.5 * rot_loss + 0.5 * baseline_rot
            
        else:
            # 原始 loss
            trans_loss = F.smooth_l1_loss(pred_trans, target_trans)
            rot_loss = F.mse_loss(pred_rot, target_rot)
        
        # 加入速度一致性正則化
        # 鼓勵相鄰預測的變化平滑
        if pred.shape[0] > 1:
            velocity_diff = pred[1:] - pred[:-1]
            smoothness_loss = 0.01 * (velocity_diff ** 2).mean()
        else:
            smoothness_loss = 0
        
        return trans_loss + rot_loss + smoothness_loss
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        prev_feat: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        calculate unified multi-task loss with automatic weighting
        
        Args:
            outputs: model output dictionary
            targets: target dictionary
            prev_feat: previous frame features (for temporal consistency)
        
        Returns:
            total_loss: total loss
            loss_dict: individual task loss dictionary (包含原始 loss 和權重)
        """
        total_loss = 0
        loss_dict = {}
        
        # Task 0: temporal consistency loss
        if 'temporal' in outputs and prev_feat is not None:
            refined = outputs['temporal']
            # consistency with previous frame
            temporal_loss = 1 - F.cosine_similarity(
                refined.float(), prev_feat.float(), dim=-1
            ).mean()
            weighted_loss = self._weighted_loss(temporal_loss, task_idx=0)
            total_loss += weighted_loss
            loss_dict['temporal'] = temporal_loss.item()
            if self.use_uncertainty_weighting:
                loss_dict['temporal_weight'] = torch.exp(-self.log_vars[0]).item()
        
        # Task 1: depth order loss
        if 'depth_order' in outputs and outputs['depth_order'] is not None:
            if 'depth_order' in targets:
                depth_order_loss = self.ce_loss(
                    outputs['depth_order'],
                    targets['depth_order']
                )
                weighted_loss = self._weighted_loss(depth_order_loss, task_idx=1)
                total_loss += weighted_loss
                loss_dict['depth_order'] = depth_order_loss.item()
                if self.use_uncertainty_weighting:
                    loss_dict['depth_order_weight'] = torch.exp(-self.log_vars[1]).item()
        
        # Task 2: depth regression loss (使用 Scale-Invariant Loss)
        if 'depth_regression' in outputs and outputs['depth_regression'] is not None:
            if 'depth_regression' in targets:
                pred_depth = outputs['depth_regression']  # [B, 3]
                target_depth = targets['depth_regression']  # [B, 3] or [B, 1]
                
                # 如果 target 只有 1 維，擴展到 3 維
                if target_depth.dim() == 1:
                    target_depth = target_depth.unsqueeze(-1).expand(-1, 3)
                elif target_depth.shape[-1] == 1:
                    target_depth = target_depth.expand(-1, 3)
                
                depth_reg_loss = self.scale_invariant_depth_loss(
                    pred_depth, target_depth
                )
                weighted_loss = self._weighted_loss(depth_reg_loss, task_idx=2)
                total_loss += weighted_loss
                loss_dict['depth_regression'] = depth_reg_loss.item()
                if self.use_uncertainty_weighting:
                    loss_dict['depth_regression_weight'] = torch.exp(-self.log_vars[2]).item()
        
        # Task 3: motion prediction loss (分離平移和旋轉，支援不確定性)
        if 'motion' in outputs and 'motion' in targets:
            # 使用預測的不確定性（如果有的話）
            motion_log_var = outputs.get('motion_log_var', None)
            
            motion_loss = self.motion_loss(
                outputs['motion'],
                targets['motion'],
                log_var=motion_log_var
            )
            weighted_loss = self._weighted_loss(motion_loss, task_idx=3)
            total_loss += weighted_loss
            loss_dict['motion'] = motion_loss.item()
            if self.use_uncertainty_weighting:
                loss_dict['motion_weight'] = torch.exp(-self.log_vars[3]).item()
            
            # 記錄平均不確定性（用於監控）
            if motion_log_var is not None:
                loss_dict['motion_avg_uncertainty'] = torch.exp(motion_log_var).mean().item()
            
            # Motion Quality 監督（如果有標籤）
            if 'motion_quality' in outputs and 'motion_quality_label' in targets:
                quality_loss = F.binary_cross_entropy(
                    outputs['motion_quality'],
                    targets['motion_quality_label']
                )
                total_loss += 0.1 * quality_loss  # 小權重
                loss_dict['motion_quality'] = quality_loss.item()
            
            # Global Scale 監督（如果有標籤）
            if 'motion_global_scale' in outputs and 'motion_scale_label' in targets:
                scale_loss = F.mse_loss(
                    outputs['motion_global_scale'],
                    targets['motion_scale_label']
                )
                total_loss += 0.1 * scale_loss
                loss_dict['scale'] = scale_loss.item()
        
        # Task 4: scene classification loss
        if 'scene_class' in outputs and 'scene_class' in targets:
            scene_loss = self.ce_loss(
                outputs['scene_class'],
                targets['scene_class']
            )
            weighted_loss = self._weighted_loss(scene_loss, task_idx=4)
            total_loss += weighted_loss
            loss_dict['scene_class'] = scene_loss.item()
            if self.use_uncertainty_weighting:
                loss_dict['scene_class_weight'] = torch.exp(-self.log_vars[4]).item()
        
        return total_loss, loss_dict
    
    def get_task_weights(self) -> Dict[str, float]:
        """取得當前各任務的自動權重 (用於監控)"""
        if self.use_uncertainty_weighting:
            task_names = ['temporal', 'depth_order', 'depth_regression', 'motion', 'scene_class']
            weights = torch.exp(-self.log_vars).detach().cpu().numpy()
            return {name: float(w) for name, w in zip(task_names, weights)}
        else:
            return {}


# ============================================================
# tools
# ============================================================

def create_unified_model(
    feat_dim: int = 1536,
    pretrained_temporal_path: Optional[str] = None,
) -> UnifiedTempoVLM:

    model = UnifiedTempoVLM(feat_dim=feat_dim)
    
    if pretrained_temporal_path:
        model.load_pretrained_temporal(pretrained_temporal_path)
    
    return model


def get_model_info(model: UnifiedTempoVLM) -> Dict:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 各分支參數量
    branch_params = {
        'shared_encoder': sum(p.numel() for p in model.shared_encoder.parameters()),
        'temporal': sum(p.numel() for n, p in model.named_parameters() if 'temporal' in n and 'gru' not in n.lower()),
        'depth_order': sum(p.numel() for p in model.depth_order_head.parameters()),
        'depth_regression': sum(p.numel() for p in model.depth_regression_head.parameters()),
        'motion': sum(p.numel() for n, p in model.named_parameters() if 'motion' in n),
        'scene_classifier': sum(p.numel() for p in model.scene_classifier.parameters()),
    }
    
    # GRU 記憶相關參數
    if model.use_gru_memory:
        gru_params = sum(p.numel() for p in model.temporal_gru.parameters())
        memory_gate_params = sum(p.numel() for p in model.memory_quality_gate.parameters())
        memory_gate_params += sum(p.numel() for p in model.memory_output_gate.parameters())
        branch_params['gru_memory'] = gru_params + memory_gate_params
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'branch_params': branch_params,
    }


# ============================================================
# Testing
# ============================================================

if __name__ == "__main__":
    print("Testing UnifiedTempoVLM...")

    # 測試無 GRU 模式
    print("\n========== 測試基本模式（無 GRU）==========")
    model = UnifiedTempoVLM(feat_dim=1536, hidden_dim=768, use_gru_memory=False)
    model.eval()
    
    batch_size = 2
    curr_feat = torch.randn(batch_size, 1536)
    prev_feat = torch.randn(batch_size, 1536)
    region_a = torch.randn(batch_size, 1536)
    region_b = torch.randn(batch_size, 1536)
    
    print("\nTesting multi-task forward propagation...")
    outputs, hidden = model(
        curr_feat=curr_feat,
        prev_feat=prev_feat,
        region_a_feat=region_a,
        region_b_feat=region_b,
        tasks=['temporal', 'depth_order', 'depth_regression', 'motion', 'scene_class']
    )
    
    print(f"  temporal output: {outputs['temporal'].shape}")
    print(f"  depth_order output: {outputs['depth_order'].shape}")
    print(f"  depth_regression output: {outputs['depth_regression'].shape}")  # 應該是 [B, 3]
    print(f"  depth_regression values: {outputs['depth_regression']}")  # 檢查是否為正數
    print(f"  motion output: {outputs['motion'].shape}")
    print(f"  motion values: {outputs['motion']}")  # 檢查運動值
    print(f"  scene_class output: {outputs['scene_class'].shape}")
    print(f"  next_hidden_state: {hidden}")  # 應該是 None（無 GRU 模式）

    # 測試 GRU 記憶模式
    print("\n========== 測試 GRU 記憶模式 ==========")
    model_gru = UnifiedTempoVLM(feat_dim=1536, hidden_dim=768, use_gru_memory=True)
    model_gru.eval()
    
    print("\n模擬連續幀處理...")
    hidden_state = None
    for frame_idx in range(5):
        curr_feat = torch.randn(batch_size, 1536)
        outputs, hidden_state = model_gru(
            curr_feat=curr_feat,
            hidden_state=hidden_state,
            tasks=['temporal']
        )
        print(f"  Frame {frame_idx}: temporal_gate={outputs.get('temporal_gate', 'N/A'):.3f}, "
              f"memory_quality={outputs.get('memory_quality', 'N/A'):.3f}, "
              f"hidden_state shape={hidden_state.shape if hidden_state is not None else 'None'}")

    print("\nTesting loss calculation with automatic weighting...")
    loss_fn = UnifiedLoss(use_uncertainty_weighting=True)
    targets = {
        'depth_order': torch.randint(0, 2, (batch_size,)),
        'depth_regression': torch.rand(batch_size, 3) * 5 + 0.5,  # 0.5~5.5m 的深度
        'motion': torch.randn(batch_size, 6) * 0.1,  # 小的運動值
        'scene_class': torch.randint(0, 20, (batch_size,)),
    }
    
    # 使用無 GRU 模型的 outputs
    outputs, _ = model(
        curr_feat=torch.randn(batch_size, 1536),
        prev_feat=torch.randn(batch_size, 1536),
        region_a_feat=region_a,
        region_b_feat=region_b,
        tasks=['temporal', 'depth_order', 'depth_regression', 'motion', 'scene_class']
    )
    
    total_loss, loss_dict = loss_fn(outputs, targets, prev_feat)
    print(f"  Total loss: {total_loss.item():.4f}")
    print(f"  Individual losses:")
    for k, v in loss_dict.items():
        if not k.endswith('_weight'):
            print(f"    {k}: {v:.4f}")
    
    print(f"\n  Auto-learned task weights (初始應該都接近 1.0):")
    weights = loss_fn.get_task_weights()
    for task, weight in weights.items():
        print(f"    {task}: {weight:.4f}")
    
    print(f"\n  Log variance parameters:")
    for i, name in enumerate(['temporal', 'depth_order', 'depth_regression', 'motion', 'scene_class']):
        print(f"    {name}: log_var = {loss_fn.log_vars[i].item():.4f}")

    print("\nModel Information:")
    info = get_model_info(model)
    print(f"  Total Parameters: {info['total_params']:,}")
    print(f"  Branch Parameters:")
    for branch, params in info['branch_params'].items():
        print(f"    {branch}: {params:,}")
    
    info_gru = get_model_info(model_gru)
    print(f"\n  GRU Model Total Parameters: {info_gru['total_params']:,}")
    print(f"  GRU Branch Parameters:")
    for branch, params in info_gru['branch_params'].items():
        print(f"    {branch}: {params:,}")
    
    # 測試 Loss 函數的參數量
    loss_params = sum(p.numel() for p in loss_fn.parameters())
    print(f"\n  Loss function learnable params: {loss_params}")

    print("\n✅ Testing completed!")
