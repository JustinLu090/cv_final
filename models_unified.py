

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple


class UnifiedTempoVLM(nn.Module):
    """
    Unified TempoVLM multi-task model with GRU Long-term Memory

    Tasks:
    - temporal: 時序一致性 (使用 GRU 長期記憶)
    - depth_order: 深度排序 (A vs B 誰更近) - 使用 GT 深度標籤訓練
    - depth_regression: 相對深度值預測 - 使用 GT 深度標籤訓練
    - motion: 相機運動預測 (6DoF)
    
    GRU 記憶功能:
    - 維護長期隱藏狀態，即使連續多幀被遮擋也能保留之前的資訊
    - 自動學習何時更新/遺忘記憶
    
    Transformer Encoder:
    - 使用多層 Transformer 替代簡單 Linear，提升特徵表達能力
    - Pre-LN 架構確保訓練穩定性
    """
    
    def __init__(
        self,
        feat_dim: int = 1536,
        hidden_dim: int = 768,
        num_scene_classes: int = 20,
        dropout: float = 0.1,
        use_gru_memory: bool = True,  # 是否使用 GRU 記憶
        use_transformer_encoder: bool = True,  # 是否使用 Transformer Encoder
        num_encoder_layers: int = 2,  # Transformer 層數
        num_heads: int = 8,  # Attention head 數量
    ):
        super().__init__()
        
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.use_gru_memory = use_gru_memory
        self.use_transformer_encoder = use_transformer_encoder
        
        # ============================================================
        # shared encoder (Transformer 或 簡單 MLP)
        # ============================================================
        if use_transformer_encoder:
            # 使用 Transformer Encoder（更強的特徵提取）
            self.input_proj = nn.Linear(feat_dim, hidden_dim)
            
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True,  # Pre-LN，訓練更穩定
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer, 
                num_layers=num_encoder_layers
            )
            self.encoder_norm = nn.LayerNorm(hidden_dim)
            
            # 為了向後兼容，創建一個 wrapper
            def shared_encoder_forward(x):
                # x: [B, feat_dim]
                x = self.input_proj(x)  # [B, hidden_dim]
                x = x.unsqueeze(1)  # [B, 1, hidden_dim] - 加 sequence 維度
                x = self.transformer_encoder(x)  # [B, 1, hidden_dim]
                x = x.squeeze(1)  # [B, hidden_dim]
                return self.encoder_norm(x)
            
            # 包裝成 module（方便參數管理）
            class SharedEncoderWrapper(nn.Module):
                def __init__(self, forward_fn):
                    super().__init__()
                    self.forward_fn = forward_fn
                
                def forward(self, x):
                    return self.forward_fn(x)
            
            self.shared_encoder = SharedEncoderWrapper(shared_encoder_forward)
        else:
            # 原始簡單 MLP（向後兼容）
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
    
    重新設計的 Loss 平衡策略：
    1. 使用固定的 loss 尺度歸一化，確保每個任務貢獻均衡
    2. 對 InfoNCE loss 進行特殊處理（值很大）
    3. 手動設定任務優先級權重
    """
    def __init__(
        self,
        num_tasks: int = 5,
        use_uncertainty_weighting: bool = False,  # 預設關閉自動權重
        # 手動權重設定（經過調校）
        task_weights: Dict[str, float] = None,
    ):
        super().__init__()
        
        self.use_uncertainty_weighting = use_uncertainty_weighting
        
        # ============================================================
        # 手動設定的任務權重（經過分析調校）
        # ============================================================
        if task_weights is None:
            self.task_weights = {
                'temporal': 0.1,          # InfoNCE loss 太大，降低權重
                'depth_order': 1.0,       # 分類任務，保持標準權重
                'depth_regression': 3.0,  # 🔥 提高深度回歸權重
                'motion': 2.0,            # 🔥 提高運動預測權重
                'scene_class': 0.5,       # 輔助任務，降低權重
                'occlusion_recon': 1.5,   # 遮擋重建
                'memory_quality_reg': 0.5, # 記憶品質正則化
            }
        else:
            self.task_weights = task_weights
        
        # 可學習的 log variance 參數 (備用)
        if use_uncertainty_weighting:
            # 用更好的初始化：depth 和 motion 的初始權重更高
            init_log_vars = torch.tensor([2.0, 0.0, -1.0, -0.5, 0.5])  # 對應權重: [0.14, 1.0, 2.7, 1.6, 0.6]
            self.log_vars = nn.Parameter(init_log_vars)
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        
        # 用於 loss 尺度追蹤的 EMA
        self.register_buffer('loss_ema', torch.ones(num_tasks))
        self.ema_decay = 0.99
    
    def _get_weight(self, task_name: str, task_idx: int = None) -> float:
        """獲取任務權重"""
        if self.use_uncertainty_weighting and task_idx is not None:
            log_var = torch.clamp(self.log_vars[task_idx], min=-4, max=4)
            return torch.exp(-log_var)
        else:
            return self.task_weights.get(task_name, 1.0)
    
    def scale_invariant_depth_loss(self, pred, target):
        """
        改進的深度 Loss：
        1. Scale-Invariant Loss
        2. L1 Loss
        3. 梯度 Loss（鼓勵平滑）
        """
        valid_mask = target > 0.1
        
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        pred_valid = pred[valid_mask].clamp(min=1e-6)
        target_valid = target[valid_mask].clamp(min=1e-6)
        
        # 1. L1 Loss（主要）
        l1_loss = F.l1_loss(pred_valid, target_valid)
        
        # 2. Scale-Invariant Loss（輔助）
        log_diff = torch.log(pred_valid) - torch.log(target_valid)
        n = log_diff.numel()
        if n > 0:
            si_loss = torch.sum(log_diff ** 2) / n - 0.5 * (torch.sum(log_diff) ** 2) / (n ** 2)
        else:
            si_loss = torch.tensor(0.0, device=pred.device)
        
        # 3. 相對誤差 Loss（鼓勵比例正確）
        rel_loss = (torch.abs(pred_valid - target_valid) / (target_valid + 1e-6)).mean()
        
        return l1_loss + 0.5 * si_loss + 0.3 * rel_loss
    
    def motion_loss(self, pred, target, log_var=None):
        """
        簡化的運動 Loss：
        - 平移用 Smooth L1
        - 旋轉用 MSE
        """
        pred_trans = pred[:, :3]
        pred_rot = pred[:, 3:]
        target_trans = target[:, :3]
        target_rot = target[:, 3:]
        
        # 簡單直接的 loss，不用不確定性
        trans_loss = F.smooth_l1_loss(pred_trans, target_trans)
        rot_loss = F.mse_loss(pred_rot, target_rot)
        
        return trans_loss + rot_loss
    
    def temporal_contrastive_loss(self, refined_curr, prev_feat):
        """
        改進的時序對比 Loss：
        1. 使用更高的溫度參數（避免 loss 過大）
        2. 加入正樣本相似度約束
        """
        batch_size = refined_curr.shape[0]
        
        if batch_size <= 1:
            return 1 - F.cosine_similarity(refined_curr.float(), prev_feat.float(), dim=-1).mean(), {}, {}
        
        # 正則化特徵
        refined_norm = F.normalize(refined_curr, p=2, dim=-1)
        prev_norm = F.normalize(prev_feat, p=2, dim=-1)
        
        # 計算相似度矩陣
        sim_matrix = refined_norm @ prev_norm.T
        
        # 🔥 使用更高的溫度，避免 loss 過大
        tau = 0.1  # 從 0.02 提高到 0.1
        
        # InfoNCE Loss
        exp_sim = torch.exp(sim_matrix / tau)
        pos_exp = torch.diag(exp_sim)
        
        mask = torch.eye(batch_size, device=sim_matrix.device).bool()
        neg_exp_sum = exp_sim.masked_fill(mask, 0).sum(dim=1)
        
        infonce_loss = -torch.log(pos_exp / (pos_exp + neg_exp_sum + 1e-8)).mean()
        
        # 🔥 加入正樣本相似度約束：鼓勵正樣本相似度 > 0.8
        pos_sim = torch.diag(sim_matrix)
        pos_sim_loss = F.relu(0.8 - pos_sim).mean()  # 如果相似度 < 0.8，有懲罰
        
        # 組合 loss（控制 InfoNCE 的影響）
        # InfoNCE 通常在 2-5 之間，我們希望總 loss 在 0.5-2 之間
        total_loss = 0.3 * infonce_loss + 0.7 * pos_sim_loss
        
        # 診斷信息
        with torch.no_grad():
            neg_sim = sim_matrix.masked_fill(mask, 0).sum() / (batch_size * (batch_size - 1))
        
        return total_loss, {'pos_sim': pos_sim.mean().item(), 'neg_sim': neg_sim.item()}, {'infonce': infonce_loss.item()}
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        prev_feat: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        計算多任務 Loss
        
        改進：
        1. 每個 loss 都有明確的尺度範圍
        2. 手動設定的權重確保平衡
        3. 詳細的診斷輸出
        """
        total_loss = torch.tensor(0.0, device=next(iter(outputs.values())).device)
        loss_dict = {}
        
        # ============================================================
        # Task 0: Temporal Consistency (對比學習)
        # 目標範圍: 0.3 - 1.0
        # ============================================================
        if 'temporal' in outputs and prev_feat is not None:
            temporal_loss, diag_info, raw_losses = self.temporal_contrastive_loss(
                outputs['temporal'], prev_feat
            )
            
            weight = self._get_weight('temporal', 0)
            total_loss = total_loss + weight * temporal_loss
            
            loss_dict['temporal'] = temporal_loss.item()
            loss_dict['temporal_weight'] = weight if isinstance(weight, float) else weight.item()
            if diag_info:
                loss_dict['temporal_pos_sim'] = diag_info['pos_sim']
                loss_dict['temporal_neg_sim'] = diag_info['neg_sim']
            if raw_losses:
                loss_dict['temporal_infonce_raw'] = raw_losses['infonce']
        
        # ============================================================
        # Task 1: Depth Order (分類)
        # 目標範圍: 0.3 - 1.0
        # ============================================================
        if 'depth_order' in outputs and outputs['depth_order'] is not None:
            if 'depth_order' in targets:
                depth_order_loss = self.ce_loss(outputs['depth_order'], targets['depth_order'])
                
                weight = self._get_weight('depth_order', 1)
                total_loss = total_loss + weight * depth_order_loss
                
                loss_dict['depth_order'] = depth_order_loss.item()
                loss_dict['depth_order_weight'] = weight if isinstance(weight, float) else weight.item()
        
        # ============================================================
        # Task 2: Depth Regression (回歸) 🔥 重要任務
        # 目標範圍: 0.1 - 0.5
        # ============================================================
        if 'depth_regression' in outputs and outputs['depth_regression'] is not None:
            if 'depth_regression' in targets:
                pred_depth = outputs['depth_regression']
                target_depth = targets['depth_regression']
                
                if target_depth.dim() == 1:
                    target_depth = target_depth.unsqueeze(-1).expand(-1, 3)
                elif target_depth.shape[-1] == 1:
                    target_depth = target_depth.expand(-1, 3)
                
                depth_reg_loss = self.scale_invariant_depth_loss(pred_depth, target_depth)
                
                weight = self._get_weight('depth_regression', 2)
                total_loss = total_loss + weight * depth_reg_loss
                
                loss_dict['depth_regression'] = depth_reg_loss.item()
                loss_dict['depth_regression_weight'] = weight if isinstance(weight, float) else weight.item()
                
                # 額外記錄原始誤差
                with torch.no_grad():
                    raw_error = F.l1_loss(pred_depth, target_depth)
                    loss_dict['depth_l1_error'] = raw_error.item()
        
        # ============================================================
        # Task 3: Motion Prediction (回歸) 🔥 重要任務
        # 目標範圍: 0.05 - 0.3
        # ============================================================
        if 'motion' in outputs and 'motion' in targets:
            motion_loss = self.motion_loss(outputs['motion'], targets['motion'])
            
            weight = self._get_weight('motion', 3)
            total_loss = total_loss + weight * motion_loss
            
            loss_dict['motion'] = motion_loss.item()
            loss_dict['motion_weight'] = weight if isinstance(weight, float) else weight.item()
        
        # ============================================================
        # Task 4: Scene Classification (輔助任務)
        # ============================================================
        if 'scene_class' in outputs and 'scene_class' in targets:
            scene_loss = self.ce_loss(outputs['scene_class'], targets['scene_class'])
            
            weight = self._get_weight('scene_class', 4)
            total_loss = total_loss + weight * scene_loss
            
            loss_dict['scene_class'] = scene_loss.item()
            loss_dict['scene_class_weight'] = weight if isinstance(weight, float) else weight.item()
        
        # ============================================================
        # 遮擋重建 Loss (如果有)
        # ============================================================
        if 'occlusion_reconstruction' in outputs and 'clean_features' in targets:
            recon_loss = F.mse_loss(
                outputs['occlusion_reconstruction'],
                targets['clean_features']
            )
            weight = self._get_weight('occlusion_recon')
            total_loss = total_loss + weight * recon_loss
            loss_dict['occlusion_recon'] = recon_loss.item()
        
        # ============================================================
        # Memory Quality 正則化 (🔥 強化版)
        # 防止 memory_quality 趨向 0 或 1
        # ============================================================
        if 'memory_quality' in outputs:
            mq = outputs['memory_quality']
            
            # 雙向懲罰：鼓勵 mq 在 0.3-0.7 之間
            # 低於 0.3 的懲罰更強（防止 GRU 不工作）
            mq_reg_low = torch.clamp(0.35 - mq, min=0) ** 2 * 2.0  # 低於 0.35 強懲罰
            mq_reg_high = torch.clamp(mq - 0.65, min=0) ** 2       # 高於 0.65 輕懲罰
            
            # 額外的中心化 loss：鼓勵接近 0.5
            mq_center = (mq - 0.5) ** 2 * 0.1
            
            mq_reg = (mq_reg_low + mq_reg_high + mq_center).mean()
            
            weight = self._get_weight('memory_quality_reg')
            total_loss = total_loss + weight * mq_reg
            
            loss_dict['memory_quality'] = mq.mean().item()
            loss_dict['memory_quality_reg'] = mq_reg.item()
        
        # ============================================================
        # 總 Loss 診斷
        # ============================================================
        loss_dict['total_loss'] = total_loss.item()
        
        return total_loss, loss_dict
    
    def get_task_weights(self) -> Dict[str, float]:
        """取得當前各任務的權重"""
        if self.use_uncertainty_weighting:
            task_names = ['temporal', 'depth_order', 'depth_regression', 'motion', 'scene_class']
            weights = torch.exp(-self.log_vars).detach().cpu().numpy()
            return {name: float(w) for name, w in zip(task_names, weights)}
        else:
            return self.task_weights.copy()


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
    branch_params = {}
    
    # Shared Encoder (包含 Transformer 或 MLP)
    if model.use_transformer_encoder:
        encoder_params = sum(p.numel() for p in model.input_proj.parameters())
        encoder_params += sum(p.numel() for p in model.transformer_encoder.parameters())
        encoder_params += sum(p.numel() for p in model.encoder_norm.parameters())
        branch_params['shared_encoder (Transformer)'] = encoder_params
    else:
        branch_params['shared_encoder (MLP)'] = sum(p.numel() for p in model.shared_encoder.parameters())
    
    branch_params['temporal'] = sum(p.numel() for n, p in model.named_parameters() if 'temporal' in n and 'gru' not in n.lower() and 'transformer' not in n.lower())
    branch_params['depth_order'] = sum(p.numel() for p in model.depth_order_head.parameters())
    branch_params['depth_regression'] = sum(p.numel() for p in model.depth_regression_head.parameters())
    branch_params['motion'] = sum(p.numel() for n, p in model.named_parameters() if 'motion' in n)
    branch_params['scene_classifier'] = sum(p.numel() for p in model.scene_classifier.parameters())
    
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

    # 測試 Transformer Encoder 模式
    print("\n========== 測試 Transformer Encoder 模式 ==========")
    model = UnifiedTempoVLM(
        feat_dim=1536, 
        hidden_dim=768, 
        use_gru_memory=False,
        use_transformer_encoder=True,
        num_encoder_layers=2,
        num_heads=8
    )
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

    # 測試 GRU + Transformer 組合模式
    print("\n========== 測試 GRU + Transformer 組合模式 ==========")
    model_gru = UnifiedTempoVLM(
        feat_dim=1536, 
        hidden_dim=768, 
        use_gru_memory=True,
        use_transformer_encoder=True,
        num_encoder_layers=2,
        num_heads=8
    )
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

    # 測試舊版 MLP 模式（向後兼容）
    print("\n========== 測試舊版 MLP 模式（向後兼容）==========")
    model_mlp = UnifiedTempoVLM(
        feat_dim=1536, 
        hidden_dim=768, 
        use_gru_memory=False,
        use_transformer_encoder=False  # 使用舊版 MLP
    )
    model_mlp.eval()
    
    outputs_mlp, _ = model_mlp(
        curr_feat=curr_feat,
        prev_feat=prev_feat,
        region_a_feat=region_a,
        region_b_feat=region_b,
        tasks=['temporal', 'depth_order', 'depth_regression', 'motion']
    )
    print(f"  MLP mode - temporal output: {outputs_mlp['temporal'].shape}")
    print(f"  MLP mode - depth_regression output: {outputs_mlp['depth_regression'].shape}")

    print("\nTesting loss calculation with automatic weighting...")
    loss_fn = UnifiedLoss(use_uncertainty_weighting=True)
    
    # 模擬 GT 深度標籤（來自 ScanNet 數據集）
    targets = {
        'depth_order': torch.randint(0, 2, (batch_size,)),
        'depth_regression': torch.rand(batch_size, 3) * 5 + 0.5,  # 0.5~5.5m 的深度（GT）
        'motion': torch.randn(batch_size, 6) * 0.1,  # 小的運動值
        'scene_class': torch.randint(0, 20, (batch_size,)),
    }
    
    # 使用 Transformer 模型的 outputs
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

    print("\n========== Model Information ==========")
    
    print("\n📊 Transformer Encoder 模式:")
    info = get_model_info(model)
    print(f"  Total Parameters: {info['total_params']:,}")
    print(f"  Branch Parameters:")
    for branch, params in info['branch_params'].items():
        print(f"    {branch}: {params:,}")
    
    print("\n🧠 GRU + Transformer 模式:")
    info_gru = get_model_info(model_gru)
    print(f"  Total Parameters: {info_gru['total_params']:,}")
    print(f"  Branch Parameters:")
    for branch, params in info_gru['branch_params'].items():
        print(f"    {branch}: {params:,}")
    
    print("\n📦 舊版 MLP 模式:")
    info_mlp = get_model_info(model_mlp)
    print(f"  Total Parameters: {info_mlp['total_params']:,}")
    print(f"  Branch Parameters:")
    for branch, params in info_mlp['branch_params'].items():
        print(f"    {branch}: {params:,}")
    
    # 比較參數增量
    param_increase = info['total_params'] - info_mlp['total_params']
    print(f"\n📈 Transformer 相比 MLP 增加參數: {param_increase:,} ({param_increase/info_mlp['total_params']*100:.1f}%)")
    
    # 測試 Loss 函數的參數量
    loss_params = sum(p.numel() for p in loss_fn.parameters())
    print(f"\n  Loss function learnable params: {loss_params}")
    
    print("\n" + "="*60)
    print("💡 重要說明:")
    print("  1. 深度標籤（depth_regression）來自 ScanNet 的 GT 深度圖")
    print("  2. 模型學習從 RGB 特徵預測深度值")
    print("  3. Transformer Encoder 提供更強的特徵表達能力")
    print("  4. 可以用 use_transformer_encoder=False 切換回舊版 MLP")
    print("="*60)

    print("\n✅ Testing completed!")
