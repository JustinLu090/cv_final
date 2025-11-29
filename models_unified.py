

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple


class UnifiedTempoVLM(nn.Module):
    """
    Unified TempoVLM multi-task model

    Tasks:
    - temporal: 時序一致性 (原有功能)
    - depth_order: 深度排序 (A vs B 誰更近)
    - depth_regression: 相對深度值預測
    - motion: 相機運動預測 (6DoF)
    """
    
    def __init__(
        self,
        feat_dim: int = 1536,
        hidden_dim: int = 768,
        num_scene_classes: int = 20,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        
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
        # temporal consistency branch
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
        # depth regression branch (predict relative depth values)
        # ============================================================
        self.depth_regression_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        
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
        region_a_feat: Optional[torch.Tensor] = None,
        region_b_feat: Optional[torch.Tensor] = None,
        tasks: List[str] = ['temporal'],
    ) -> Dict[str, torch.Tensor]:
        """
        
        Args:
            curr_feat: 當前幀特徵 [B, feat_dim]
            prev_feat: 前一幀特徵 [B, feat_dim] (temporal/motion 用)
            region_a_feat: 區域 A 特徵 [B, feat_dim] (depth_order 用)
            region_b_feat: 區域 B 特徵 [B, feat_dim] (depth_order 用)
            tasks: 要執行的任務列表
        
        Returns:
            包含各任務輸出的字典
        """
        outputs = {}
        
        # 編碼當前幀
        curr_enc = self.shared_encoder(curr_feat)  # [B, hidden_dim]
        
        # ============================================================
        # Task 1 : temporal consistency
        # ============================================================
        if 'temporal' in tasks and prev_feat is not None:
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
        # Task 3 : depth regression
        # ============================================================
        if 'depth_regression' in tasks:
            outputs['depth_regression'] = self.depth_regression_head(curr_enc)  # [B, 1]
        
        # ============================================================
        # Task 4 : camera motion prediction
        # ============================================================
        if 'motion' in tasks and prev_feat is not None:
            prev_enc = self.shared_encoder(prev_feat)
            combined = torch.cat([curr_enc, prev_enc], dim=-1)
            fused = self.motion_fusion(combined)
            outputs['motion'] = self.motion_head(fused)  # [B, 6]
        
        if 'scene_class' in tasks:
            outputs['scene_class'] = self.scene_classifier(curr_enc)  # [B, num_classes]
        
        return outputs
    
    def forward_temporal(self, curr_feat: torch.Tensor, prev_feat: torch.Tensor) -> torch.Tensor:
        outputs = self.forward(curr_feat, prev_feat, tasks=['temporal'])
        return outputs['temporal']
    
    def forward_depth_order(
        self, 
        region_a_feat: torch.Tensor, 
        region_b_feat: torch.Tensor
    ) -> torch.Tensor:
        outputs = self.forward(
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
        outputs = self.forward(curr_feat, prev_feat, tasks=['motion'])
        return outputs['motion']


class UnifiedLoss(nn.Module):
    """
    unified multi-task loss for UnifiedTempoVLM
    """
    def __init__(
        self,
        temporal_weight: float = 1.0,
        depth_order_weight: float = 1.0,
        motion_weight: float = 1.0,
        scene_class_weight: float = 0.5,
    ):
        super().__init__()
        
        self.temporal_weight = temporal_weight
        self.depth_order_weight = depth_order_weight
        self.motion_weight = motion_weight
        self.scene_class_weight = scene_class_weight
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        prev_feat: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        calculate unified multi-task loss
        
        Args:
            outputs: model output dictionary
            targets: target dictionary
            prev_feat: previous frame features (for temporal consistency)
        
        Returns:
            total_loss: total loss
            loss_dict: individual task loss dictionary
        """
        total_loss = 0
        loss_dict = {}
        
        # temporal consistency loss
        if 'temporal' in outputs and prev_feat is not None:
            refined = outputs['temporal']
            # consistency with previous frame
            temporal_loss = 1 - F.cosine_similarity(
                refined.float(), prev_feat.float(), dim=-1
            ).mean()
            total_loss += temporal_loss * self.temporal_weight
            loss_dict['temporal'] = temporal_loss.item()
        
        # depth order loss
        if 'depth_order' in outputs and outputs['depth_order'] is not None:
            if 'depth_order' in targets:
                depth_order_loss = self.ce_loss(
                    outputs['depth_order'],
                    targets['depth_order']
                )
                total_loss += depth_order_loss * self.depth_order_weight
                loss_dict['depth_order'] = depth_order_loss.item()
        
        # depth regression loss
        if 'depth_regression' in outputs and outputs['depth_regression'] is not None:
            if 'depth_regression' in targets:
                # smooth L1 loss
                depth_reg_loss = F.smooth_l1_loss(
                    outputs['depth_regression'].squeeze(-1),
                    targets['depth_regression']
                )
                total_loss += depth_reg_loss * self.depth_order_weight  # common weight
                loss_dict['depth_regression'] = depth_reg_loss.item()
        
        # motion prediction loss
        if 'motion' in outputs and 'motion' in targets:
            motion_loss = self.mse_loss(
                outputs['motion'],
                targets['motion']
            )
            total_loss += motion_loss * self.motion_weight
            loss_dict['motion'] = motion_loss.item()
        
        # scene classification loss
        if 'scene_class' in outputs and 'scene_class' in targets:
            scene_loss = self.ce_loss(
                outputs['scene_class'],
                targets['scene_class']
            )
            total_loss += scene_loss * self.scene_class_weight
            loss_dict['scene_class'] = scene_loss.item()
        
        return total_loss, loss_dict


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
        'temporal': sum(p.numel() for n, p in model.named_parameters() if 'temporal' in n),
        'depth_order': sum(p.numel() for p in model.depth_order_head.parameters()),
        'motion': sum(p.numel() for n, p in model.named_parameters() if 'motion' in n),
        'scene_classifier': sum(p.numel() for p in model.scene_classifier.parameters()),
    }
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'branch_params': branch_params,
    }


# ============================================================
# 測試
# ============================================================

if __name__ == "__main__":
    print("測試 UnifiedTempoVLM...")
    
    model = UnifiedTempoVLM(feat_dim=1536, hidden_dim=768)
    model.eval()
    
    batch_size = 2
    curr_feat = torch.randn(batch_size, 1536)
    prev_feat = torch.randn(batch_size, 1536)
    region_a = torch.randn(batch_size, 1536)
    region_b = torch.randn(batch_size, 1536)
    
    print("\n測試多任務前向傳播...")
    outputs = model(
        curr_feat=curr_feat,
        prev_feat=prev_feat,
        region_a_feat=region_a,
        region_b_feat=region_b,
        tasks=['temporal', 'depth_order', 'motion', 'scene_class']
    )
    
    print(f"  temporal output: {outputs['temporal'].shape}")
    print(f"  depth_order output: {outputs['depth_order'].shape}")
    print(f"  motion output: {outputs['motion'].shape}")
    print(f"  scene_class output: {outputs['scene_class'].shape}")
    
    print("\n測試損失計算...")
    loss_fn = UnifiedLoss()
    targets = {
        'depth_order': torch.randint(0, 2, (batch_size,)),
        'motion': torch.randn(batch_size, 6),
        'scene_class': torch.randint(0, 20, (batch_size,)),
    }
    
    total_loss, loss_dict = loss_fn(outputs, targets, prev_feat)
    print(f"  Total loss: {total_loss.item():.4f}")
    for k, v in loss_dict.items():
        print(f"  {k} loss: {v:.4f}")
    
    print("\n模型資訊:")
    info = get_model_info(model)
    print(f"  總參數: {info['total_params']:,}")
    print(f"  各分支參數:")
    for branch, params in info['branch_params'].items():
        print(f"    {branch}: {params:,}")
    
    print("\n✅ 測試完成！")
