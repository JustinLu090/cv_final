import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
from collections import OrderedDict

# ==============================================================================
# Core Module: GRU memory unit
# ==============================================================================

class GRUMemoryCell(nn.Module):
    """
    TempoVLM GRU long-term memory unit
    """
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Core GRU
        self.gru = nn.GRUCell(input_size, hidden_size)
        
        # Memory quality gate
        self.memory_quality_gate = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
        # Memory output gate
        self.memory_output_gate = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Sigmoid()
        )

    def forward(self, input_feat: torch.Tensor, hidden_state: torch.Tensor):
        # 1. Memory quality gate
        combined_for_quality = torch.cat([input_feat, hidden_state], dim=-1)
        quality_score = self.memory_quality_gate(combined_for_quality)  # [B, 1]
        
        # 2. Update memory with GRU
        gru_input = input_feat * quality_score + hidden_state * (1 - quality_score)
        next_hidden_state = self.gru(gru_input, hidden_state)

        # 3. Determine how much memory to use in output
        combined_for_output = torch.cat([input_feat, next_hidden_state], dim=-1)
        output_gate = self.memory_output_gate(combined_for_output)  # [B, hidden_dim]
        
        # 4. Fuse current observation with long-term memory
        fused_enc = output_gate * next_hidden_state + (1 - output_gate) * input_feat
        
        return fused_enc, next_hidden_state, quality_score


# ==============================================================================
# Core Module: Unified multi-task model
# ==============================================================================

# Residual MLP block (for depth regression)
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )
    def forward(self, x):
        return x + self.net(x)

class UnifiedTempoVLM(nn.Module):
    """
    Unified TempoVLM multi-task model with GRU Long-term Memory
    """
    
    def __init__(
        self,
        feat_dim: int = 1536,
        hidden_dim: int = 768,
        num_scene_classes: int = 20,
        dropout: float = 0.1,
        use_gru_memory: bool = True,
        use_transformer_encoder: bool = True,
        num_encoder_layers: int = 2,
        num_heads: int = 8,
    ):
        super().__init__()
        
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.use_gru_memory = use_gru_memory
        self.use_transformer_encoder = use_transformer_encoder
        self.max_depth = 10.0 # ScanNet max depth

        # ============================================================
        # shared encoder
        # ============================================================
        if use_transformer_encoder:
            self.input_proj = nn.Linear(feat_dim, hidden_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
                dropout=dropout, activation='gelu', batch_first=True, norm_first=True,
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
            self.encoder_norm = nn.LayerNorm(hidden_dim)
            
            self.shared_encoder = nn.Sequential(OrderedDict([
                ('input_proj', self.input_proj),
                ('encoder_layer', self.transformer_encoder),
                ('encoder_norm', self.encoder_norm)
            ]))
            
            def shared_encoder_forward(x):
                x = self.input_proj(x)
                x = self.transformer_encoder(x.unsqueeze(1)).squeeze(1)
                return self.encoder_norm(x)
            
            self._shared_encoder_forward = shared_encoder_forward

        else:
            self.shared_encoder = nn.Sequential(
                nn.Linear(feat_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self._shared_encoder_forward = self.shared_encoder.forward
        
        # ============================================================
        # GRU Long-term Memory
        # ============================================================
        if use_gru_memory:
            self.temporal_gru = GRUMemoryCell(hidden_dim, hidden_dim)

        # ============================================================
        # temporal consistency branch
        # ============================================================
        self.temporal_output = nn.Sequential(
            nn.Linear(hidden_dim, feat_dim),
        )

        # ============================================================
        # [NEW] Contrastive Projection Head (mitigate feature collapse)
        # ============================================================
        self.contrastive_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 128)  # Compress to 128 dims for compact and separable features
        )

        # ============================================================
        # depth regression branch
        # ============================================================
        # Use ResidualBlock to improve representation power
        self.depth_regression_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), 
            ResidualBlock(hidden_dim, dropout), 
            ResidualBlock(hidden_dim, dropout), 
            nn.Linear(hidden_dim, 3), 
        )
        
        # Absolute-scale prediction head
        self.depth_scale_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1), 
            nn.Softplus(), 
        )
        
        # ============================================================
        # Other task heads
        # ============================================================
        self.depth_order_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )
        
        self.motion_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        self.motion_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 6),
        )
        self.motion_scale = nn.Parameter(torch.tensor([0.05, 0.05, 0.05, 0.02, 0.02, 0.02]))
        self.motion_uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4), nn.GELU(), nn.Linear(hidden_dim // 4, 6),
        )
        self.global_scale_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4), nn.GELU(), nn.Linear(hidden_dim // 4, 1), nn.Softplus(),
        )
        self.motion_quality_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, 1), nn.Sigmoid(),
        )
        self.place_embedding = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2),
        )
        self.scene_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim // 2, num_scene_classes),
        )
        
        self._init_weights()
    
    def init_hidden_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device)

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
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def load_pretrained_temporal(self, checkpoint_path: str, strict: bool = False):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        print("Original checkpoint keys:")
        for k, v in state_dict.items():
            print(f"   {k}: {v.shape}")
        
        compatible_keys = []
        try:
            self.load_state_dict(state_dict, strict=False)
            print("Partial pretrained weights loaded successfully (strict=False)")
        except Exception as e:
            print(f"Warning: error while loading pretrained weights: {e}")
        return compatible_keys

    def forward(
        self,
        curr_feat: torch.Tensor,
        prev_feat: Optional[torch.Tensor] = None,
        hidden_state: Optional[torch.Tensor] = None,
        region_a_feat: Optional[torch.Tensor] = None,
        region_b_feat: Optional[torch.Tensor] = None,
        tasks: List[str] = ['temporal'],
    ) -> Tuple[Dict[str, torch.Tensor], Optional[torch.Tensor]]:
        
        outputs = {}
        next_hidden_state = None
        
        # 1. Encode current frame
        curr_enc = self._shared_encoder_forward(curr_feat)
        batch_size = curr_feat.shape[0]
        device = curr_feat.device
        
        # 2. GRU long-term memory
        fused_enc = curr_enc 
        if self.use_gru_memory:
            if hidden_state is None:
                hidden_state = self.init_hidden_state(batch_size, device)
            
            fused_enc, next_hidden_state, quality_score = self.temporal_gru(curr_enc, hidden_state)
            outputs['memory_quality'] = quality_score.mean()
        
        # 3. Temporal Consistency Output
        if 'temporal' in tasks:
            refined_features = self.temporal_output(fused_enc)
            outputs['temporal'] = curr_feat + refined_features
            
            # [NEW] Compute projection-head features for contrastive learning
            # Use fused memory features (fused_enc)
            outputs['contrastive_feat'] = self.contrastive_head(fused_enc)
        
        # 4. Depth regression
        if 'depth_regression' in tasks:
            # 1. Predict relative depth shape (0~1)
            raw_depth = self.depth_regression_head(fused_enc)
            depth_relative = torch.sigmoid(raw_depth) 
            
            # 2. Predict scene scale factor
            depth_scale = self.depth_scale_head(fused_enc)
            
            # 3. Combine to absolute depth
            outputs['depth_regression'] = depth_relative * depth_scale 
            outputs['depth_relative'] = depth_relative 
            outputs['depth_scale'] = depth_scale.squeeze(-1)
        
        # 5. Depth Order
        if 'depth_order' in tasks and region_a_feat is not None:
            region_a_enc = self._shared_encoder_forward(region_a_feat)
            region_b_enc = self._shared_encoder_forward(region_b_feat)
            combined = torch.cat([region_a_enc, region_b_enc], dim=-1)
            outputs['depth_order'] = self.depth_order_head(combined)
        
        # 6. Motion Prediction
        if 'motion' in tasks and prev_feat is not None:
            prev_enc = self._shared_encoder_forward(prev_feat)
            combined = torch.cat([fused_enc, prev_enc], dim=-1)
            fused = self.motion_fusion(combined)
            raw_motion = self.motion_head(fused)
            
            motion = raw_motion * self.motion_scale.unsqueeze(0)
            motion_log_var = self.motion_uncertainty_head(fused)
            global_scale = self.global_scale_head(fused_enc)
            global_scale = 0.5 + 1.5 * torch.sigmoid(global_scale - 1)
            motion_quality = self.motion_quality_head(combined)
            place_emb = self.place_embedding(fused_enc)
            
            outputs['motion'] = motion
            outputs['motion_log_var'] = motion_log_var
            outputs['motion_global_scale'] = global_scale
            outputs['motion_quality'] = motion_quality
            outputs['place_embedding'] = place_emb
        
        # 7. Scene Classification
        if 'scene_class' in tasks:
            outputs['scene_class'] = self.scene_classifier(fused_enc)

        return outputs, next_hidden_state

    def forward_temporal(
        self, 
        curr_feat: torch.Tensor, 
        prev_feat: torch.Tensor = None,
        hidden_state: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
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
    
    def encode_and_project(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Helper: encode raw 1536-d features and project to 128-d contrastive space
        Used to compute contrastive targets (prev_feat) and negatives (global_neg)
        """
        # 1. Shared Encoder (1536 -> 768)
        enc = self._shared_encoder_forward(feat)
        # 2. Projection Head (768 -> 128)
        proj = self.contrastive_head(enc)
        return proj

# ==============================================================================
# Loss function
# ==============================================================================

class UnifiedLoss(nn.Module):
    """
    Unified multi-task loss for UnifiedTempoVLM
    
    [Update: strengthen depth-regression loss with scale robustness and supervision]
    """
    def __init__(
        self,
        num_tasks: int = 5,
        use_uncertainty_weighting: bool = False,
        task_weights: Dict[str, float] = None,
    ):
        super().__init__()
        
        self.use_uncertainty_weighting = use_uncertainty_weighting
        
        if task_weights is None:
            self.task_weights = {
                'temporal': 0.1,          
                'depth_order': 1.0,       
                'depth_regression': 5.0,  # Increase depth regression weight
                'motion': 2.0,            
                'scene_class': 0.5,       
                'occlusion_recon': 1.5,   
                'memory_quality_reg': 0.5,
            }
        else:
            self.task_weights = task_weights
        
        if use_uncertainty_weighting:
            init_log_vars = torch.tensor([2.0, 0.0, -1.0, -0.5, 0.5])
            self.log_vars = nn.Parameter(init_log_vars)
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        
    def _get_weight(self, task_name: str, task_idx: int = None) -> float:
        """Get task weight"""
        if self.use_uncertainty_weighting and task_idx is not None:
            log_var = torch.clamp(self.log_vars[task_idx], min=-4, max=4)
            return torch.exp(-log_var)
        else:
            return self.task_weights.get(task_name, 1.0)
    
    def scale_invariant_depth_loss(self, pred, target):
        """Compute scale-invariant depth loss (SI) and relative error loss (Rel)"""
        
        # Filter invalid values
        valid_mask = target > 0.1
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device), torch.tensor(0.0, device=pred.device)

        pred_valid = pred[valid_mask].clamp(min=1e-6)
        target_valid = target[valid_mask].clamp(min=1e-6)
        
        # 1. Scale-Invariant Loss (SI Loss)
        log_diff = torch.log(pred_valid) - torch.log(target_valid)
        n = log_diff.numel()
        if n > 0:
            # SI loss: penalize variance of log-depth differences
            si_loss = torch.mean(log_diff ** 2) - 0.85 * (torch.mean(log_diff) ** 2) 
        else:
            si_loss = torch.tensor(0.0, device=pred.device)
        
        # 2. Relative error loss (Rel loss)
        rel_loss = torch.mean(torch.abs(pred_valid - target_valid) / (target_valid + 1e-6))

        return si_loss, rel_loss
    
    # ... (motion_loss and temporal_contrastive_loss remain unchanged)
    def motion_loss(self, pred, target, log_var=None):
        pred_trans = pred[:, :3]; pred_rot = pred[:, 3:]
        target_trans = target[:, :3]; target_rot = target[:, 3:]
        trans_loss = F.smooth_l1_loss(pred_trans, target_trans)
        rot_loss = F.mse_loss(pred_rot, target_rot)
        return trans_loss + rot_loss
    
    def temporal_contrastive_loss(self, refined_curr, prev_feat):
        batch_size = refined_curr.shape[0]
        if batch_size <= 1:
            return 1 - F.cosine_similarity(refined_curr.float(), prev_feat.float(), dim=-1).mean(), {}, {}
        
        refined_norm = F.normalize(refined_curr, p=2, dim=-1)
        prev_norm = F.normalize(prev_feat, p=2, dim=-1)
        
        sim_matrix = refined_norm @ prev_norm.T
        tau = 0.1 
        
        exp_sim = torch.exp(sim_matrix / tau)
        pos_exp = torch.diag(exp_sim)
        
        mask = torch.eye(batch_size, device=sim_matrix.device).bool()
        neg_exp_sum = exp_sim.masked_fill(mask, 0).sum(dim=1)
        
        infonce_loss = -torch.log(pos_exp / (pos_exp + neg_exp_sum + 1e-8)).mean()
        
        pos_sim = torch.diag(sim_matrix)
        pos_sim_loss = F.relu(0.8 - pos_sim).mean() 
        
        total_loss = 0.3 * infonce_loss + 0.7 * pos_sim_loss
        
        with torch.no_grad():
            neg_sim = sim_matrix.masked_fill(mask, 0).sum() / (batch_size * (batch_size - 1))
        
        return total_loss, {'pos_sim': pos_sim.mean().item(), 'neg_sim': neg_sim.item()}, {'infonce': infonce_loss.item()}
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        prev_feat: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        
        total_loss = torch.tensor(0.0, device=next(iter(outputs.values())).device)
        loss_dict = OrderedDict()
        
        # ... (Temporal loss and depth-order loss remain unchanged)
        if 'temporal' in outputs and prev_feat is not None:
            temporal_loss, diag_info, raw_losses = self.temporal_contrastive_loss(outputs['temporal'], prev_feat)
            weight = self._get_weight('temporal', 0)
            total_loss = total_loss + weight * temporal_loss
            loss_dict['temporal'] = temporal_loss.item()
            if diag_info:
                loss_dict['temporal_pos_sim'] = diag_info['pos_sim']
                loss_dict['temporal_neg_sim'] = diag_info['neg_sim']

        if 'depth_order' in outputs and outputs['depth_order'] is not None:
            if 'depth_order' in targets:
                depth_order_loss = self.ce_loss(outputs['depth_order'], targets['depth_order'])
                weight = self._get_weight('depth_order', 1)
                total_loss = total_loss + weight * depth_order_loss
                loss_dict['depth_order'] = depth_order_loss.item()

        # ============================================================
        # Task 2: Depth Regression (regression) - strengthened scale robustness
        # ============================================================
        if 'depth_regression' in outputs and outputs['depth_regression'] is not None:
            if 'depth_regression' in targets:
                pred_depth = outputs['depth_regression']
                target_depth = targets['depth_regression']
                
                # 1. Dynamic scale alignment (median normalization)
                valid_mask = target_depth > 0.1
                if valid_mask.sum() > 0:
                    pred_valid = pred_depth[valid_mask]
                    target_valid = target_depth[valid_mask]
                    
                    # Compute median ratio (GT median / prediction median)
                    median_target = target_valid.median()
                    median_pred = pred_valid.median()
                    dynamic_scale_factor = median_target / (median_pred + 1e-6)
                    
                    # Calibrate predicted depth by median ratio to match batch GT scale
                    pred_aligned = pred_depth * dynamic_scale_factor
                    
                    # 2. Loss computation (using aligned predictions)
                    si_loss, rel_loss = self.scale_invariant_depth_loss(pred_aligned, target_depth)
                    
                    # 3. [Fix] Scale consistency loss (supervise scale head)
                    if 'depth_scale' in outputs:
                        pred_scale_proxy = outputs['depth_scale']
                        
                        # [Fix]: Expand scalar target to [B] to match predictions
                        target_scale_proxy = median_target.view(1).expand_as(pred_scale_proxy)
                        
                        scale_loss = F.l1_loss(torch.log(pred_scale_proxy + 1e-6), torch.log(target_scale_proxy + 1e-6))
                    else:
                        scale_loss = torch.tensor(0.0, device=pred_depth.device)

                    # 4. Combine losses
                    depth_reg_loss = 3.0 * si_loss + 1.0 * rel_loss + 0.5 * scale_loss
                    
                    # Apply task weight
                    weight = self._get_weight('depth_regression', 2)
                    total_loss = total_loss + weight * depth_reg_loss
                    
                    loss_dict['depth_regression'] = depth_reg_loss.item()
                    loss_dict['depth_SI'] = si_loss.item() 
                    loss_dict['depth_Rel'] = rel_loss.item()
                    loss_dict['depth_Scale'] = scale_loss.item()
                else:
                    depth_reg_loss = torch.tensor(0.0, device=pred_depth.device)
                    total_loss = total_loss + depth_reg_loss
                    loss_dict['depth_regression'] = 0.0
        
        # ... (Motion, scene class, occlusion reconstruction, and memory quality remain unchanged)
        if 'motion' in outputs and 'motion' in targets:
            motion_loss = self.motion_loss(outputs['motion'], targets['motion'])
            weight = self._get_weight('motion', 3)
            total_loss = total_loss + weight * motion_loss
            loss_dict['motion'] = motion_loss.item()

        if 'scene_class' in outputs and 'scene_class' in targets:
            scene_loss = self.ce_loss(outputs['scene_class'], targets['scene_class'])
            weight = self._get_weight('scene_class', 4)
            total_loss = total_loss + weight * scene_loss
            loss_dict['scene_class'] = scene_loss.item()

        if 'occlusion_reconstruction' in outputs and 'clean_features' in targets:
            recon_loss = F.mse_loss(outputs['occlusion_reconstruction'], targets['clean_features'])
            weight = self._get_weight('occlusion_recon')
            total_loss = total_loss + weight * recon_loss
            loss_dict['occlusion_recon'] = recon_loss.item()

        if 'memory_quality' in outputs:
            mq = outputs['memory_quality']
            mq_reg_low = torch.clamp(0.35 - mq, min=0) ** 2 * 2.0  
            mq_reg_high = torch.clamp(mq - 0.65, min=0) ** 2       
            mq_center = (mq - 0.5) ** 2 * 0.1
            mq_reg = (mq_reg_low + mq_reg_high + mq_center).mean()
            weight = self._get_weight('memory_quality_reg')
            total_loss = total_loss + weight * mq_reg
            loss_dict['memory_quality'] = mq.mean().item()
            loss_dict['memory_quality_reg'] = mq_reg.item()
        
        # Total loss diagnostics
        loss_dict['total_loss'] = total_loss.item()
        
        return total_loss, loss_dict
    
    def get_task_weights(self) -> Dict[str, float]:
        if self.use_uncertainty_weighting:
            task_names = ['temporal', 'depth_order', 'depth_regression', 'motion', 'scene_class']
            weights = torch.exp(-self.log_vars).detach().cpu().numpy()
            return {name: float(w) for name, w in zip(task_names, weights)}
        else:
            return self.task_weights.copy()

def get_model_info(model: UnifiedTempoVLM) -> Dict:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    branch_params = {}
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
    
    if model.use_gru_memory:
        gru_params = sum(p.numel() for p in model.temporal_gru.parameters())
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'branch_params': branch_params,
    }
