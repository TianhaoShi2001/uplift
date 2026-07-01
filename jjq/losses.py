import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 核心阶段一 (Stage 1): Model C 专属 Loss
# ==========================================

class FocalLoss(nn.Module):
    """解决样本极度不平衡 (动态压制易分样本的梯度权重)"""
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        targets = targets.type_as(logits)
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = focal_weight * (1 - p_t) ** self.gamma * bce_loss
        
        if self.reduction == 'mean': return focal_loss.mean()
        elif self.reduction == 'sum': return focal_loss.sum()
        return focal_loss

def mmd_loss(x, y, kernel="rbf", sigma=1.0):
    """
    保证表征的干预不变性 (Intervention Invariance)
    修复了 T=1 和 T=0 样本数量不一致时的 Tensor 广播 Bug
    """
    if x.size(0) == 0 or y.size(0) == 0:
        return torch.tensor(0.0, device=x.device, requires_grad=True)
    
    # xx: (N1, N1), yy: (N2, N2), zz: (N1, N2)
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    
    # 🌟 修复点：提取对角线（每个样本的 L2 范数平方），并重塑维度准备广播
    rx = xx.diag().unsqueeze(1) # 维度: (N1, 1)
    ry = yy.diag().unsqueeze(0) # 维度: (1, N2)
    
    if kernel == "rbf":
        # 利用广播机制计算距离矩阵
        # rx + rx.t() 维度: (N1, 1) + (1, N1) -> (N1, N1)
        dxx = rx + rx.t() - 2. * xx
        # ry.t() + ry 维度: (N2, 1) + (1, N2) -> (N2, N2)
        dyy = ry.t() + ry - 2. * yy
        # rx + ry     维度: (N1, 1) + (1, N2) -> (N1, N2)
        dxy = rx + ry - 2. * zz
        
        K_xx = torch.exp(-dxx / (2.0 * sigma ** 2))
        K_yy = torch.exp(-dyy / (2.0 * sigma ** 2))
        K_xy = torch.exp(-dxy / (2.0 * sigma ** 2))
    else: 
        K_xx, K_yy, K_xy = xx, yy, zz
        
    return K_xx.mean() + K_yy.mean() - 2. * K_xy.mean()

def sliced_wasserstein_distance(source_features, target_features, num_projections=128, p=2):
    """
    计算切片 Wasserstein 距离 (SWD)，极低显存占用
    """
    if source_features.size(0) <= 1 or target_features.size(0) <= 1:
        return torch.tensor(0.0, device=source_features.device, requires_grad=True)
        
    dim = source_features.size(1)
    device = source_features.device
    
    # 随机投影方向
    projections = torch.randn(dim, num_projections, device=device)
    projections = projections / torch.norm(projections, dim=0, keepdim=True)
    
    # 投影到一维
    source_proj = torch.matmul(source_features, projections)
    target_proj = torch.matmul(target_features, projections)
    
    # 排序
    source_proj_sorted, _ = torch.sort(source_proj, dim=0)
    target_proj_sorted, _ = torch.sort(target_proj, dim=0)
    
    # 注意这里加上 requires_grad_，因为当一边样本极少时可能导致梯度断裂
    dist = torch.mean(torch.pow(torch.abs(source_proj_sorted - target_proj_sorted), p))
    return dist

def moment_matching_loss(source_features, target_features):
    """
    一阶（均值）和二阶（协方差）矩匹配，极速且免调超参
    """
    if source_features.size(0) <= 1 or target_features.size(0) <= 1:
        return torch.tensor(0.0, device=source_features.device, requires_grad=True)
        
    # 一阶矩对齐
    mean_source = source_features.mean(dim=0)
    mean_target = target_features.mean(dim=0)
    mean_loss = torch.mean(torch.pow(mean_source - mean_target, 2))
    
    # 二阶矩对齐
    def compute_covariance(x):
        batch_size = x.size(0)
        x_centered = x - x.mean(dim=0, keepdim=True)
        return torch.matmul(x_centered.t(), x_centered) / (batch_size - 1 + 1e-5)
        
    cov_source = compute_covariance(source_features)
    cov_target = compute_covariance(target_features)
    cov_loss = torch.mean(torch.pow(cov_source - cov_target, 2))
    
    return mean_loss + cov_loss


class UpliftGroupDROLoss(nn.Module):
    def __init__(self, grouping_mode="2d_coarse", ng=0.01, ema_gamma=0.1, max_clip=0.5):
        """
        :param grouping_mode: 
            "1d_coarse" -> 按 pi_01 粗切 4 组 (0-5%, 5-10%, 10-20%, 20-100%)
            "2d_coarse" -> 粗切 4 组 x 4 种因果事实 (T, Y) = 16 组
            "1d_fine"   -> 按 pi_01 细切 10 组 (每 10% 一刀)
            "2d_fine"   -> 细切 10 组 x 4 种因果事实 = 40 组
        """
        super().__init__()
        self.grouping_mode = grouping_mode
        self.ng = ng
        self.ema_gamma = ema_gamma
        self.max_clip = max_clip
        
        if grouping_mode == "1d_coarse": self.num_groups = 4
        elif grouping_mode == "2d_coarse": self.num_groups = 16
        elif grouping_mode == "1d_fine": self.num_groups = 11
        elif grouping_mode == "2d_fine": self.num_groups = 44
        else: raise ValueError(f"未知的分组模式: {grouping_mode}")
        
        # 注册持久化 Buffer (在 Batch 间保持权重)
        self.register_buffer("group_weights", torch.ones(self.num_groups) / self.num_groups)
        self.register_buffer("ema_loss", torch.zeros(self.num_groups))
        self.register_buffer("ema_initialized", torch.zeros(self.num_groups, dtype=torch.bool))

    def forward(self, bce_losses, pi_01, t, y):
        pi_01 = pi_01.detach().float()
        
        # --- 1. 动态计算分位数，实现物理隔离分桶 ---
        if "coarse" in self.grouping_mode:
            # 粗切 4 组：0-5%, 5-10%, 10-20%, 20-100%
            boundaries = torch.quantile(pi_01, torch.tensor([0.80, 0.90, 0.95], device=pi_01.device))
            # bucketize 后：<=80%是0，80-90%是1，90-95%是2，>95%是3。用 3 减去它，保证 Top 5% 是 Bin 0
            pi_bins = 3 - torch.bucketize(pi_01, boundaries)
        else:
            # 🌟 细切 11 组：坚守头部隔离底线！
            # 切点为: 10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%, 95% (共10刀)
            fine_quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
            boundaries = torch.quantile(pi_01, torch.tensor(fine_quantiles, device=pi_01.device))
            
            # bucketize 之后产生 11 个桶 (0 到 10)
            # >95% 会被分到桶 10，为了让 Top 5% 变成 Bin 0，用 10 减去它
            pi_bins = 10 - torch.bucketize(pi_01, boundaries) 
            # 结果映射：
            # Bin 0:  0-5%   (真大佬，重点保护)
            # Bin 1:  5-10%  (羊毛党，精准靶点)
            # Bin 2:  10-20% 
            # Bin 3:  20-30%
            # ...
            # Bin 10: 90-100%
            
        # --- 2. 判断是否进行二维交叉 ---
        if "2d" in self.grouping_mode:
            causal_state = (t.long() * 2 + y.long()) # 0, 1, 2, 3 四个象限
            num_pi_bins = 4 if "coarse" in self.grouping_mode else 10
            group_idx = pi_bins * 4 + causal_state
        else:
            group_idx = pi_bins
            
        group_idx = group_idx.long()

        # --- 3. 统计当前 Batch 内各个组的 Loss ---
        group_counts = torch.bincount(group_idx, minlength=self.num_groups).float()
        group_loss_sum = torch.zeros(self.num_groups, device=bce_losses.device)
        group_loss_sum.scatter_add_(0, group_idx, bce_losses)
        
        valid_groups = group_counts > 0
        current_group_loss = torch.zeros_like(self.ema_loss)
        current_group_loss[valid_groups] = group_loss_sum[valid_groups] / group_counts[valid_groups]

        # --- 4. DRO 核心演化算法 (不参与反向传播) ---
        with torch.no_grad():
            # EMA 平滑
            first_time = (~self.ema_initialized) & valid_groups
            if first_time.any():
                self.ema_loss[first_time] = current_group_loss[first_time]
                self.ema_initialized[first_time] = True
            
            self.ema_loss[valid_groups] = (1 - self.ema_gamma) * self.ema_loss[valid_groups] + self.ema_gamma * current_group_loss[valid_groups]
            
            # 梯度重加权 (指数上升)
            new_weights = self.group_weights * torch.exp(self.ng * self.ema_loss)
            
            # 强制截断 (防止某个组吃掉所有算力走火入魔)
            new_weights = torch.clamp(new_weights, max=self.max_clip)
            self.group_weights = new_weights / new_weights.sum()

        # --- 5. 组装为可反向传播的最终 Loss ---
        # 巧妙地分发到样本维度：样本权重 = 组权重 / 组内人数
        sample_weights = self.group_weights[group_idx] / torch.clamp(group_counts[group_idx], min=1.0)
        final_dro_loss = torch.sum(sample_weights * bce_losses)
        
        return final_dro_loss


# ==========================================
# 核心阶段三 (Stage 3): Model Y 靶向联合优化 Loss
# ==========================================
def compute_stage3_loss(y0_pred, y1_pred, targets, treatment, pi_dict, config, dro_criterion=None):
    loss_types = config.get("loss_types", ["bce"])
    y_pred_observed = torch.where(treatment == 1, y1_pred, y0_pred)
    bce_raw = F.binary_cross_entropy_with_logits(y_pred_observed, targets.float(), reduction='none')
    
    total_loss = 0.0
    loss_components = {} 
    
    # ---------------------------------------------------------
    # 1. 靶向重加权 (Strata Weighted BCE)
    # 核心：越有可能是 Complier，拟合观测数据的权重越大
    # ---------------------------------------------------------
    if "strata_weighted" in loss_types:
        # 使用 p_complier 作为权重，加上一个极小值防止全 0
        weight_c = pi_dict["p_complier"].detach() + 1e-4
        
        # 归一化，保证 Batch 级别的 Loss 量级与标准 BCE 对齐
        weight_c = weight_c / weight_c.mean() 
        sw_loss = (bce_raw * weight_c).mean()
        
        total_loss += sw_loss
        loss_components["sw_loss"] = sw_loss.item()
    elif "group_dro" in loss_types and dro_criterion is not None:
        dro_loss = dro_criterion(bce_raw, pi_dict["p_complier"], treatment, targets)
        total_loss += dro_loss
        loss_components["dro_loss"] = dro_loss.item()
        
    elif"prior_conflict" in loss_types:
        pi_01 = pi_dict["p_complier"].detach()
        alpha = config.get("conflict_alpha", 2.0)
        mode = config.get("conflict_mode", "both") # 取值: 'wool_only', 'gold_only', 'both'
        
        # 1. 所有人初始权重为 1.0 (稳住大盘)
        weights = torch.ones_like(targets, dtype=torch.float)
        
        # 2. 定位两类错位人群
        mask_wool = (treatment == 1) & (targets == 0) # 羊毛党：发了券没买
        mask_gold = (treatment == 1) & (targets == 1) # 隐藏金子：发了券真买了
        
        # 3. 根据 Mode 动态施加精准打击
        if mode in ["wool_only", "both"] and mask_wool.any():
            # co 越高，打脸越疼，权重加得越多
            weights[mask_wool] += alpha * pi_01[mask_wool]
            
        if mode in ["gold_only", "both"] and mask_gold.any():
            # co 越低，被模型看扁得越惨，权重加得越多
            weights[mask_gold] += alpha * (1.0 - pi_01[mask_gold])
            
        # 4. 均值归一化：保证 Batch 总 Loss 不会因为系数爆炸而导致梯度崩盘
        weights = weights / (weights.mean() + 1e-8)
        
        # 5. 计算加权 Loss
        conflict_loss = (bce_raw * weights).mean()
        total_loss += conflict_loss
        loss_components["conflict_loss"] = conflict_loss.item()
    else:
        # 纯净的 Baseline BCE
        bce_loss = bce_raw.mean()
        total_loss += bce_loss
        loss_components["bce_loss"] = bce_loss.item()
        
    # ---------------------------------------------------------
    # 2. 方差正则化 (Variance/Consistency Regularization) 
    # [🌟 重构版：软加权机制]
    # 核心：不管你发不发券都不会改变结局的人 (Never & Always)，他们的 Uplift 必须趋近于 0
    # ---------------------------------------------------------
    if "variance_reg" in loss_types:
        # 我们用模型算出来的概率作为惩罚权重
        p_never = pi_dict["p_never"].detach()
        p_always = pi_dict["p_always"].detach()
        
        # 综合“非敏感客”的概率作为强迫 y1 和 y0 对齐的权重
        # 你也可以只用 p_never，看你的业务假设
        weight_v = (p_never + p_always) 
        
        # 计算每个人的预测增益的平方
        uplift_variance = torch.pow(y1_pred - y0_pred, 2)
        
        # 🌟 软加权惩罚：概率越大，惩罚越重
        var_penalty = (uplift_variance * weight_v).mean()
        
        var_lambda = config.get("var_lambda", 1.0)
        total_loss += var_lambda * var_penalty
        loss_components["var_loss"] = (var_lambda * var_penalty).item()

    # ---------------------------------------------------------
    # 3. Pairwise 排序约束 (Rank Loss)
    # ---------------------------------------------------------
    if "pairwise" in loss_types:
        uplift_pred = y1_pred - y0_pred
        pi_01 = pi_dict["p_complier"].detach()
        q_high = torch.quantile(pi_01, 0.75) 
        q_low = torch.quantile(pi_01, 0.25)  
        
        high_mask = pi_01 >= q_high
        low_mask = pi_01 <= q_low
        
        if high_mask.any() and low_mask.any():
            u_high = uplift_pred[high_mask].mean()
            u_low = uplift_pred[low_mask].mean()
            margin = config.get("rank_margin", 0.05)
            rank_loss = F.relu(-(u_high - u_low) + margin)
            rank_alpha = config.get("rank_alpha", 1.0)
            total_loss += rank_alpha * rank_loss
            loss_components["rank_loss"] = (rank_alpha * rank_loss).item()

    # ---------------------------------------------------------
    # 4. MOTTO DA Loss (Distribution Alignment)
    # ---------------------------------------------------------
    if "motto_da" in loss_types and "treatment_shared_outputs" in pi_dict:
        treatment_shared_outputs = pi_dict["treatment_shared_outputs"]
        da_weight = config.get("motto_da_weight", 1.0)
        num_experts = treatment_shared_outputs.shape[0]
        batch_size = treatment_shared_outputs.shape[1]
        
        da_losses = []
        # 使用 MMD 近似 geomloss
        for expert_idx in range(num_experts):
            expert_outputs = treatment_shared_outputs[expert_idx]
            z_t1, z_t0 = expert_outputs[treatment == 1], expert_outputs[treatment == 0]
            if len(z_t1) > 0 and len(z_t0) > 0:
                expert_loss = mmd_loss(z_t1, z_t0, sigma=config.get("mmd_sigma", 1.0))
                da_losses.append(expert_loss)
                
        if da_losses:
            da_loss_val = torch.stack(da_losses).mean()
            # 防爆裁剪
            da_loss_val = torch.clamp(da_loss_val, max=1000.0)
            total_loss += da_weight * da_loss_val
            loss_components["da_loss"] = (da_weight * da_loss_val).item()
            
    return total_loss, loss_components


# # ==========================================
# # 核心阶段三 (Stage 3): Model Y 靶向联合优化 Loss
# # ==========================================
# def compute_stage3_loss(y0_pred, y1_pred, targets, treatment, pi_dict, config):
#     loss_types = config.get("loss_types", ["bce"])
#     y_pred_observed = torch.where(treatment == 1, y1_pred, y0_pred)
#     bce_raw = F.binary_cross_entropy_with_logits(y_pred_observed, targets.float(), reduction='none')
    
#     total_loss = 0.0
#     loss_components = {} 
    
#     # 1. 靶向重加权
#     if "strata_weighted" in loss_types:
#         weight = pi_dict["p_complier"].detach() + 1e-4
#         # 🌟 加一行归一化：让 Batch 内所有人的权重加起来等于人数 (均值为 1)
#         # 这样 sw_loss 的绝对数值量级，就会和普通的 bce_loss 保持同一水平线
#         weight = weight / weight.mean() 
#         sw_loss = (bce_raw * weight).mean()
#         total_loss += sw_loss
#         loss_components["sw_loss"] = sw_loss.item()
#     else:
#         bce_loss = bce_raw.mean()
#         total_loss += bce_loss
#         loss_components["bce_loss"] = bce_loss.item()
        
#     # 2. Pairwise 排序约束
#     if "pairwise" in loss_types:
#         uplift_pred = y1_pred - y0_pred
#         pi_01 = pi_dict["p_complier"].detach()
#         q_high = torch.quantile(pi_01, 0.75) 
#         q_low = torch.quantile(pi_01, 0.25)  
        
#         high_mask = pi_01 >= q_high
#         low_mask = pi_01 <= q_low
        
#         if high_mask.any() and low_mask.any():
#             u_high = uplift_pred[high_mask].mean()
#             u_low = uplift_pred[low_mask].mean()
#             margin = config.get("rank_margin", 0.05)
#             rank_loss = F.relu(-(u_high - u_low) + margin)
#             rank_alpha = config.get("rank_alpha", 1.0)
#             total_loss += rank_alpha * rank_loss
#             loss_components["rank_loss"] = (rank_alpha * rank_loss).item()

#     # 3. 方差/极值惩罚
#     if "variance_reg" in loss_types:
#         pi_00 = pi_dict["p_never"].detach()
#         never_threshold = config.get("never_threshold", 0.9)
#         never_mask = pi_00 > never_threshold
#         if never_mask.any():
#             var_penalty = torch.pow(y1_pred[never_mask] - y0_pred[never_mask], 2).mean()
#             var_lambda = config.get("var_lambda", 1.0)
#             total_loss += var_lambda * var_penalty
#             loss_components["var_loss"] = (var_lambda * var_penalty).item()
            
#     return total_loss, loss_components

# ==========================================
# 本地验证脚本 (If __main__)
# ==========================================
if __name__ == "__main__":
    print("🚀 开始验证 losses.py 算子模块...\n")
    torch.manual_seed(42)
    
    batch_size = 8
    logits = torch.randn(batch_size, requires_grad=True)
    targets = torch.randint(0, 2, (batch_size,)).float()
    t = torch.randint(0, 2, (batch_size,)).float()       
    z_c = torch.randn(batch_size, 16, requires_grad=True)
    
    print("-" * 50)
    print("🧪 测试 1: Stage 1 - Focal Loss")
    focal_fn = FocalLoss(alpha=0.25, gamma=2.0)
    print(f"标准 BCE Loss: {F.binary_cross_entropy_with_logits(logits, targets).item():.4f}")
    print(f"Focal Loss:    {focal_fn(logits, targets).item():.4f}")
    
    print("-" * 50)
    print("🧪 测试 2: Stage 1 - MMD Loss (干预不变性)")
    # 打印一下 T 的分布，确认 N1 != N2
    print(f"干预标签 T 的分布: T=1有{(t==1).sum().item()}个, T=0有{(t==0).sum().item()}个")
    z_t1, z_t0 = z_c[t == 1], z_c[t == 0]
    loss_mmd = mmd_loss(z_t1, z_t0, kernel="rbf", sigma=1.0)
    print(f"MMD Loss:      {loss_mmd.item():.4f} (Tensor尺寸不同也能完美计算！)")
    
    print("-" * 50)
    print("🧪 测试 3: Stage 3 - 靶向联合优化魔法台")
    y0_pred = torch.randn(batch_size, requires_grad=True)
    y1_pred = torch.randn(batch_size, requires_grad=True)
    pi_dict = {
        "p_never": torch.tensor([0.95, 0.99, 0.1, 0.1, 0.5, 0.5, 0.5, 0.5]),
        "p_complier": torch.tensor([0.01, 0.01, 0.8, 0.9, 0.2, 0.3, 0.1, 0.2]),
        "p_always": torch.tensor([0.04, 0.0, 0.1, 0.0, 0.3, 0.2, 0.4, 0.3])
    }
    stage3_cfg = {
        "loss_types": ["strata_weighted", "pairwise", "variance_reg"],
        "rank_margin": 0.05, "rank_alpha": 1.0,
        "never_threshold": 0.9, "var_lambda": 2.0
    }
    total_loss, loss_details = compute_stage3_loss(y0_pred, y1_pred, targets, t, pi_dict, stage3_cfg)
    total_loss.backward()
    
    print(f"总靶向 Loss:   {total_loss.item():.4f}")
    for k, v in loss_details.items():
        print(f"  - {k:<10}: {v:.4f}")
    print(f"\n✅ 梯度回传检查: y1_pred.grad 是否存在? -> {y1_pred.grad is not None}")



