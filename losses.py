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

import torch
import torch.nn.functional as F

def sliced_wasserstein_distance(z1, z2, num_projections=128, p=2):
    """
    鲁棒的 Sliced Wasserstein Distance 实现 (完美兼容 N1 != N2)
    z1: (N1, dim) 干预组特征
    z2: (N2, dim) 控制组特征
    """
    # 🛡️ 拦截 1：防止某个组里没人导致 NaN
    if z1.size(0) == 0 or z2.size(0) == 0:
        return torch.tensor(0.0, device=z1.device if z1.device.type != 'cpu' else z2.device)

    dim = z1.size(1)
    device = z1.device

    # 1. 生成随机投影矩阵 (从单位超球面上采样)
    # 🛡️ 拦截 2：必须做 L2 归一化 (dim=0 对应着特征维度)
    projections = torch.randn(dim, num_projections, device=device)
    projections = projections / torch.norm(projections, p=2, dim=0, keepdim=True)

    # 2. 投影到一维空间
    # proj1 shape: (N1, num_projections)
    # proj2 shape: (N2, num_projections)
    proj1 = torch.matmul(z1, projections)
    proj2 = torch.matmul(z2, projections)

    # 3. 独立对每条投影线上的点进行排序
    proj1_sorted, _ = torch.sort(proj1, dim=0)
    proj2_sorted, _ = torch.sort(proj2, dim=0)

    # 4. 核心修复：处理 N1 != N2 的情况
    N1, N2 = proj1.size(0), proj2.size(0)
    if N1 != N2:
        # 转换 shape 适配 interpolate: (batch, channels, length) -> (1, num_projections, N)
        proj1_sorted = proj1_sorted.t().unsqueeze(0)
        proj2_sorted = proj2_sorted.t().unsqueeze(0)

        # 找最大的长度，把短的那个通过线性插值拉长，模拟经验 CDF 的对齐
        target_size = max(N1, N2)
        
        # align_corners=True 需要 length > 1，加个小保护
        align_corners = True if (N1 > 1 and N2 > 1) else False

        proj1_sorted = F.interpolate(proj1_sorted, size=target_size, mode='linear', align_corners=align_corners)
        proj2_sorted = F.interpolate(proj2_sorted, size=target_size, mode='linear', align_corners=align_corners)

        # 转换回原来的 shape: (target_size, num_projections)
        proj1_sorted = proj1_sorted.squeeze(0).t()
        proj2_sorted = proj2_sorted.squeeze(0).t()

    # 5. 计算 Wasserstein 距离 (p-norm)
    # 计算均方误差 (W_2^2)，这比开根号的 W_2 梯度更稳定
    wasserstein_distance = torch.pow(torch.abs(proj1_sorted - proj2_sorted), p).mean()

    return wasserstein_distance

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
    
    elif "prior_conflict" in loss_types:
        pi_01 = pi_dict["p_complier"].detach()
        # 🌟 默认值调整：默认为 "all" 或保持与原有兼容，但通过下面的 alpha=0.0 和 none 强制让其默认不作妖
        mode = config.get("conflict_mode", "all") 
        
        # 🌟 实验 2 & 5：三个惩罚系数解耦，默认全部设为 0.0！不配参数默认完全不加权！
        alpha_wool = config.get("conflict_alpha_wool", 0.0)
        alpha_gold = config.get("conflict_alpha_gold", 0.0)
        alpha_walkin = config.get("conflict_alpha_walkin", 0.0)
        
        # 🌟 实验 3 (Focal)：控制超参，默认设为 "none" (不进行任何动态衰减与截断)
        focal_type = config.get("conflict_focal_type", "none") 
        gamma = config.get("conflict_gamma", 2.0)
        global_margin = config.get("conflict_global_margin", 1.0) 
        
        # 🌟 实验 4 (OHEM)：控制超参，默认强行关闭 (False)
        use_ohem = config.get("conflict_use_ohem", False)
        ohem_pct = config.get("conflict_ohem_pct", 0.10) 
        
        # 🌟 实验 6 (Weight Clip)：控制超参，默认强行关闭 (False)
        use_weight_clip = config.get("conflict_use_weight_clip", False)
        max_weight_thres = config.get("conflict_max_weight_thres", 3.0) 
        
        # 1. 所有人初始权重为 1.0 (标准大盘平稳底座)
        weights = torch.ones_like(targets, dtype=torch.float)
        
        # 2. 精准定位三类因果错位人群
        mask_wool = (treatment == 1) & (targets == 0)    # 羊毛党
        mask_gold = (treatment == 1) & (targets == 1)    # 隐藏金子
        mask_walkin = (treatment == 0) & (targets == 1)  # Walk-in
        
        # 3. 计算纯静态大棒权重 static_weights，并用 has_conflict 记录当前哪些样本真正被当前 mode 激活了
        # 🌟 保持稳定的 Float32 底座
        static_weights = torch.zeros_like(targets, dtype=torch.float)
        has_conflict = torch.zeros_like(targets, dtype=torch.bool)
        
        # 🌟 绝杀修复：在切片赋值右侧统一追加 .float()，强行把 autocast 的 float16 提升为 float32！
        if (mode == "all" or "wool" in mode) and mask_wool.any():
            static_weights[mask_wool] = (alpha_wool * pi_01[mask_wool]).float()
            has_conflict[mask_wool] = True
            
        if (mode == "all" or "gold" in mode) and mask_gold.any():
            static_weights[mask_gold] = (alpha_gold * (1.0 - pi_01[mask_gold])).float()
            has_conflict[mask_gold] = True
            
        if (mode == "all" or "walkin" in mode) and mask_walkin.any():
            static_weights[mask_walkin] = (alpha_walkin * pi_01[mask_walkin]).float()
            has_conflict[mask_walkin] = True
            
        # 4. 🌟 Focal 截断托底逻辑安全加固 🌟
        # 🌟 同样在这里追加 .float()，保证后向计算梯度和权重归一化时数据精度绝对纯净
        p_final = torch.sigmoid(y_pred_observed).detach().float() 
        
        if focal_type == "none":
            weights += static_weights
        elif focal_type == "global_bounded":
            raw_focal_w = 1.0 + static_weights * torch.pow(1.0 - p_final, gamma)
            focal_weights = torch.clamp(raw_focal_w, min=global_margin)
            weights = torch.where(has_conflict, focal_weights, torch.ones_like(weights))
            
        # 5. 核心实现：实验 4 - 在线困难样本挖掘 (OHEM for V10)
        if use_ohem:
            with torch.no_grad():
                q_thresh = torch.quantile(bce_raw.detach(), 1.0 - ohem_pct)
                is_easy_sample = bce_raw < q_thresh
                # 易分样本直接把 V10 加成干掉，退化回权重 1.0
                weights[is_easy_sample] = 1.0

        # 6. 核心实现：实验 6 - 静态权重裁剪 (Weight Clipping)
        if use_weight_clip:
            weights = torch.clamp(weights, max=max_weight_thres)

        # 7. 均值归一化防梯度崩盘
        weights = weights / (weights.mean() + 1e-8)
        conflict_loss = (bce_raw * weights).mean()
        total_loss += conflict_loss
        loss_components["conflict_loss"] = conflict_loss.item()
    else:
        bce_loss = bce_raw.mean()
        total_loss += bce_loss
        loss_components["bce_loss"] = bce_loss.item()

    # elif "prior_conflict" in loss_types:
    #     pi_01 = pi_dict["p_complier"].detach()
    #     mode = config.get("conflict_mode", "all") # 可选: 'wool', 'gold', 'walkin', 'wool_gold', 'wool_walkin', 'gold_walkin', 'all'
        
    #     # 🌟 核心改进：贯彻你的意志，三个关键系数彻底独立拆分，绝不混淆
    #     alpha_wool = config.get("conflict_alpha_wool", 0.0)
    #     alpha_gold = config.get("conflict_alpha_gold", 0.0)
    #     alpha_walkin = config.get("conflict_alpha_walkin", 0.0)
        
    #     # 1. 所有人初始权重为 1.0 (平稳底座)
    #     weights = torch.ones_like(targets, dtype=torch.float)
        
    #     # 2. 精准定位三类因果错位人群
    #     mask_wool = (treatment == 1) & (targets == 0)    # 羊毛党：发了券却不买
    #     mask_gold = (treatment == 1) & (targets == 1)    # 隐藏金子：发了券买了
    #     mask_walkin = (treatment == 0) & (targets == 1)  # Walk-in：没发券自己买了
        
    #     # 3. 🌟 精细化解耦逻辑：只要关键字在 mode 中，或者 mode 为 'all'，就启动对应的靶向加权
    #     # 这样天然支持：单独 ('wool')、两两组合 ('wool_gold') 和全开 ('all')
        
    #     # --- A. 羊毛党加权 ---
    #     if (mode == "all" or "wool" in mode) and mask_wool.any():
    #         # co 越高，打脸越疼
    #         weights[mask_wool] += alpha_wool * pi_01[mask_wool]
            
    #     # --- B. 隐藏金子加权 ---
    #     if (mode == "all" or "gold" in mode) and mask_gold.any():
    #         # co 越低，被模型看扁得越惨
    #         weights[mask_gold] += alpha_gold * (1.0 - pi_01[mask_gold])
            
    #     # --- C. Walk-in 加权 ---
    #     if (mode == "all" or "walkin" in mode) and mask_walkin.any():
    #         # 完美镜像诉求：Walk-in 惩罚方式随 Complier 概率正相关递增
    #         weights[mask_walkin] += alpha_walkin * pi_01[mask_walkin]
            
    #     # 4. 均值归一化：防爆防溢出
    #     weights = weights / (weights.mean() + 1e-8)
    #     conflict_loss = (bce_raw * weights).mean()
    #     total_loss += conflict_loss
    #     loss_components["conflict_loss"] = conflict_loss.item()
    # elif "prior_conflict" in loss_types:
    #     pi_01 = pi_dict["p_complier"].detach()
    #     alpha = config.get("conflict_alpha", 2.0)
    #     mode = config.get("conflict_mode", "both") # 可选：'wool_only', 'gold_only', 'both', 'walkin_v10'
        
    #     # 1. 所有人初始权重为 1.0 (平稳底座)
    #     weights = torch.ones_like(targets, dtype=torch.float)
        
    #     # 2. 精准定位三类因果错位人群
    #     mask_wool = (treatment == 1) & (targets == 0)    # 羊毛党：发了券却不买
    #     mask_gold = (treatment == 1) & (targets == 1)    # 隐藏金子：发了券买了
    #     mask_walkin = (treatment == 0) & (targets == 1)  # 🌟 新增 Walk-in：没发券自己进店买了
        
    #     # 3. 根据调整后的 Mode 动态施加精准惩罚权重
    #     if mode in ["wool_only", "both"] and mask_wool.any():
    #         weights[mask_wool] += alpha * pi_01[mask_wool]
            
    #     if mode in ["gold_only", "both"] and mask_gold.any():
    #         weights[mask_gold] += alpha * (1.0 - pi_01[mask_gold])
            
    #     if mode in ["walkin_v10", "both"] and mask_walkin.any():
    #         # 🌟 完美对齐诉求：Walk-in 惩罚方式和 Wool 镜像一致，随 Complier 概率正相关递增
    #         weights[mask_walkin] += alpha * pi_01[mask_walkin]
            
    #     # 4. 均值归一化：防爆防溢出
    #     weights = weights / (weights.mean() + 1e-8)
        
    #     conflict_loss = (bce_raw * weights).mean()
    #     total_loss += conflict_loss
    #     loss_components["conflict_loss"] = conflict_loss.item()
    # else:
    #     # 纯净的 Baseline BCE
    #     bce_loss = bce_raw.mean()
    #     total_loss += bce_loss
    #     loss_components["bce_loss"] = bce_loss.item()
        
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
            


    return total_loss, loss_components


def compute_mtmt_loss(preds_dict, y, c, t, pi_dict, config, dro_criterion=None):
    """
    MTMT 专属多任务 Loss 引擎。
    从 preds_dict 中解包出主副任务，分别算 Loss 并加权。
    """
    # 1. 解包主任务 (预测 Y)
    y0_main, y1_main = preds_dict["main_task"]
    # 2. 解包辅任务 (预测 C)
    c0_aux, c1_aux = preds_dict["aux_task"]
    
    # 3. 分别调用你原有的算 Loss 逻辑
    loss_main, comp_main = compute_stage3_loss(
        y0_main, y1_main, y, t, pi_dict, config, dro_criterion
    )
    
    loss_aux, comp_aux = compute_stage3_loss(
        c0_aux, c1_aux, c, t, pi_dict, config, dro_criterion
    )
    
    # 4. 多任务权重聚合 (可以在 config 里配 aux_weight)
    aux_weight = config.get("aux_weight", 0.5)
    total_loss = loss_main + aux_weight * loss_aux
    
    # 5. 整理日志，分清楚谁是谁的 loss
    loss_components = {f"main_{k}": v for k, v in comp_main.items()}
    loss_components.update({f"aux_{k}": v for k, v in comp_aux.items()})
    
    return total_loss, loss_components


def compute_ecup_loss(preds_dict, y, c, t, config):
    """
    ECUP 全链路联合损失函数 (Entire Chain Loss)
    [修复版]: 完美兼容 PyTorch AMP (autocast) 的数值安全实现
    """
    # 1. 解析 Logits
    c0_logit, c1_logit = preds_dict["c_logits"]
    cvr0_logit, cvr1_logit = preds_dict["cvr_logits"]
    
    # 2. 根据真实的 T 提取当前观测到的 Logit
    c_logit_obs = torch.where(t == 1, c1_logit, c0_logit)
    cvr_logit_obs = torch.where(t == 1, cvr1_logit, cvr0_logit)
    
    # ==========================================
    # 🌟 修复 1: CTR Loss (主 C 任务)
    # 直接使用带 logits 的内置函数，PyTorch 底层自动保证数值稳定性
    # ==========================================
    loss_c = F.binary_cross_entropy_with_logits(c_logit_obs, c.float(), reduction='mean')
    
    # ==========================================
    # 🌟 修复 2: CTCVR Loss (主 Y 任务)
    # 概率乘积无法直接使用内置 logits 函数。
    # 手动实现交叉熵，并使用 clamp 限制极值，彻底绕过 autocast 报错并防止 NaN
    # ==========================================
    p_c = torch.sigmoid(c_logit_obs)
    p_cvr = torch.sigmoid(cvr_logit_obs)
    
    # 计算联合概率 pCTCVR = pCTR * pCVR
    p_y = p_c * p_cvr 
    
    # 截断极值：防止概率变成绝对的 0 或 1 导致 log(0) 抛出 NaN
    p_y = torch.clamp(p_y, min=1e-7, max=1.0 - 1e-7)
    
    # 手动计算 BCE Loss (等价于 F.binary_cross_entropy)
    loss_y = -(y.float() * torch.log(p_y) + (1.0 - y.float()) * torch.log(1.0 - p_y)).mean()
    
    # 3. 计算多目标联合 Loss
    ctcvr_weight = config.get("ctcvr_weight", 1.0)
    total_loss = loss_c + ctcvr_weight * loss_y
    
    loss_components = {
        "ecup_ctr_loss": loss_c.item(),
        "ecup_ctcvr_loss": loss_y.item()
    }
    
    return total_loss, loss_components


# ==========================================
# 🌟 新增：Naive Multitask 联合 Loss
# ==========================================
def compute_naive_mt_loss(preds_dict, y, c, t, pi_dict, config, dro_criterion=None):
    """
    TARNET_MT 专属 Loss 引擎：直接利用 BCE (或其它配置好的 Stage 3 Loss) 对双任务同时反向传播
    """
    # 1. 主任务 Y Loss 
    y0_main, y1_main = preds_dict["main_task"]
    loss_y, comp_y = compute_stage3_loss(y0_main, y1_main, y, t, pi_dict, config, dro_criterion)
    
    # 2. 辅任务 C Loss (预测他本身是不是 Complier / Clicker)
    c0_aux, c1_aux = preds_dict["aux_task"]
    # 辅助任务强制走纯纯的 BCE，防止受到主任务 rank/variance 等特定正则项的影响
    c_pred_observed = torch.where(t == 1, c1_aux, c0_aux)
    loss_c = F.binary_cross_entropy_with_logits(c_pred_observed, c.float(), reduction='mean')
    
    # 3. 静态加权聚合
    mt_c_weight = config.get("mt_c_weight", 0.5) # 默认给 C 任务一半的权重
    total_loss = loss_y + mt_c_weight * loss_c
    
    # 4. 整理并吐出组件日志
    loss_components = {f"y_{k}": v for k, v in comp_y.items()}
    loss_components["c_bce_loss"] = loss_c.item()
    
    return total_loss, loss_components

def compute_motto_loss(preds_dict, y, c, t, config):
    """
    MOTTO 联合损失函数
    包含：事实预测误差 (Factual Risk) + 选择性分布对齐 (SDA)
    """
    # 1. 提取预测 Logits
    y0_main, y1_main = preds_dict["main_task"]
    y0_aux, y1_aux = preds_dict["aux_task"]
    
    # 2. 提取需要进行对齐的“干预共享表征”
    sda_reps = preds_dict.get("sda_reps", {})
    trt_sh_main = sda_reps.get("main")
    trt_sh_aux = sda_reps.get("aux")
    
    # ---------------------------------------------------------
    # 🌟 模块 A: Factual Risk (事实预测误差)
    # ---------------------------------------------------------
    # 获取观测到的真实 Logit
    obs_main = torch.where(t == 1, y1_main, y0_main)
    obs_aux = torch.where(t == 1, y1_aux, y0_aux)
    
    # 计算二元交叉熵 (AMP 安全)
    loss_main = F.binary_cross_entropy_with_logits(obs_main, y.float(), reduction='mean')
    loss_aux = F.binary_cross_entropy_with_logits(obs_aux, c.float(), reduction='mean')
    
    # ---------------------------------------------------------
    # 🌟 模块 B: Selective Distribution Alignment (SDA) Loss
    # 目标：让 T=0 和 T=1 的干预共享表征向全局平均分布靠拢
    # ---------------------------------------------------------
    sda_loss = torch.tensor(0.0, device=y.device)
    alpha_sda = config.get("alpha_sda", 0.1) # SDA 权重超参
    
    if alpha_sda > 0 and trt_sh_main is not None:
        mask_t0 = (t == 0)
        mask_t1 = (t == 1)
        
        # 确保 Batch 中同时存在两组样本
        if mask_t0.sum() > 0 and mask_t1.sum() > 0:
            # Main 任务表征对齐 (一阶 MMD)
            mean_main_all = trt_sh_main.mean(dim=0)
            sda_main_t0 = F.mse_loss(trt_sh_main[mask_t0].mean(dim=0), mean_main_all)
            sda_main_t1 = F.mse_loss(trt_sh_main[mask_t1].mean(dim=0), mean_main_all)
            
            # Aux 任务表征对齐
            mean_aux_all = trt_sh_aux.mean(dim=0)
            sda_aux_t0 = F.mse_loss(trt_sh_aux[mask_t0].mean(dim=0), mean_aux_all)
            sda_aux_t1 = F.mse_loss(trt_sh_aux[mask_t1].mean(dim=0), mean_aux_all)
            
            sda_loss = sda_main_t0 + sda_main_t1 + sda_aux_t0 + sda_aux_t1

    # 3. 总体 Loss 加权求和
    aux_weight = config.get("aux_weight", 1.0)
    total_loss = loss_main + aux_weight * loss_aux + alpha_sda * sda_loss
    
    loss_components = {
        "motto_main_loss": loss_main.item(),
        "motto_aux_loss": loss_aux.item(),
        "motto_sda_loss": sda_loss.item() if isinstance(sda_loss, torch.Tensor) else 0.0
    }
    
    return total_loss, loss_components


def compute_dragonnet_loss(y0_pred, y1_pred, targets, treatment, pi_dict, config):
    """
    DragonNet 专属损失函数: 包含 Y 的预测、T 的预测，以及 TMLE 靶向正则化
    """
    # 1. 事实数据的标准 BCE
    y_pred_obs = torch.where(treatment == 1, y1_pred, y0_pred)
    loss_y = F.binary_cross_entropy_with_logits(y_pred_obs, targets.float())
    
    # 2. 倾向分 T 的 BCE
    t_logit = pi_dict["t_logit"]
    loss_t = F.binary_cross_entropy_with_logits(t_logit, treatment.float())
    
    # 3. TMLE 靶向正则化 (Targeted Regularization)
    propensity = torch.sigmoid(t_logit).detach() # 阻断梯度回传到 propensity head
    propensity = torch.clamp(propensity, 1e-4, 1.0 - 1e-4) # 防止分母为 0
    
    eps = pi_dict["epsilon"]
    
    # 计算扰动项 H(X)
    perturb = (treatment.float() / propensity) - ((1.0 - treatment.float()) / (1.0 - propensity))
    
    # 叠加扰动项计算新的 Loss
    y_pred_perturbed = y_pred_obs + eps * perturb
    loss_reg = F.binary_cross_entropy_with_logits(y_pred_perturbed, targets.float())
    
    alpha = config.get("dragon_alpha", 1.0)
    beta = config.get("dragon_beta", 1.0)
    
    total_loss = loss_y + alpha * loss_t + beta * loss_reg
    
    loss_components = {
        "dragon_loss_y": loss_y.item(), 
        "dragon_loss_t": (alpha * loss_t).item(), 
        "dragon_loss_reg": (beta * loss_reg).item()
    }
    
    return total_loss, loss_components
def compute_efin_loss(y0_pred, y1_pred, targets, treatment, pi_dict, config):
    """
    EFIN 专属损失函数: 主任务 Loss + 组别反转约束 Loss
    """
    # 1. 主任务标准 BCE (只用观测到的 T 对应的预测值)
    y_pred_obs = torch.where(treatment == 1, y1_pred, y0_pred)
    loss_y = F.binary_cross_entropy_with_logits(y_pred_obs, targets.float())
    
    # 2. Intervention Constraint Loss (Paper Eq 14)
    # 论文要求使用 inverse label (1 - T) 进行训练，以制造干扰，平衡特征分布
    t_constraint_logit = pi_dict["efin_t_logit"] # ["efin_constraint_logit"]
    inverse_treatment = 1.0 - treatment.float()
    loss_c = F.binary_cross_entropy_with_logits(t_constraint_logit, inverse_treatment)
    
    # 3. 总 Loss 融合
    # lam = config.get("efin_lambda", 0.01) # 权重默认给个 0.01
    total_loss = loss_y + loss_c
    
    loss_components = {
        "efin_loss_y": loss_y.item(), 
        "efin_loss_c": (loss_c).item()
    }
    
    return total_loss, loss_components
def compute_descn_loss(mu0_logit, mu1_logit, y, t, pi_dict, trial_cfg):
    """
    DESCN 专属损失预估引擎 (防自适应混合精度崩溃全防御版)
    """
    device = mu0_logit.device
    pi_logit = pi_dict["descn_pi_logit"]
    tau_logit = pi_dict["descn_tau_logit"]
    
    # 1. L_π: 倾向分基本分类损失 (Safe for autocast)
    loss_pi = F.binary_cross_entropy_with_logits(pi_logit, t.float())
    
    # ----------------------------------------------------
    # 🌟 核心防御：退出混合精度，强制切入全精度 float32 计算乘积 ESN 损失
    # ----------------------------------------------------
    with torch.cuda.amp.autocast(enabled=False):
        # 强转为 float32 避免半精度溢出
        p_pi = torch.clamp(torch.sigmoid(pi_logit.float()), min=0.001, max=0.999)
        p_mu1 = torch.sigmoid(mu1_logit.float())
        p_mu0 = torch.sigmoid(mu0_logit.float())
        
        # 2. L_ESTR & L_ESCR: 联合事件概率乘积
        p_estr = p_pi * p_mu1
        p_escr = (1.0 - p_pi) * p_mu0
        
        label_estr = (y == 1) & (t == 1)
        label_escr = (y == 1) & (t == 0)
        
        # 强转为 float32 执行纯粹的 binary_cross_entropy，消灭 autocast 报错
        loss_estr = F.binary_cross_entropy(p_estr, label_estr.float())
        loss_escr = F.binary_cross_entropy(p_escr, label_escr.float())
    
    # 3. L_CrossTR & L_CrossCR: 交叉反向样本掩码约束损失 (Safe for autocast)
    treat_mask = (t == 1)
    control_mask = (t == 0)
    
    # CrossTR: mean_{W=1} BCE(mu0_logit + tau_logit, Y)
    if treat_mask.sum() > 0:
        cross_tr_pred = mu0_logit[treat_mask] + tau_logit[treat_mask]
        loss_crosstr = F.binary_cross_entropy_with_logits(cross_tr_pred, y[treat_mask].float())
    else:
        loss_crosstr = torch.tensor(0.0, device=device)
        
    # CrossCR: mean_{W=0} BCE(mu1_logit - tau_logit, Y)
    if control_mask.sum() > 0:
        cross_cr_pred = mu1_logit[control_mask] - tau_logit[control_mask]
        loss_crosscr = F.binary_cross_entropy_with_logits(cross_cr_pred, y[control_mask].float())
    else:
        loss_crosscr = torch.tensor(0.0, device=device)

    # 4. 融合调参权重组合
    w_alpha = trial_cfg.get("descn_alpha", 0.5)
    w_beta1 = trial_cfg.get("descn_beta1", 1.0)
    w_beta0 = trial_cfg.get("descn_beta0", 1.0)
    w_gamma1 = trial_cfg.get("descn_gamma1", 0.5)  
    w_gamma0 = trial_cfg.get("descn_gamma0", 0.1)
    
    total_loss = (w_alpha * loss_pi + 
                  w_beta1 * loss_estr + 
                  w_beta0 * loss_escr + 
                  w_gamma1 * loss_crosstr + 
                  w_gamma0 * loss_crosscr)
    
    loss_comp = {
        "loss_pi": loss_pi.item(),
        "loss_estr": loss_estr.item(),
        "loss_escr": loss_escr.item(),
        "loss_crosstr": loss_crosstr.item(),
        "loss_crosscr": loss_crosscr.item()
    }
    
    return total_loss, loss_comp

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



