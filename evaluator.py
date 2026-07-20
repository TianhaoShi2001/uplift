import torch
import numpy as np
import pandas as pd
import os
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score
from tqdm import tqdm

def get_perfect_uplift_order(y_true, t):
    """👉 生成理论上的完美排序，用于归一化 AUUC"""
    # 完美排序优先级：
    # 1. T=1 且 Y=1 (被干预且转化，最想捞的人)
    # 2. T=0 且 Y=0 (没干预没转化，也可以干预)
    # 3. T=1 且 Y=0 (干预了没转化，白费钱)
    # 4. T=0 且 Y=1 (自然转化，绝对不要干预)
    score = np.zeros_like(y_true, dtype=float)
    score[(t == 1) & (y_true == 1)] = 4
    score[(t == 0) & (y_true == 0)] = 3
    score[(t == 1) & (y_true == 0)] = 2
    score[(t == 0) & (y_true == 1)] = 1
    # 加上微小随机噪声防止 argsort 处理相同分数时导致曲线阶梯不平滑
    score += np.random.uniform(0, 0.1, size=len(score))
    return np.argsort(score)[::-1]

def calculate_uplift_metrics(target_true, uplift_pred, t, prefix="Target"):
    """👉 还原原版：计算曲线面积 (横纵坐标归一化，随机基线为 0.5)"""
    target_true, uplift_pred, t = np.array(target_true), np.array(uplift_pred), np.array(t)
    if len(target_true) == 0: return {}

    # 1. 按预测分数降序排列
    order = np.argsort(uplift_pred)[::-1]
    y_true_sorted = target_true[order]
    t_sorted = t[order]

    # 2. 累加统计
    n_t_cum = np.cumsum(t_sorted == 1)
    n_c_cum = np.cumsum(t_sorted == 0)
    y_t_cum = np.cumsum(y_true_sorted * (t_sorted == 1))
    y_c_cum = np.cumsum(y_true_sorted * (t_sorted == 0))

    # 安全除法保护
    n_t_safe = np.where(n_t_cum == 0, 1e-6, n_t_cum)
    n_c_safe = np.where(n_c_cum == 0, 1e-6, n_c_cum)

    # 横轴 (0 到 1)
    x_axis = np.arange(1, len(target_true) + 1) / len(target_true)

    # ==================================
    # 🌟 你的原味 AUUC
    # ==================================
    with np.errstate(invalid='ignore', divide='ignore'):
        uplift_curve = (y_t_cum / n_t_safe - y_c_cum / n_c_safe) * (n_t_cum + n_c_cum)
    uplift_curve = np.nan_to_num(uplift_curve)
    
    # 除以终点，拉平到 1.0
    if abs(uplift_curve[-1]) > 1e-10:
        uplift_curve = uplift_curve / uplift_curve[-1]
    
    auuc = np.trapz(uplift_curve, x=x_axis)

    # ==================================
    # 🌟 你的原味 AUQC (Qini)
    # ==================================
    with np.errstate(invalid='ignore', divide='ignore'):
        qini_curve = y_t_cum - (y_c_cum * n_t_safe / n_c_safe)
    qini_curve = np.nan_to_num(qini_curve)
    
    # 除以终点，拉平到 1.0
    if abs(qini_curve[-1]) > 1e-10:
        qini_curve = qini_curve / qini_curve[-1]
        
    auqc = np.trapz(qini_curve, x=x_axis)

    # ==================================
    # Lift@K 业务核心指标 (这玩意做不了假)
    # ==================================
    def get_lift_at_k(k):
        idx = int(len(target_true) * k / 100.0)
        if idx == 0: return 0.0
        cr_t = np.sum(y_true_sorted[:idx] * (t_sorted[:idx] == 1)) / (np.sum(t_sorted[:idx] == 1) + 1e-6)
        cr_c = np.sum(y_true_sorted[:idx] * (t_sorted[:idx] == 0)) / (np.sum(t_sorted[:idx] == 0) + 1e-6)
        return cr_t - cr_c
        
    return {
        f"{prefix}_AUUC": float(auuc), 
        f"{prefix}_AUQC": float(auqc), 
        f"{prefix}_Lift@10": float(get_lift_at_k(10)),
        f"{prefix}_Lift@30": float(get_lift_at_k(30))
    }

def calc_class_and_calib(target_true, y_prob, prefix="Target"):
    """👉 补充 Recall, Precision 等值准指标"""
    if len(target_true) == 0: return {}
    target_true, y_prob = np.array(target_true), np.array(y_prob)
    preds = (y_prob > 0.5).astype(int)
    
    metrics = {
        f"{prefix}_PCOC": float(y_prob.mean() / (target_true.mean() + 1e-6)),
        f"{prefix}_MAE": float(np.abs(y_prob - target_true).mean()),
        f"{prefix}_Recall": float(recall_score(target_true, preds, zero_division=0)),
        f"{prefix}_Precision": float(precision_score(target_true, preds, zero_division=0)),
        f"{prefix}_AUC": float(roc_auc_score(target_true, y_prob) if len(np.unique(target_true)) > 1 else 0.5)
    }
    return metrics


# after 0420

import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
# 假设你的 metrics 计算函数都已正常 import
# from utils.metrics import calculate_uplift_metrics, calc_class_and_calib

@torch.no_grad()
def evaluate_and_dump(model, loader, device, task_name, save_path=None, max_steps=float('inf'), show_pbar=False):
    model.eval()
    
    t_all, y_all, c_all = [], [], []
    prob0_all, prob1_all, pi_01_all = [], [], []
    c0_prob_all, c1_prob_all = [], [] 
    
    # 🌟 新增：专门装白盒私货的小推车
    white_box_collections = {}

    pbar_loader = tqdm(loader, desc="🔍 Evaluating", leave=False) if show_pbar else loader
    
    for i, batch in enumerate(pbar_loader):
        if i >= max_steps: break
            
        # 兼容 4 元素或 5 元素 batch
        x_cont, x_cat, t, y = batch[0], batch[1], batch[2], batch[3]
        c = batch[4] if len(batch) == 5 else torch.zeros_like(y)
            
        x_cont = x_cont.to(device) if x_cont is not None else None
        x_cat = {k: v.to(device) for k, v in x_cat.items()} if x_cat is not None else None
        
        # ==========================================
        # 统一模型推理
        # ==========================================
        out = model(x_cont, x_cat)
        pi_dict = {}

        if task_name == "train_y":
            # 1. 统一安全提取 pi_dict (不论 out 是字典还是元组)
            pi_dict = out.get("pi_dict", {}) if isinstance(out, dict) else (out[2] if len(out) > 2 else {})
            
            # 2. 扁平化分发主、副任务预测
            if isinstance(out, dict) and out.get("ecup_0717new"):
                # ECUP_0717new 分支：只在这里 sigmoid 一次，得到真正的 pCTR * pCVR，
                # 避免重蹈原版 ECUP 分支 "预乘概率又被下面通用收尾再 sigmoid 一次" 的覆辙
                # （那个 bug 会把 out_0 压到 [0.5, 0.73] 区间）。
                c0_logit, c1_logit = out["c_logits"]
                cvr0_logit, cvr1_logit = out["cvr_logits"]
                
                c_out_0 = torch.sigmoid(c0_logit)
                c_out_1 = torch.sigmoid(c1_logit)
                
                out_0 = c_out_0 * torch.sigmoid(cvr0_logit)
                out_1 = c_out_1 * torch.sigmoid(cvr1_logit)
                
                c0_prob_all.append(c_out_0.cpu().numpy())
                c1_prob_all.append(c_out_1.cpu().numpy())
                already_prob = True

            elif isinstance(out, dict) and "c_logits" in out:
                # 🌿 ECUP 分支
                c0_logit, c1_logit = out["c_logits"]
                c_out_0, c_out_1 = torch.sigmoid(c0_logit), torch.sigmoid(c1_logit)
                out_0 = c_out_0 * torch.sigmoid(out["cvr_logits"][0])
                out_1 = c_out_1 * torch.sigmoid(out["cvr_logits"][1])
                c0_prob_all.append(c_out_0.cpu().numpy())
                c1_prob_all.append(c_out_1.cpu().numpy())
                
            elif isinstance(out, dict) and "main_task" in out:
                # 🌿 MTMT / MOTTO 多任务通用分支
                out_0, out_1 = out["main_task"]
                if "aux_task" in out:
                    c0_logit, c1_logit = out["aux_task"]
                    c0_prob_all.append(torch.sigmoid(c0_logit).cpu().numpy())
                    c1_prob_all.append(torch.sigmoid(c1_logit).cpu().numpy())
                    
            else:
                # 🌿 原版 TARNET 及其变体分支
                out_0, out_1 = out[0], out[1]
                
            # 提取先验 (如果没有则填 0)
            pi_01 = pi_dict.get("p_complier", torch.zeros_like(out_0)).detach().cpu().numpy()
            
        else:
            # 🌿 训练 C 模型的简单分支
            out_0, out_1 = out[0], out[1]
            pi_01 = np.zeros(len(t))

        # ==========================================
        # 🌟 白盒私货收集 (只抓取带 wb_ 前缀的 key)
        # ==========================================
        for k, v in pi_dict.items():
            if str(k).startswith("wb_"):
                if k not in white_box_collections:
                    white_box_collections[k] = []
                white_box_collections[k].append(v.detach().cpu().numpy())

        # ==========================================
        # 常规数据收集
        # ==========================================
        prob0_all.append(torch.sigmoid(out_0).cpu().numpy())
        prob1_all.append(torch.sigmoid(out_1).cpu().numpy())
        pi_01_all.append(pi_01)
        t_all.append(t.cpu().numpy())
        y_all.append(y.cpu().numpy())
        c_all.append(c.cpu().numpy())
        
    if len(t_all) == 0: return {}, pd.DataFrame()

    # 向量化拼接
    t_np = np.concatenate(t_all)
    y_np = np.concatenate(y_all)
    c_np = np.concatenate(c_all)
    prob0_np = np.concatenate(prob0_all)
    prob1_np = np.concatenate(prob1_all)
    pi_01_np = np.concatenate(pi_01_all)
    uplift_pred = prob1_np - prob0_np
    observed_prob = np.where(t_np == 1, prob1_np, prob0_np)
    
    # 指标计算 (这里假设你已经 import 了对应函数)
    metrics = {}
    metrics.update(calculate_uplift_metrics(y_np, uplift_pred, t_np, prefix="Target_Y"))
    metrics.update(calc_class_and_calib(y_np, observed_prob, prefix="Target_Y"))
    metrics.update(calculate_uplift_metrics(c_np, uplift_pred, t_np, prefix="Target_C"))
    metrics.update(calc_class_and_calib(c_np, observed_prob, prefix="Target_C"))

    # 组装基础 DataFrame
    df = pd.DataFrame({
        "t": t_np, "y_true": y_np, "c_true": c_np,
        "y0_prob": prob0_np, "y1_prob": prob1_np,
        "uplift_pred": uplift_pred, "pi_01_prior": pi_01_np
    })
    
    # 拼接多任务 C 概率
    if len(c0_prob_all) > 0:
        c0_np = np.concatenate(c0_prob_all)
        c1_np = np.concatenate(c1_prob_all)
        df["c0_prob"] = c0_np
        df["c1_prob"] = c1_np
        df["c_uplift_pred"] = c1_np - c0_np

    # ==========================================
    # 🌟 白盒物资智能分发 (The Smart Dispatcher)
    # ==========================================
    embeddings_dict = {} # 专门装高维矩阵的容器
    
    for k, v_list in white_box_collections.items():
        stacked_v = np.concatenate(v_list, axis=0)
        
        # 维度探测：一维/低维存 CSV，高维存 NPZ
        if stacked_v.ndim == 1:
            df[k] = stacked_v
        elif stacked_v.ndim == 2 and stacked_v.shape[1] <= 10:
            for dim in range(stacked_v.shape[1]):
                df[f"{k}_dim{dim}"] = stacked_v[:, dim]
        else:
            embeddings_dict[k] = stacked_v

    # ==========================================
    # 双轨制落盘
    # ==========================================
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # 标量落盘 CSV
        df.to_csv(save_path, index=False)
        
        # 高维张量独立落盘 Numpy (解耦防止卡死)
        if embeddings_dict:
            npy_path = save_path.replace(".csv", "_embeddings.npz")
            np.savez_compressed(npy_path, **embeddings_dict)
            print(f"📦 高维白盒特征已独立存至: {npy_path}")
        
    return metrics, df


# before 0420
# @torch.no_grad()
# def evaluate_and_dump(model, loader, device, task_name, save_path=None, max_steps=float('inf'), show_pbar=False):
#     model.eval()
    
#     t_all, y_all, c_all = [], [], []
#     prob0_all, prob1_all, pi_01_all = [], [], []
    
#     # 🌟 新增：准备专门存多任务的 C 预测
#     c0_prob_all, c1_prob_all = [], [] 

#     pbar_loader = tqdm(loader, desc="🔍 Evaluating", leave=False) if show_pbar else loader
    
#     for i, batch in enumerate(pbar_loader):
#         if i >= max_steps: break
            
#         if len(batch) == 5: 
#             x_cont, x_cat, t, y, c = batch
#         else: 
#             x_cont, x_cat, t, y = batch
#             c = torch.zeros_like(y)
            
#         x_cont = x_cont.to(device) if x_cont is not None else None
#         x_cat = {k: v.to(device) for k, v in x_cat.items()} if x_cat is not None else None
        
#         if task_name == "train_y":
#             out = model(x_cont, x_cat)
#             pi_dict = {} 

#             if isinstance(out, dict) and "c_logits" in out:
#                 # ==============================
#                 # 🌟 ECUP 分支
#                 # ==============================
#                 c0_logit, c1_logit = out["c_logits"]
#                 cvr0_logit, cvr1_logit = out["cvr_logits"]
                
#                 # C 任务概率
#                 c_out_0 = torch.sigmoid(c0_logit)
#                 c_out_1 = torch.sigmoid(c1_logit)
                
#                 # Y 任务 (CTCVR) 概率 = pCTR * pCVR
#                 out_0 = c_out_0 * torch.sigmoid(cvr0_logit)
#                 out_1 = c_out_1 * torch.sigmoid(cvr1_logit)
                
#                 c0_prob_all.append(c_out_0.cpu().numpy())
#                 c1_prob_all.append(c_out_1.cpu().numpy())
                
#                 # 🌟 修复点：一定要把字典里的 pi_dict 捞出来
#                 pi_dict = out.get("pi_dict", {}) 
                
#             elif isinstance(out, dict) and "main_task" in out:
#                 # ==============================
#                 # 🌟 MTMT & TARNET_MT 多任务通用分支
#                 # ==============================
#                 out_0, out_1 = out["main_task"]
#                 c0_logit, c1_logit = out["aux_task"]
                
#                 # 显式抓取 C 任务概率，用于后续评估和落盘
#                 c_out_0 = torch.sigmoid(c0_logit)
#                 c_out_1 = torch.sigmoid(c1_logit)
#                 c0_prob_all.append(c_out_0.cpu().numpy())
#                 c1_prob_all.append(c_out_1.cpu().numpy())
                
#                 pi_dict = out.get("pi_dict", {})
                
#             else:
#                 # ==============================
#                 # 🌟 原版 TARNET 及其变体分支
#                 # ==============================
#                 out_0, out_1, pi_dict = out
                
#             pi_01 = pi_dict.get("p_complier", torch.zeros_like(out_0)).cpu().numpy()
#         else:
#             out_0, out_1, _ = model(x_cont, x_cat)
#             pi_01 = np.zeros(len(t))
            
#         prob0_all.append(torch.sigmoid(out_0).cpu().numpy())
#         prob1_all.append(torch.sigmoid(out_1).cpu().numpy())
#         pi_01_all.append(pi_01)
        
#         t_all.append(t.cpu().numpy())
#         y_all.append(y.cpu().numpy())
#         c_all.append(c.cpu().numpy())
        
#     if len(t_all) == 0: return {}, pd.DataFrame()

#     # 向量化拼接
#     t_np = np.concatenate(t_all)
#     y_np = np.concatenate(y_all)
#     c_np = np.concatenate(c_all)
#     prob0_np = np.concatenate(prob0_all)
#     prob1_np = np.concatenate(prob1_all)
#     pi_01_np = np.concatenate(pi_01_all)
#     uplift_pred = prob1_np - prob0_np
#     observed_prob = np.where(t_np == 1, prob1_np, prob0_np)
    
#     metrics = {}
#     metrics.update(calculate_uplift_metrics(y_np, uplift_pred, t_np, prefix="Target_Y"))
#     metrics.update(calc_class_and_calib(y_np, observed_prob, prefix="Target_Y"))
    
#     metrics.update(calculate_uplift_metrics(c_np, uplift_pred, t_np, prefix="Target_C"))
#     metrics.update(calc_class_and_calib(c_np, observed_prob, prefix="Target_C"))

#     # 1. 组装你原汁原味的 DataFrame（没有任何破坏）
#     df = pd.DataFrame({
#         "t": t_np, "y_true": y_np, "c_true": c_np,
#         "y0_prob": prob0_np, "y1_prob": prob1_np,
#         "uplift_pred": uplift_pred, "pi_01_prior": pi_01_np
#     })
    
#     # 🌟 2. 核心追加逻辑：如果刚才存到了 C 的预测，就在 df 后面直接加三列
#     if len(c0_prob_all) > 0:
#         c0_np = np.concatenate(c0_prob_all)
#         c1_np = np.concatenate(c1_prob_all)
#         df["c0_prob"] = c0_np
#         df["c1_prob"] = c1_np
#         df["c_uplift_pred"] = c1_np - c0_np

#     if save_path:
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)
#         df.to_csv(save_path, index=False)
        
#     return metrics, df

# @torch.no_grad()
# def evaluate_and_dump(model, loader, device, task_name, save_path=None, max_steps=float('inf'), show_pbar=False):
#     model.eval()
    
#     # 🌟 核心提速点：丢弃 for 循环 append 字典，直接装进 list 之后 concat
#     t_all, y_all, c_all = [], [], []
#     prob0_all, prob1_all, pi_01_all = [], [], []
    
#     pbar_loader = tqdm(loader, desc="🔍 Evaluating", leave=False) if show_pbar else loader
    
#     for i, batch in enumerate(pbar_loader):
#         if i >= max_steps: break
            
#         if len(batch) == 5: 
#             x_cont, x_cat, t, y, c = batch
#         else: 
#             x_cont, x_cat, t, y = batch
#             c = torch.zeros_like(y)
            
#         x_cont = x_cont.to(device) if x_cont is not None else None
#         x_cat = {k: v.to(device) for k, v in x_cat.items()} if x_cat is not None else None
        
#         if task_name == "train_y":
#             out_0, out_1, pi_dict = model(x_cont, x_cat)
#             pi_01 = pi_dict.get("p_complier", torch.zeros_like(out_0)).cpu().numpy()
#         else:
#             out_0, out_1, _ = model(x_cont, x_cat)
#             pi_01 = np.zeros(len(t))
            
#         prob0_all.append(torch.sigmoid(out_0).cpu().numpy())
#         prob1_all.append(torch.sigmoid(out_1).cpu().numpy())
#         pi_01_all.append(pi_01)
        
#         t_all.append(t.cpu().numpy())
#         y_all.append(y.cpu().numpy())
#         c_all.append(c.cpu().numpy())
        
#     if len(t_all) == 0: return {}, pd.DataFrame()

#     # 向量化拼接
#     t_np = np.concatenate(t_all)
#     y_np = np.concatenate(y_all)
#     c_np = np.concatenate(c_all)
#     prob0_np = np.concatenate(prob0_all)
#     prob1_np = np.concatenate(prob1_all)
#     pi_01_np = np.concatenate(pi_01_all)
#     uplift_pred = prob1_np - prob0_np
#     observed_prob = np.where(t_np == 1, prob1_np, prob0_np)
    
#     metrics = {}
    
#     # 根据你的要求，清晰前缀分类。无论你此时跑的是 Y 还是 C 模型，我们用前缀区分清晰。
#     # 比如当前在跑 task_y，那最受关注的就是 Target_Y
#     metrics.update(calculate_uplift_metrics(y_np, uplift_pred, t_np, prefix="Target_Y"))
#     metrics.update(calc_class_and_calib(y_np, observed_prob, prefix="Target_Y"))
    
#     metrics.update(calculate_uplift_metrics(c_np, uplift_pred, t_np, prefix="Target_C"))
#     metrics.update(calc_class_and_calib(c_np, observed_prob, prefix="Target_C"))

#     # 落盘还是留着，画图做验证方便
#     df = pd.DataFrame({
#         "t": t_np, "y_true": y_np, "c_true": c_np,
#         "y0_prob": prob0_np, "y1_prob": prob1_np,
#         "uplift_pred": uplift_pred, "pi_01_prior": pi_01_np
#     })
    
#     if save_path:
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)
#         df.to_csv(save_path, index=False)
        
#     return metrics, df