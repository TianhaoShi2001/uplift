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

@torch.no_grad()
def evaluate_and_dump(model, loader, device, task_name, save_path=None, max_steps=float('inf'), show_pbar=False):
    model.eval()
    
    # 🌟 核心提速点：丢弃 for 循环 append 字典，直接装进 list 之后 concat
    t_all, y_all, c_all = [], [], []
    prob0_all, prob1_all, pi_01_all = [], [], []
    
    pbar_loader = tqdm(loader, desc="🔍 Evaluating", leave=False) if show_pbar else loader
    
    for i, batch in enumerate(pbar_loader):
        if i >= max_steps: break
            
        if len(batch) == 5: 
            x_cont, x_cat, t, y, c = batch
        else: 
            x_cont, x_cat, t, y = batch
            c = torch.zeros_like(y)
            
        x_cont = x_cont.to(device) if x_cont is not None else None
        x_cat = {k: v.to(device) for k, v in x_cat.items()} if x_cat is not None else None
        
        if task_name == "train_y":
            preds = model(x_cont, x_cat)
            if isinstance(preds, dict) and "main_task" in preds:
                out_0, out_1 = preds["main_task"]
                pi_dict = preds.get("pi_dict", {})
            else:
                out_0, out_1, pi_dict = preds
            pi_01 = pi_dict.get("p_complier", torch.zeros_like(out_0)).cpu().numpy()
        else:
            out_0, out_1, _ = model(x_cont, x_cat)
            pi_01 = np.zeros(len(t))
            
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
    
    metrics = {}
    
    # 根据你的要求，清晰前缀分类。无论你此时跑的是 Y 还是 C 模型，我们用前缀区分清晰。
    # 比如当前在跑 task_y，那最受关注的就是 Target_Y
    metrics.update(calculate_uplift_metrics(y_np, uplift_pred, t_np, prefix="Target_Y"))
    metrics.update(calc_class_and_calib(y_np, observed_prob, prefix="Target_Y"))
    
    metrics.update(calculate_uplift_metrics(c_np, uplift_pred, t_np, prefix="Target_C"))
    metrics.update(calc_class_and_calib(c_np, observed_prob, prefix="Target_C"))

    # 落盘还是留着，画图做验证方便
    df = pd.DataFrame({
        "t": t_np, "y_true": y_np, "c_true": c_np,
        "y0_prob": prob0_np, "y1_prob": prob1_np,
        "uplift_pred": uplift_pred, "pi_01_prior": pi_01_np
    })
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        
    return metrics, df