import os
import time
import json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm 

# 🌟 引入 AMP (自动混合精度) 核心组件
from torch.cuda.amp import autocast, GradScaler

from data_pipeline import UpliftDataset, uplift_collate_fn
from models import *
from losses import *
from evaluator import evaluate_and_dump

# ==========================================
# 🛡️ 史上最稳的 Ray 汇报器
# ==========================================
def safe_report(metrics):
    try:
        from ray import train
        if train.get_context().get_trial_name(): 
            train.report(metrics)
            return
    except Exception:
        pass
    try:
        from ray import tune
        tune.report(**metrics)
    except Exception:
        pass

# ==========================================
# 🏗️ 动态模型构建中心
# ==========================================
def build_model(config, data_spec, device):
    cont_dim = len(data_spec["feature_cols"]) - len(data_spec.get("categorical_cols", []))
    cat_cards = data_spec.get("categorical_cardinalities", {})
    model_type = config.get("model", "TARNET")
    
    if model_type == "MTMT":
        return MTMT(
            continuous_dim=cont_dim, 
            categorical_cardinalities=cat_cards, 
            hidden_dims=config["hidden_dims"], 
            dropout_rate=config["dropout_rate"]
        ).to(device)
    elif model_type == "MOTTO_DA":
        return MOTTO_DA(
            continuous_dim=cont_dim, 
            categorical_cardinalities=cat_cards, 
            hidden_dims=config["hidden_dims"], 
            dropout_rate=config["dropout_rate"]
        ).to(device)
    elif model_type == "TARNET":
        if config["task"] == "train_c":
            return TARNET_Baseline(continuous_dim=cont_dim, categorical_cardinalities=cat_cards, hidden_dims=config["hidden_dims"], dropout_rate=config["dropout_rate"]).to(device)
            
        elif config["task"] == "train_y":
            c_path = config.get("c_ckpt_path")
            
            # 🌟 修复点：默认用 Y 的参数，但如果有 C 的 config，就强制用 C 的！
            c_hidden_dims = config["hidden_dims"]
            c_dropout_rate = config["dropout_rate"]
            
            if c_path and os.path.exists(c_path):
                # 去找 C 的那张 best_config.json 图纸
                c_config_path = c_path.replace("best_model.pth", "best_config.json")
                if os.path.exists(c_config_path):
                    import json
                    with open(c_config_path, 'r') as f:
                        c_cfg = json.load(f)
                        c_hidden_dims = c_cfg.get("hidden_dims", c_hidden_dims)
                        c_dropout_rate = c_cfg.get("dropout_rate", c_dropout_rate)
            
            # 🌟 用正确的 C 图纸来构建 C 的空壳
            model_c = TARNET_Baseline(
                continuous_dim=cont_dim, 
                categorical_cardinalities=cat_cards, 
                hidden_dims=c_hidden_dims, 
                dropout_rate=c_dropout_rate
            ).to(device)
            
            if c_path and os.path.exists(c_path):
                model_c.load_state_dict(torch.load(c_path, map_location=device))
                
            if config.get("c_fusion_mode") == "moe":
                return TARNET_MoE(continuous_dim=cont_dim, categorical_cardinalities=cat_cards, hidden_dims=config["hidden_dims"], dropout_rate=config["dropout_rate"], c_model=model_c).to(device)
            elif config.get("c_fusion_mode") == "res_moe": # 👈 V6 的专属通道
                return TARNET_Residual_MoE(continuous_dim=cont_dim, categorical_cardinalities=cat_cards, hidden_dims=config["hidden_dims"], dropout_rate=config["dropout_rate"], c_model=model_c).to(device)
            elif config.get("c_fusion_mode") == "v7_truncated_moe":
                return TARNET_V7_Truncated_MoE(
                    continuous_dim=cont_dim, 
                    categorical_cardinalities=cat_cards, 
                    hidden_dims=config["hidden_dims"], 
                    dropout_rate=config["dropout_rate"], 
                    c_model=model_c,
                    truncation_mode=config.get("truncation_mode", "hard"),
                    truncation_pct=config.get("truncation_pct", 0.05),
                    truncation_temp=config.get("truncation_temp", 10.0),
                    ema_momentum=config.get("ema_momentum", 0.9)
                ).to(device)
            elif config.get("c_fusion_mode") == "v8_evolution_moe":
                return TARNET_V8_Evolution_MoE(
                    continuous_dim=cont_dim, 
                    categorical_cardinalities=cat_cards, 
                    hidden_dims=config["hidden_dims"], 
                    dropout_rate=config["dropout_rate"], 
                    c_model=model_c,
                    v8_scheme=config.get("v8_scheme", 3),               # 动态传入是方案几
                    shared_emb_dim=config["hidden_dims"][-1],           # 自动取底座的最后一层维度
                    truncation_pct=config.get("truncation_pct", 0.05),  # 接收 Ray 传来的 0.05 或 0.50
                    truncation_temp=config.get("truncation_temp", 10.0),
                    ema_momentum=config.get("ema_momentum", 0.9)
                ).to(device)
            else:
                return TARNET_Proposed(continuous_dim=cont_dim, categorical_cardinalities=cat_cards, hidden_dims=config["hidden_dims"], dropout_rate=config["dropout_rate"], c_fusion_mode=config["c_fusion_mode"], c_embedding_dim=config.get("c_embedding_dim", 4), c_model=model_c).to(device)
    else:
        raise ValueError(f"❌ 不支持的模型骨架: {model_type}")

# ==========================================
# 📊 终极落盘引擎 (只在最后被 main.py 唤起一次)
# ==========================================
# ==========================================
# 📊 终极落盘引擎 (只在最后被 main.py 唤起一次)
# ==========================================
def run_final_evaluation(model, loaders_dict, device, config):
    import json # 确保顶部有 import json
    
    run_hier = config.get("run_hierarchy", "default_run")
    project_root = config.get("project_root", ".") 
    res_dir = os.path.join(project_root, f"results/{run_hier}") 
    os.makedirs(res_dir, exist_ok=True)

    mode = config.get("mode", "tune")
    eval_max_steps = config.get("max_steps_per_epoch", float('inf')) if mode == "debug" else float('inf')
    
    print("\n" + "="*80)
    print(f"📈 闭环终测：加载全局最优权重进行全量验证，结果落盘至: {res_dir}")
    print("="*80)
    
    final_metrics = {}
    for split_name, loader in loaders_dict.items():
        if loader is None: continue
            
        # 1. 存全量预测分数的 CSV
        csv_path = os.path.join(res_dir, f"{split_name}_dist.csv")
        metrics, _ = evaluate_and_dump(model, loader, device, config["task"], save_path=csv_path, max_steps=eval_max_steps, show_pbar=True)
        
        # 将当前 split (train/valid/test) 的指标拼接到全局字典里
        for k, v in metrics.items():
            final_metrics[f"{split_name.capitalize()}_{k}"] = v
            
        print(f"[{split_name.upper():<5}] 🎯Y_AUUC: {metrics.get('Target_Y_AUUC',0):.4f} | Y_AUC: {metrics.get('Target_Y_AUC',0):.4f} | Y_MAE: {metrics.get('Target_Y_MAE',0):.4f}")
        print(f"        🎯C_AUUC: {metrics.get('Target_C_AUUC',0):.4f} | C_AUC: {metrics.get('Target_C_AUC',0):.4f} | C_MAE: {metrics.get('Target_C_MAE',0):.4f}")

    # 🌟 新增核心逻辑：把所有指标汇总落盘
    metrics_save_path = os.path.join(res_dir, "final_metrics.json")
    try:
        with open(metrics_save_path, 'w') as f:
            json.dump(final_metrics, f, ensure_ascii=False, indent=4)
        print(f"\n✅ 所有评估指标 (Metrics) 已成功落盘至: {metrics_save_path}")
    except Exception as e:
        print(f"❌ 指标落盘失败: {e}")

    return final_metrics

# ==========================================
# 🚀 核心训练微操官
# ==========================================
def train_trial(trial_cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_spec = trial_cfg["data"]
    accumulate_steps = trial_cfg.get("accumulate_steps", 1)
    
    mode = trial_cfg.get("mode", "tune")
    max_steps = trial_cfg.get("max_steps_per_epoch", float('inf'))
    run_hier = trial_cfg.get("run_hierarchy", "default_run")
    
    eval_max_steps = max_steps if mode == "debug" else float('inf')
    show_pbar = (mode in ["debug", "eval"])
    
    num_workers = trial_cfg.get("num_workers", 0)
    try:
        train_loader = DataLoader(UpliftDataset(trial_cfg, split="train"), batch_size=trial_cfg["batch_size"], shuffle=True, collate_fn=uplift_collate_fn, pin_memory=True, num_workers=num_workers)
        valid_loader = DataLoader(UpliftDataset(trial_cfg, split="valid"), batch_size=trial_cfg["batch_size"], shuffle=False, collate_fn=uplift_collate_fn, pin_memory=True, num_workers=num_workers)
        test_loader = DataLoader(UpliftDataset(trial_cfg, split="test"), batch_size=trial_cfg["batch_size"], shuffle=False, collate_fn=uplift_collate_fn, pin_memory=True, num_workers=num_workers)
    except Exception as e:
        print(f"⚠️ 数据集加载出错: {e}")
        return
        
    loaders_dict = {"train": train_loader, "valid": valid_loader, "test": test_loader}
    model = build_model(trial_cfg, data_spec, device)
    
    # --- EVAL 模式直通车：主程序调用时，直接验证，不走下面的训练 ---
    if mode == "eval":
        ckpt_path = trial_cfg.get("eval_ckpt_path")
        if not os.path.exists(ckpt_path): return
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state.get("model_state_dict", state))
        run_final_evaluation(model, loaders_dict, device, trial_cfg)
        return

    dro_criterion = None
    if "group_dro" in trial_cfg.get("loss_types", []):
        dro_criterion = UpliftGroupDROLoss(
            grouping_mode=trial_cfg.get("dro_grouping_mode", "2d_coarse"),
            ng=trial_cfg.get("dro_ng", 0.01),
            ema_gamma=trial_cfg.get("dro_ema", 0.1),
            max_clip=trial_cfg.get("dro_clip", 0.5)
        ).to(device)

    # --- 正常训练逻辑 ---
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=trial_cfg["learning_rate"], weight_decay=trial_cfg["weight_decay"])
    focal_criterion = FocalLoss(alpha=trial_cfg.get("focal_alpha", 0.5), gamma=trial_cfg.get("focal_gamma", 0.0))
    scaler = GradScaler()
    
    project_root = trial_cfg.get("project_root", ".")
    ckpt_dir = os.path.join(project_root, f"ckpts/{run_hier}")
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # 🌟 定义竞争打擂台的路径
    best_ckpt_save_path = os.path.join(ckpt_dir, "best_model.pth")
    best_config_save_path = os.path.join(ckpt_dir, "best_config.json")
    global_metric_path = os.path.join(ckpt_dir, "global_best_metric.txt")

    num_epochs = trial_cfg.get("num_epochs", 100)
    patience = trial_cfg.get("patience", 10) 
    epochs_no_improve = 0
    local_best_metric = -float('inf')

    for epoch in range(1, num_epochs + 1):
        train_start = time.time()
        model.train()
        epoch_loss = 0.0
        epoch_loss_main = 0.0
        epoch_loss_aux = 0.0
        optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc=f"🚀 Epoch {epoch} Train", leave=False) if show_pbar else train_loader
        for batch_idx, (x_cont, x_cat, t, y, c) in enumerate(pbar):
            if batch_idx >= max_steps: break 
            x_cont = x_cont.to(device, non_blocking=True) if x_cont is not None else None
            x_cat = {k: v.to(device, non_blocking=True) for k, v in x_cat.items()} if x_cat is not None else None
            t, y, c = t.to(device, non_blocking=True), y.to(device, non_blocking=True), c.to(device, non_blocking=True)
            
            with autocast():
                if trial_cfg["task"] == "train_c":
                    c0_pred, c1_pred, z_c = model(x_cont, x_cat)
                    cls_loss = 2 * focal_criterion(torch.where(t == 1, c1_pred, c0_pred), c.float())
                    align_weight = trial_cfg.get("align_weight", trial_cfg.get("mmd_alpha", 0.1))
                    
                    if align_weight > 0:
                        align_method = trial_cfg.get("align_method", "mmd")
                        z_t1, z_t0 = z_c[t == 1], z_c[t == 0]
                        if align_method == "mmd": align_loss = mmd_loss(z_t1, z_t0, sigma=trial_cfg.get("mmd_sigma", 1.0))
                        elif align_method == "swd": align_loss = sliced_wasserstein_distance(z_t1, z_t0, num_projections=128)
                        elif align_method == "moment": align_loss = moment_matching_loss(z_t1, z_t0)
                        else: align_loss = torch.tensor(0.0, device=device)
                    else:
                        align_loss = torch.tensor(0.0, device=device)
                    loss = cls_loss + align_weight * align_loss
                           
                elif trial_cfg["task"] == "train_y":
                    preds = model(x_cont, x_cat)
                    if isinstance(preds, dict) and "main_task" in preds:
                        # 多任务学习分支 (MTMT 等)
                        (y0_pred, y1_pred) = preds["main_task"]
                        (c0_pred, c1_pred) = preds["aux_task"]
                        pi_dict = preds.get("pi_dict", {})
                        
                        loss_main, _ = compute_stage3_loss(y0_pred, y1_pred, y, t, pi_dict, trial_cfg, dro_criterion=dro_criterion)
                        loss_aux, _ = compute_stage3_loss(c0_pred, c1_pred, c, t, pi_dict, trial_cfg, dro_criterion=dro_criterion)
                        
                        # 静态权重：主任务 1.0，辅助任务 0.2
                        loss = 1.0 * loss_main + 0.2 * loss_aux
                        
                        # 累加各个任务的 Loss 以便监控
                        epoch_loss_main += loss_main.item()
                        epoch_loss_aux += loss_aux.item()
                    else:
                        # 单任务分支
                        y0_pred, y1_pred, pi_dict = preds
                        loss, _ = compute_stage3_loss(y0_pred, y1_pred, y, t, pi_dict, trial_cfg, dro_criterion=dro_criterion)
            
            scaler.scale(loss / accumulate_steps).backward()
            
            if (batch_idx + 1) % accumulate_steps == 0 or (batch_idx + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            epoch_loss += loss.item()
                
        train_time = time.time() - train_start

        # 仅验证获取指标
        val_start = time.time()
        val_metrics, _ = evaluate_and_dump(model, valid_loader, device, trial_cfg["task"], max_steps=eval_max_steps, show_pbar=show_pbar)
        val_time = time.time() - val_start
        
        prefix = "Target_C" if trial_cfg["task"] == "train_c" else "Target_Y"
        target_metric = val_metrics.get(f"{prefix}_AUUC", 0)
        
        is_best = ""
        # 1. 局部早停监控
        if target_metric > local_best_metric:
            local_best_metric = target_metric
            epochs_no_improve = 0  
        else:
            epochs_no_improve += 1
            
        # 🌟 2. 核心逻辑：全局打擂台！
        global_best = -float('inf')
        if os.path.exists(global_metric_path):
            try:
                with open(global_metric_path, 'r') as f:
                    content = f.read().strip()
                    if content: global_best = float(content)
            except Exception: pass # 并发冲突时静默
            
        if target_metric > global_best:
            try:
                # 破纪录，挂牌
                with open(global_metric_path, 'w') as f:
                    f.write(str(target_metric))
                # 存模型
                torch.save(model.state_dict(), best_ckpt_save_path)
                # 存 Config（剥离掉不能 JSON 化的烂属性）
                safe_cfg = {k: v for k, v in trial_cfg.items() if isinstance(v, (int, float, str, list, bool)) or v is None}
                with open(best_config_save_path, 'w') as f:
                    json.dump(safe_cfg, f, ensure_ascii=False, indent=4)
                is_best = "🌍(GLOBAL!)"
            except Exception: pass
            
        # 记录多任务 Loss 埋点
        report_metrics = {"epoch": epoch, "loss": epoch_loss / max(1, min(max_steps, len(train_loader))), **val_metrics}
        if epoch_loss_main > 0 or epoch_loss_aux > 0:
            report_metrics["loss_main"] = epoch_loss_main / max(1, min(max_steps, len(train_loader)))
            report_metrics["loss_aux"] = epoch_loss_aux / max(1, min(max_steps, len(train_loader)))
            
        safe_report(report_metrics)
        
        m, p = val_metrics, prefix
        print(f"Ep {epoch:<2} {is_best:<10} | T:{train_time:.1f}s V:{val_time:.1f}s | "
              f"[{p}] AUUC:{m.get(f'{p}_AUUC',0):.4f} AUQC:{m.get(f'{p}_AUQC',0):.4f} "
              f"L10:{m.get(f'{p}_Lift@10',0):.4f} PCOC:{m.get(f'{p}_PCOC',0):.4f} "
              f"MAE:{m.get(f'{p}_MAE',0):.4f} Rec:{m.get(f'{p}_Recall',0):.4f} AUC:{m.get(f'{p}_AUC',0):.4f}")
        
        if epoch_loss_main > 0 or epoch_loss_aux > 0:
            print(f"   ↳ [MTMT Loss 监控] Total: {epoch_loss/max(1, len(train_loader)):.4f} | Main: {epoch_loss_main/max(1, len(train_loader)):.4f} | Aux: {epoch_loss_aux/max(1, len(train_loader)):.4f}")

        if epochs_no_improve >= patience:
            print(f"🛑 早停触发: 目标指标已连续 {patience} 个 Epoch 未提升。")
            break