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
def safe_report(metrics, mode="tune"):
    # 只要是复现类模式，强行在控制台打出指标
    if mode in ["reproduce", "reproduce_eval"]:
        ep = metrics.get("epoch", "?")
        loss = metrics.get("loss", 0.0)
        auuc = metrics.get("Target_Y_AUUC", metrics.get("Target_C_AUUC", 0.0))
        auc = metrics.get("Target_Y_AUC", metrics.get("Target_C_AUC", 0.0))
        print(f"📊 [MONITOR] Ep {ep} | Loss: {loss:.4f} | AUUC: {auuc:.4f} | AUC: {auc:.4f}")
        return # 拦截 Ray，防止报错假死

    # Tune 模式正常向 Ray 汇报
    try:
        from ray import train
        if train.get_context().get_trial_name(): 
            train.report(metrics)
            return
    except Exception as e:
        print(f"⚠️ [Ray Train Report Failed]: {e}")
        
    try:
        from ray import tune
        tune.report(**metrics)
    except Exception as e:
        print(f"⚠️ [Ray Tune Report Failed]: {e}")

# ==========================================
# 🏗️ 动态模型构建中心
# ==========================================
def build_model(config, data_spec, device):
    cont_dim = len(data_spec["feature_cols"]) - len(data_spec.get("categorical_cols", []))
    cat_cards = data_spec.get("categorical_cardinalities", {})
    model_type = config.get("model", "TARNET")

    if model_type == "TARNET":
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
                    # import json
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
            elif config.get("c_fusion_mode") == "v11_aligned_moe":
                return TARNET_V11_Aligned_MoE(
                    continuous_dim=cont_dim, 
                    categorical_cardinalities=cat_cards, 
                    hidden_dims=config["hidden_dims"], 
                    dropout_rate=config["dropout_rate"], 
                    c_model=model_c,
                    v11_scheme=config.get("v11_scheme", 4),   # 接收 4, 6, 7, 8
                    align_type=config.get("align_type", "lift"), 
                    align_temp=config.get("align_temp", 1.0)
                ).to(device)
            else:
                return TARNET_Proposed(continuous_dim=cont_dim, categorical_cardinalities=cat_cards, hidden_dims=config["hidden_dims"], dropout_rate=config["dropout_rate"], c_fusion_mode=config["c_fusion_mode"], c_embedding_dim=config.get("c_embedding_dim", 4), c_model=model_c).to(device)
    elif model_type == "MTMT":
        return MTMT_STMT(
            continuous_dim=cont_dim,
            categorical_cardinalities=cat_cards,
            num_experts=config.get("num_experts", 4),
            expert_type=config.get("expert_type", "mlp"),           # 🌟 新增：对应搜索空间
            expert_hidden_dims=config.get("expert_hidden_dims", [64]), # 🌟 修正：参数名和类型
            dropout_rate=config.get("dropout_rate", 0.1),            # 🌟 新增：对应搜索空间
            t_emb_dim=config.get("t_emb_dim", 16) 
        ).to(device)
    # 找到 trainer.py 的 build_model，加上：
    elif model_type == "ECUP":
        return ECUP_Model(
            continuous_dim=cont_dim,
            categorical_cardinalities=cat_cards,
            # hidden_dims=config.get("hidden_dims", [128, 64]),
            d_dim=config.get("d_dim", 16),
            gamma=config.get("gamma", 1.0),
            tower_h=config.get("tower_h", 128),
            tae_h=config.get("tae_h", 64),
            num_heads=config.get("num_heads", 2)
        ).to(device)
    elif model_type == "MOTTO":
        return MOTTO_Model(
            continuous_dim=cont_dim,
            categorical_cardinalities=cat_cards,
            d_dim=config.get("d_dim", 16),                    # 🌟 接入公平投影维度
            bottom_dim=config.get("bottom_dim", 128),
            expert_hidden_dims=config.get("expert_hidden_dims", [64, 64]), # 🌟 接入列表参数
            tower_dim=config.get("tower_dim", 64),
            use_specific_experts=config.get("use_specific_experts", True)
        ).to(device)
    # 找到 trainer.py 的 build_model，加上：
    elif model_type == "TARNET_MT":
        return TARNET_Naive_MT(
            continuous_dim=cont_dim,
            categorical_cardinalities=cat_cards,
            hidden_dims=config.get("hidden_dims", [128, 64, 32]),
            dropout_rate=config.get("dropout_rate", 0.0)
        ).to(device)
    elif model_type == "DragonNet":
        return DragonNet(continuous_dim=cont_dim, categorical_cardinalities=cat_cards, hidden_dims=config.get("hidden_dims", [128, 64])).to(device)
    elif model_type == "EUEN":
        return EUEN(continuous_dim=cont_dim, categorical_cardinalities=cat_cards, hidden_dims=config.get("hidden_dims", [128, 64])).to(device)
    elif model_type == "EFIN":
        # EFIN 的超参提取 (注意要把 embed_dim 也传进去)
        embed_dim = config.get("efin_embed_dim", 16)
        hidden_dims = config.get("hidden_dims", [128, 64])
        return EFIN(continuous_dim=cont_dim, categorical_cardinalities=cat_cards, 
                             embed_dim=embed_dim, hidden_dims=hidden_dims).to(device)
    # 👇 新增 S 和 T Learner 的构建分支
    elif model_type == "DESCN":
            return DESCN(
                continuous_dim=cont_dim, 
                categorical_cardinalities=cat_cards, 
                shared_dims=config.get("hidden_dims", [128]), # 论文默认 shared 128
                tower_dims=config.get("hidden_dims", [128])                              # 论文默认 tower 64
            ).to(device)
    
    elif model_type == "S_Learner":
        return S_Learner(continuous_dim=cont_dim, categorical_cardinalities=cat_cards, hidden_dims=config.get("hidden_dims", [128, 64])).to(device)
    elif model_type == "T_Learner":
        return T_Learner(continuous_dim=cont_dim, categorical_cardinalities=cat_cards, hidden_dims=config.get("hidden_dims", [128, 64])).to(device)
    elif model_type == "CFRNet":
        return CFRNet(continuous_dim=cont_dim, categorical_cardinalities=cat_cards, hidden_dims=config.get("hidden_dims", [128, 64])).to(device)
    else:
        raise ValueError(f"❌ 不支持的模型骨架: {model_type}")

# ==========================================
# 📊 终极落盘引擎 (只在最后被 main.py 唤起一次)
# ==========================================
# ==========================================
# 📊 终极落盘引擎 (只在最后被 main.py 唤起一次)
# ==========================================
def run_final_evaluation(model, loaders_dict, device, config):
    run_hier = config.get("run_hierarchy", "default_run")
    seed = config.get("seed", 42)
    mode = config.get("mode", "tune")
    
    # 🌟 无论是 reproduce 还是 reproduce_eval 都要劫持路径
    if mode in ["reproduce", "reproduce_eval"]:
        run_hier = os.path.join(run_hier, f"seed_{seed}")
        
    project_root = config.get("project_root", ".") 
    res_dir = os.path.join(project_root, f"results/{run_hier}") 
    os.makedirs(res_dir, exist_ok=True)

    eval_max_steps = config.get("max_steps_per_epoch", float('inf')) if mode == "debug" else float('inf')
    
    print("\n" + "="*80)
    print(f"📈 闭环终测：加载全局最优权重进行全量验证，结果落盘至: {res_dir}")
    print("="*80)
    
    final_metrics = {}
    for split_name, loader in loaders_dict.items():
        if loader is None: continue
            
        csv_path = os.path.join(res_dir, f"{split_name}_dist.csv")
        metrics, _ = evaluate_and_dump(model, loader, device, config["task"], save_path=csv_path, max_steps=eval_max_steps, show_pbar=True)
        
        for k, v in metrics.items():
            final_metrics[f"{split_name.capitalize()}_{k}"] = v
            
        print(f"[{split_name.upper():<5}] 🎯Y_AUUC: {metrics.get('Target_Y_AUUC',0):.4f} | Y_AUC: {metrics.get('Target_Y_AUC',0):.4f} | Y_MAE: {metrics.get('Target_Y_MAE',0):.4f}")
        print(f"        🎯C_AUUC: {metrics.get('Target_C_AUUC',0):.4f} | C_AUC: {metrics.get('Target_C_AUC',0):.4f} | C_MAE: {metrics.get('Target_C_MAE',0):.4f}")

    metrics_save_path = os.path.join(res_dir, "final_metrics.json")
    try:
        with open(metrics_save_path, 'w') as f:
            json.dump(final_metrics, f, ensure_ascii=False, indent=4)
        print(f"\n✅ 所有评估指标 (Metrics) 已成功落盘至: {metrics_save_path}")
    except Exception as e:
        print(f"❌ 指标落盘失败: {e}")

    return final_metrics
# def run_final_evaluation(model, loaders_dict, device, config):
#     # import json # 确保顶部有 import json
    
#     run_hier = config.get("run_hierarchy", "default_run")
#     seed = config.get("seed", 42)
    
#     if mode == "reproduce":
#         run_hier = os.path.join(run_hier, f"seed_{seed}")
#     project_root = config.get("project_root", ".") 
#     res_dir = os.path.join(project_root, f"results/{run_hier}") 
#     os.makedirs(res_dir, exist_ok=True)

#     mode = config.get("mode", "tune")

#     eval_max_steps = config.get("max_steps_per_epoch", float('inf')) if mode == "debug" else float('inf')
    
#     print("\n" + "="*80)
#     print(f"📈 闭环终测：加载全局最优权重进行全量验证，结果落盘至: {res_dir}")
#     print("="*80)
    
#     final_metrics = {}
#     for split_name, loader in loaders_dict.items():
#         if loader is None: continue
            
#         # 1. 存全量预测分数的 CSV
#         csv_path = os.path.join(res_dir, f"{split_name}_dist.csv")
#         metrics, _ = evaluate_and_dump(model, loader, device, config["task"], save_path=csv_path, max_steps=eval_max_steps, show_pbar=True)
        
#         # 将当前 split (train/valid/test) 的指标拼接到全局字典里
#         for k, v in metrics.items():
#             final_metrics[f"{split_name.capitalize()}_{k}"] = v
            
#         print(f"[{split_name.upper():<5}] 🎯Y_AUUC: {metrics.get('Target_Y_AUUC',0):.4f} | Y_AUC: {metrics.get('Target_Y_AUC',0):.4f} | Y_MAE: {metrics.get('Target_Y_MAE',0):.4f}")
#         print(f"        🎯C_AUUC: {metrics.get('Target_C_AUUC',0):.4f} | C_AUC: {metrics.get('Target_C_AUC',0):.4f} | C_MAE: {metrics.get('Target_C_MAE',0):.4f}")

#     # 🌟 新增核心逻辑：把所有指标汇总落盘
#     metrics_save_path = os.path.join(res_dir, "final_metrics.json")
#     try:
#         with open(metrics_save_path, 'w') as f:
#             json.dump(final_metrics, f, ensure_ascii=False, indent=4)
#         print(f"\n✅ 所有评估指标 (Metrics) 已成功落盘至: {metrics_save_path}")
#     except Exception as e:
#         print(f"❌ 指标落盘失败: {e}")

#     return final_metrics

# ==========================================
# 🚀 核心训练微操官
# ==========================================
def train_trial(trial_cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_spec = trial_cfg["data"]
    accumulate_steps = trial_cfg.get("accumulate_steps", 1)
    
    mode = trial_cfg.get("mode", "tune")
    seed = trial_cfg.get("seed", 42)

    max_steps = trial_cfg.get("max_steps_per_epoch", float('inf'))
    run_hier = trial_cfg.get("run_hierarchy", "default_run")

    if mode in ["reproduce", "reproduce_eval"]:
        run_hier = os.path.join(run_hier, f"seed_{seed}")
        
    eval_max_steps = max_steps if mode == "debug" else float('inf')
    show_pbar = (mode in ["debug", "eval"])
    
    project_root = trial_cfg.get("project_root", ".")
    ckpt_dir = os.path.join(project_root, f"ckpts/{run_hier}")
    res_dir = os.path.join(project_root, f"results/{run_hier}")
    os.makedirs(ckpt_dir, exist_ok=True)
    
    best_ckpt_save_path = os.path.join(ckpt_dir, "best_model.pth")
    best_config_save_path = os.path.join(ckpt_dir, "best_config.json")
    global_metric_path = os.path.join(ckpt_dir, "global_best_metric.txt")
    metrics_save_path = os.path.join(res_dir, "final_metrics.json")

    # 🌟 跳过逻辑: 正常 reproduce 且已有结果
    if mode == "reproduce" and os.path.exists(metrics_save_path):
        print(f"⏩ [SKIP] Seed {seed} 已经存在 final_metrics.json，安全跳过。")
        return

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
    
    # --- 🌟 EVAL 直通车 (处理原有 eval 和新的 reproduce_eval) ---
    if mode == "eval":
        ckpt_path = trial_cfg.get("eval_ckpt_path")
        if not os.path.exists(ckpt_path): return
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state.get("model_state_dict", state))
        run_final_evaluation(model, loaders_dict, device, trial_cfg)
        return
        
    if mode == "reproduce_eval":
        if os.path.exists(best_ckpt_save_path):
            print(f"🔄 [REPRODUCE-EVAL] Seed {seed} 直接加载本地中断权重评估...")
            state = torch.load(best_ckpt_save_path, map_location=device)
            model.load_state_dict(state.get("model_state_dict", state))
            run_final_evaluation(model, loaders_dict, device, trial_cfg)
        else:
            print(f"❌ [ERROR] 未找到 Seed {seed} 的权重文件: {best_ckpt_save_path}，无法执行评估！")
        return
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # data_spec = trial_cfg["data"]
    # accumulate_steps = trial_cfg.get("accumulate_steps", 1)
    
    # mode = trial_cfg.get("mode", "tune")
    # seed = trial_cfg.get("seed", 42)

    # max_steps = trial_cfg.get("max_steps_per_epoch", float('inf'))
    # run_hier = trial_cfg.get("run_hierarchy", "default_run")

    # if mode == "reproduce":
    #     run_hier = os.path.join(run_hier, f"seed_{seed}")
    # eval_max_steps = max_steps if mode == "debug" else float('inf')
    # show_pbar = (mode in ["debug", "eval"])
    
    # num_workers = trial_cfg.get("num_workers", 0)
    # try:
    #     train_loader = DataLoader(UpliftDataset(trial_cfg, split="train"), batch_size=trial_cfg["batch_size"], shuffle=True, collate_fn=uplift_collate_fn, pin_memory=True, num_workers=num_workers)
    #     valid_loader = DataLoader(UpliftDataset(trial_cfg, split="valid"), batch_size=trial_cfg["batch_size"], shuffle=False, collate_fn=uplift_collate_fn, pin_memory=True, num_workers=num_workers)
    #     test_loader = DataLoader(UpliftDataset(trial_cfg, split="test"), batch_size=trial_cfg["batch_size"], shuffle=False, collate_fn=uplift_collate_fn, pin_memory=True, num_workers=num_workers)
    # except Exception as e:
    #     print(f"⚠️ 数据集加载出错: {e}")
    #     return
        
    # loaders_dict = {"train": train_loader, "valid": valid_loader, "test": test_loader}
    # model = build_model(trial_cfg, data_spec, device)
    
    # # --- EVAL 模式直通车：主程序调用时，直接验证，不走下面的训练 ---
    # if mode == "eval":
    #     ckpt_path = trial_cfg.get("eval_ckpt_path")
    #     if not os.path.exists(ckpt_path): return
    #     state = torch.load(ckpt_path, map_location=device)
    #     model.load_state_dict(state.get("model_state_dict", state))
    #     run_final_evaluation(model, loaders_dict, device, trial_cfg)
    #     return


    # project_root = trial_cfg.get("project_root", ".")
    # ckpt_dir = os.path.join(project_root, f"ckpts/{run_hier}")
    # os.makedirs(ckpt_dir, exist_ok=True)
    
    # # 🌟 定义竞争打擂台的路径
    # best_ckpt_save_path = os.path.join(ckpt_dir, "best_model.pth")
    # best_config_save_path = os.path.join(ckpt_dir, "best_config.json")
    # global_metric_path = os.path.join(ckpt_dir, "global_best_metric.txt")

    dro_criterion = None
    if "group_dro" in trial_cfg.get("loss_types", []):
        dro_criterion = UpliftGroupDROLoss(
            grouping_mode=trial_cfg.get("dro_grouping_mode", "2d_coarse"),
            ng=trial_cfg.get("dro_ng", 0.01),
            ema_gamma=trial_cfg.get("dro_ema", 0.1),
            max_clip=trial_cfg.get("dro_clip", 0.5)
        ).to(device)

    # --- 正常训练逻辑 ---
    optim_mode = trial_cfg.get('optim_mode', 'adam')
    if optim_mode == 'adam':
        optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=trial_cfg["learning_rate"], 
                           weight_decay=trial_cfg["weight_decay"])
    elif optim_mode == 'adamw':
        optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=trial_cfg["learning_rate"], 
                           weight_decay=trial_cfg["weight_decay"])
    focal_criterion = FocalLoss(alpha=trial_cfg.get("focal_alpha", 0.5), gamma=trial_cfg.get("focal_gamma", 0.0))
    scaler = GradScaler()
    


    num_epochs = trial_cfg.get("num_epochs", 100)
    patience = trial_cfg.get("patience", 10) 
    epochs_no_improve = 0
    local_best_metric = -float('inf')

    for epoch in range(1, num_epochs + 1):
        train_start = time.time()
        model.train()
        epoch_loss = 0.0
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
                    if trial_cfg.get("model") == "ECUP":
                        # 🌟 ECUP 专属分支: 接收字典，调用 ECUP 专属全链路 Loss
                        preds_dict = model(x_cont, x_cat)
                        loss, loss_comp = compute_ecup_loss(preds_dict, y, c, t, trial_cfg)
                    elif trial_cfg.get("model") == "MTMT":
                        # 🌟 MTMT 多任务分支
                        preds_dict = model(x_cont, x_cat)
                        loss, loss_comp = compute_mtmt_loss(
                            preds_dict, y, c, t, preds_dict["pi_dict"], trial_cfg, dro_criterion
                        )
                    elif trial_cfg.get("model") == "MOTTO":
                        # 🌟 拦截 MOTTO，分发到专属 Loss
                        preds_dict = model(x_cont, x_cat)
                        loss, loss_comp = compute_motto_loss(preds_dict, y, c, t, trial_cfg)
                    elif trial_cfg.get("model") == "DragonNet":
                        y0_pred, y1_pred, pi_dict = model(x_cont, x_cat)
                        loss, loss_comp = compute_dragonnet_loss(y0_pred, y1_pred, y, t, pi_dict, trial_cfg)
                    elif trial_cfg.get("model") == "EFIN":
                        y0_pred, y1_pred, pi_dict = model(x_cont, x_cat)
                        loss, loss_comp = compute_efin_loss(y0_pred, y1_pred, y, t, pi_dict, trial_cfg)
                    elif trial_cfg.get("model") == "S_Learner":
                        # 注意这里传了 t=t
                        y0_pred, y1_pred, pi_dict = model(x_cont, x_cat, t=t) 
                        loss, _ = compute_stage3_loss(y0_pred, y1_pred, y, t, pi_dict, trial_cfg, dro_criterion=dro_criterion)
                    # 找到 trainer.py 的 train_trial 的 forward 模块，插入这段：
                    elif trial_cfg.get("model") == "TARNET_MT":
                        # 🌟 Naive Multitask 分支
                        preds_dict = model(x_cont, x_cat)
                        loss, loss_comp = compute_naive_mt_loss(
                            preds_dict, y, c, t, preds_dict["pi_dict"], trial_cfg, dro_criterion
                        )
                    # 👇 新增 CFRNet 专属拦截 (计算表征距离)
                    elif trial_cfg.get("model") == "CFRNet":
                        y0_pred, y1_pred, pi_dict = model(x_cont, x_cat)
                        loss, loss_comp = compute_stage3_loss(y0_pred, y1_pred, y, t, pi_dict, trial_cfg, dro_criterion=dro_criterion)
                        z = pi_dict["z"]
                        z_t1, z_t0 = z[t == 1], z[t == 0]
                        cfr_weight = trial_cfg.get("cfr_weight", 0.1) # 获取对齐强度
                        if cfr_weight > 0 and z_t1.size(0) > 0 and z_t0.size(0) > 0:
                            # 默认调用你刚刚修复防崩版的 SWD
                            align_loss = sliced_wasserstein_distance(z_t1, z_t0)
                            
                            loss += cfr_weight * align_loss
                            loss_comp["cfr_align_loss"] = (cfr_weight * align_loss).item()
                    elif trial_cfg.get("model") == "DESCN":
                        y0_pred, y1_pred, pi_dict = model(x_cont, x_cat)
                        loss, loss_comp = compute_descn_loss(y0_pred, y1_pred, y, t, pi_dict, trial_cfg)
                    

                    else:
                        y0_pred, y1_pred, pi_dict = model(x_cont, x_cat)
                        # loss, _ = compute_stage3_loss(y0_pred, y1_pred, y, t, pi_dict, trial_cfg)
                        # V9特点：传入DRO
                        loss, _ = compute_stage3_loss(y0_pred, y1_pred, y, t, pi_dict, trial_cfg, dro_criterion=dro_criterion)
            
            scaler.scale(loss / accumulate_steps).backward()
            
            if (batch_idx + 1) % accumulate_steps == 0 or (batch_idx + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            epoch_loss += loss.item()
        train_time = time.time() - train_start

        val_start = time.time()
        val_metrics, _ = evaluate_and_dump(model, valid_loader, device, trial_cfg["task"], max_steps=eval_max_steps, show_pbar=show_pbar)
        test_metrics, _ = evaluate_and_dump(model, test_loader, device, trial_cfg["task"], max_steps=eval_max_steps, show_pbar=False)
        val_time = time.time() - val_start
        
        prefix = "Target_C" if trial_cfg["task"] == "train_c" else "Target_Y"
        target_metric = val_metrics.get(f"{prefix}_AUUC", 0)

        is_local_best = False
        if target_metric > local_best_metric:
            local_best_metric = target_metric
            local_best_val_metrics = val_metrics
            local_best_test_metrics = test_metrics
            best_epoch = epoch
            is_local_best = True
            epochs_no_improve = 0  
        else:
            epochs_no_improve += 1

        is_best = ""
        global_best = -float('inf')
        if os.path.exists(global_metric_path):
            try:
                with open(global_metric_path, 'r') as f:
                    content = f.read().strip()
                    if content: global_best = float(content)
            except Exception:
                pass 
            
        if target_metric > global_best:
            try:
                with open(global_metric_path, 'w') as f:
                    f.write(str(target_metric))
                torch.save(model.state_dict(), best_ckpt_save_path)
                safe_cfg = {k: v for k, v in trial_cfg.items() if isinstance(v, (int, float, str, list, bool)) or v is None}
                with open(best_config_save_path, 'w') as f:
                    json.dump(safe_cfg, f, ensure_ascii=False, indent=4)
                is_best = "🌍(GLOBAL!)"
            except Exception:
                pass
            
        report_dict = {"epoch": epoch, "loss": epoch_loss / max(1, min(max_steps, len(train_loader))), **val_metrics}
        report_dict.update({f"local_best_valid_{k}": v for k, v in local_best_val_metrics.items()})
        report_dict.update({f"test_{k}": v for k, v in test_metrics.items()})
        report_dict.update({f"local_best_test_{k}": v for k, v in local_best_test_metrics.items()})
        report_dict.update({"best_epoch": best_epoch})

        safe_report(report_dict, mode=mode)
        
        mv, mt, p = val_metrics, test_metrics, prefix
        print(f"Ep {epoch:<2} {is_best:<10} | T:{train_time:.1f}s V+Test:{val_time:.1f}s")
        print(f"   [VAL ] AUUC:{mv.get(f'{p}_AUUC',0):.4f} | AUQC:{mv.get(f'{p}_AUQC',0):.4f} | L10:{mv.get(f'{p}_Lift@10',0):.4f} | AUC:{mv.get(f'{p}_AUC',0):.4f}")
        print(f"   [TEST] AUUC:{mt.get(f'{p}_AUUC',0):.4f} | AUQC:{mt.get(f'{p}_AUQC',0):.4f} | L10:{mt.get(f'{p}_Lift@10',0):.4f} | AUC:{mt.get(f'{p}_AUC',0):.4f}")
        print("-" * 60)

        if epochs_no_improve >= patience:
            print(f"🛑 早停触发: 目标指标已连续 {patience} 个 Epoch 未提升。")
            break
        # train_time = time.time() - train_start
        # # ==========================================
        # # 🌟 修改点 1：同时对 Valid 和 Test 进行评估
        # # ==========================================
        # val_start = time.time()
        
        # # 1. 跑验证集 (Valid / Eval)
        # val_metrics, _ = evaluate_and_dump(model, valid_loader, device, trial_cfg["task"], max_steps=eval_max_steps, show_pbar=show_pbar)
        
        # # 2. 跑测试集 (Test) - 关掉 pbar 防止屏幕太乱
        # test_metrics, _ = evaluate_and_dump(model, test_loader, device, trial_cfg["task"], max_steps=eval_max_steps, show_pbar=False)
        
        # val_time = time.time() - val_start
        
        # prefix = "Target_C" if trial_cfg["task"] == "train_c" else "Target_Y"
        
        # # ⚠️ 铁律：打擂台和早停，必须且只能用 Valid 的指标！
        # target_metric = val_metrics.get(f"{prefix}_AUUC", 0)
        


        # # 🌟 2. 核心逻辑：锁定“局部最优” (仅针对当前这一组参数)
        # is_local_best = False
        # if target_metric > local_best_metric:
        #     local_best_metric = target_metric
        #     local_best_val_metrics = val_metrics
        #     local_best_test_metrics = test_metrics
        #     best_epoch = epoch
        #     is_local_best = True
        #     epochs_no_improve = 0  
        # else:
        #     epochs_no_improve += 1

        # # 🌟 3. 打印实时对比 (带上下铺对比，方便调参)
        # # print(f"\n--- Epoch {epoch} {'🔥 [NEW LOCAL BEST]' if is_local_best else ''} ---")
        # # print(f"   [VAL ] AUUC: {val_metrics.get(f'{prefix}_AUUC',0):.4f} | AUC: {val_metrics.get(f'{prefix}_AUC',0):.4f}")
        # # print(f"   [TEST] AUUC: {test_metrics.get(f'{prefix}_AUUC',0):.4f} | AUC: {test_metrics.get(f'{prefix}_AUC',0):.4f}")
        
        # # 🌟 4. 如果是局部最优，打印一份“高光快照”供提取
        # # if is_local_best:
        # #     print(f"📍 [LOCAL SNAPSHOT] 捕捉到当前参数组的最优时刻 (Epoch {epoch})")
        # #     print(f"📍 Best Valid AUUC: {local_best_metric:.4f}")
        # #     print(f"📍 Best Test  AUUC: {test_metrics.get(f'{prefix}_AUUC',0):.4f}")


        # # is_best = ""
        # # # 1. 局部早停监控
        # # if target_metric > local_best_metric:
        # #     local_best_metric = target_metric
        # #     epochs_no_improve = 0  
        # # else:
        # #     epochs_no_improve += 1
            
        # # 2. 全局打擂台
        # global_best = -float('inf')
        # if os.path.exists(global_metric_path):
        #     try:
        #         with open(global_metric_path, 'r') as f:
        #             content = f.read().strip()
        #             if content: global_best = float(content)
        #     except Exception: pass 
            
        # if target_metric > global_best:
        #     try:
        #         with open(global_metric_path, 'w') as f:
        #             f.write(str(target_metric))
        #         torch.save(model.state_dict(), best_ckpt_save_path)
        #         safe_cfg = {k: v for k, v in trial_cfg.items() if isinstance(v, (int, float, str, list, bool)) or v is None}
        #         with open(best_config_save_path, 'w') as f:
        #             json.dump(safe_cfg, f, ensure_ascii=False, indent=4)
        #         is_best = "🌍(GLOBAL!)"
        #     except Exception: pass
            
        # # 汇报给 Ray (附带上 test 的核心指标方便事后分析)
        # report_dict = {"epoch": epoch, "loss": epoch_loss / max(1, min(max_steps, len(train_loader))), **val_metrics}
        # report_dict.update({f"local_best_valid_{k}": v for k, v in local_best_val_metrics.items()})
        # report_dict.update({f"test_{k}": v for k, v in test_metrics.items()})
        # report_dict.update({f"local_best_test_{k}": v for k, v in local_best_test_metrics.items()})
        # report_dict.update({"best_epoch":best_epoch})

        # safe_report(report_dict, mode=mode)
        
        # # ==========================================
        # # 🌟 修改点 2：将 Valid 和 Test 指标双排打印对比！
        # # ==========================================
        # mv, mt, p = val_metrics, test_metrics, prefix
        
        # # 打印 Epoch 头部信息
        # print(f"Ep {epoch:<2} {is_best:<10} | T:{train_time:.1f}s V+Test:{val_time:.1f}s")
        # # 打印 Valid 指标
        # print(f"   [VAL ] AUUC:{mv.get(f'{p}_AUUC',0):.4f} | AUQC:{mv.get(f'{p}_AUQC',0):.4f} | L10:{mv.get(f'{p}_Lift@10',0):.4f} | AUC:{mv.get(f'{p}_AUC',0):.4f}")
        # # 打印 Test 指标
        # print(f"   [TEST] AUUC:{mt.get(f'{p}_AUUC',0):.4f} | AUQC:{mt.get(f'{p}_AUQC',0):.4f} | L10:{mt.get(f'{p}_Lift@10',0):.4f} | AUC:{mt.get(f'{p}_AUC',0):.4f}")
        # print("-" * 60) # 加个分割线更清爽

        # if epochs_no_improve >= patience:
        #     print(f"🛑 早停触发: 目标指标已连续 {patience} 个 Epoch 未提升。")
        #     break
        # # # 仅验证获取指标
        # # val_start = time.time()
        # # val_metrics, _ = evaluate_and_dump(model, valid_loader, device, trial_cfg["task"], max_steps=eval_max_steps, show_pbar=show_pbar)
        # # val_time = time.time() - val_start
        
        # # prefix = "Target_C" if trial_cfg["task"] == "train_c" else "Target_Y"
        # # target_metric = val_metrics.get(f"{prefix}_AUUC", 0)
        
        # # is_best = ""
        # # # 1. 局部早停监控
        # # if target_metric > local_best_metric:
        # #     local_best_metric = target_metric
        # #     epochs_no_improve = 0  
        # # else:
        # #     epochs_no_improve += 1
            
        # # # 🌟 2. 核心逻辑：全局打擂台！
        # # global_best = -float('inf')
        # # if os.path.exists(global_metric_path):
        # #     try:
        # #         with open(global_metric_path, 'r') as f:
        # #             content = f.read().strip()
        # #             if content: global_best = float(content)
        # #     except Exception: pass # 并发冲突时静默
            
        # # if target_metric > global_best:
        # #     try:
        # #         # 破纪录，挂牌
        # #         with open(global_metric_path, 'w') as f:
        # #             f.write(str(target_metric))
        # #         # 存模型
        # #         torch.save(model.state_dict(), best_ckpt_save_path)
        # #         # 存 Config（剥离掉不能 JSON 化的烂属性）
        # #         safe_cfg = {k: v for k, v in trial_cfg.items() if isinstance(v, (int, float, str, list, bool)) or v is None}
        # #         with open(best_config_save_path, 'w') as f:
        # #             json.dump(safe_cfg, f, ensure_ascii=False, indent=4)
        # #         is_best = "🌍(GLOBAL!)"
        # #     except Exception: pass
            
        # # safe_report({"epoch": epoch, "loss": epoch_loss / max(1, min(max_steps, len(train_loader))), **val_metrics})
        
        # # m, p = val_metrics, prefix
        # # print(f"Ep {epoch:<2} {is_best:<10} | T:{train_time:.1f}s V:{val_time:.1f}s | "
        # #       f"[{p}] AUUC:{m.get(f'{p}_AUUC',0):.4f} AUQC:{m.get(f'{p}_AUQC',0):.4f} "
        # #       f"L10:{m.get(f'{p}_Lift@10',0):.4f} PCOC:{m.get(f'{p}_PCOC',0):.4f} "
        # #       f"MAE:{m.get(f'{p}_MAE',0):.4f} Rec:{m.get(f'{p}_Recall',0):.4f} AUC:{m.get(f'{p}_AUC',0):.4f}")

        # # if epochs_no_improve >= patience:
        # #     print(f"🛑 早停触发: 目标指标已连续 {patience} 个 Epoch 未提升。")
        # #     break
    
    if mode == "reproduce":
        print("\n" + "🏁"*30)
        print(f"🏁 训练结束！加载 Seed {seed} 最佳权重进行终极评估...")
        if os.path.exists(best_ckpt_save_path):
            state = torch.load(best_ckpt_save_path, map_location=device)
            model.load_state_dict(state.get("model_state_dict", state))
            print(f"✅ [REPRODUCE] 加载本轮最佳权重成功，进入收尾评估...")
        run_final_evaluation(model, loaders_dict, device, trial_cfg)