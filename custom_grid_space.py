# -*- coding: utf-8 -*-
"""
💾 custom_grid_space.py
包含 17 组高分参数的纯净配置文件。
🌟 彻底解决参数丢失与残留 Bug：自动融合默认安全底座参数！
"""

# 👑 1. 定义大盘全量默认安全底座 (对应你给出的默认配置项)
DEFAULT_BASE_SPACE = {
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "hidden_dims": [128, 64, 32],
    "dropout_rate": 0.0,
    "batch_size": 65536 * 4,
    "accumulate_steps": 4,
    "model": "TARNET",
    
    "head_hidden_dims": None,
    "conflict_alpha_wool": 0.0,
    "conflict_alpha_gold": 0.0,
    "conflict_alpha_walkin": 0.0,
    "conflict_focal_type": "none",
    "conflict_gamma": 2.0,
    "conflict_global_margin": 1.0,
    "conflict_use_ohem": False,
    "conflict_ohem_pct": 0.10,
    "conflict_use_weight_clip": False,
    "conflict_max_weight_thres": 3.0,
    "conflict_two_stage_mode": False,
    "conflict_stage1_epochs": 0,
    "conflict_freeze_base_in_stage2": False,
    "ours_s6_use_logit_clamp": False,
    "ours_s6_clamp_val": 2.0,
    "ours_s6_temp": None  # 确保初始状态下温度干净
}

# 🟢 2. Pure V10 核心差异字典 (共 8 组)
RAW_PURE_V10 = {
    "y_pure_v10_h32_a1.0_wd1e4": {
        "model": "TARNET_Baseline_PureV10", "c_fusion_mode": "res_moe", "loss_types": ["prior_conflict"],
        "head_hidden_dims": [32], "conflict_mode": "all",
        "conflict_alpha_wool": 1.0, "conflict_alpha_gold": 1.0, "conflict_alpha_walkin": 1.0, "weight_decay": 0.0001
    },
    "y_pure_v10_h16_a5.0_wd1e5": {
        "model": "TARNET_Baseline_PureV10", "c_fusion_mode": "res_moe", "loss_types": ["prior_conflict"],
        "head_hidden_dims": [16], "conflict_mode": "all",
        "conflict_alpha_wool": 5.0, "conflict_alpha_gold": 5.0, "conflict_alpha_walkin": 5.0, "weight_decay": 1e-05
    },
    "y_pure_v10_h64_32_a0.5_wd1e4": {
        "model": "TARNET_Baseline_PureV10", "c_fusion_mode": "res_moe", "loss_types": ["prior_conflict"],
        "head_hidden_dims": [64, 32], "conflict_mode": "all",
        "conflict_alpha_wool": 0.5, "conflict_alpha_gold": 0.5, "conflict_alpha_walkin": 0.5, "weight_decay": 0.0001
    },
    "y_pure_v10_h16_a1.0_wd1e4": {
        "model": "TARNET_Baseline_PureV10", "c_fusion_mode": "res_moe", "loss_types": ["prior_conflict"],
        "head_hidden_dims": [16], "conflict_mode": "all",
        "conflict_alpha_wool": 1.0, "conflict_alpha_gold": 1.0, "conflict_alpha_walkin": 1.0, "weight_decay": 0.0001
    },
    "y_pure_v10_hNone_a0.5_5.0_1.0_wd1e5": {
        "model": "TARNET_Baseline_PureV10", "c_fusion_mode": "res_moe", "loss_types": ["prior_conflict"],
        "head_hidden_dims": None, "conflict_mode": "all",
        "conflict_alpha_wool": 0.5, "conflict_alpha_gold": 5.0, "conflict_alpha_walkin": 1.0, "weight_decay": 1e-05
    },
    "y_pure_v10_hNone_a0.5_10.0_0.5_wd1e5": {
        "model": "TARNET_Baseline_PureV10", "c_fusion_mode": "res_moe", "loss_types": ["prior_conflict"],
        "head_hidden_dims": None, "conflict_mode": "all",
        "conflict_alpha_wool": 0.5, "conflict_alpha_gold": 10.0, "conflict_alpha_walkin": 0.5, "weight_decay": 1e-05
    },
    "y_pure_v10_hNone_a0.5_0.5_0.1_wd1e5": {
        "model": "TARNET_Baseline_PureV10", "c_fusion_mode": "res_moe", "loss_types": ["prior_conflict"],
        "head_hidden_dims": None, "conflict_mode": "all",
        "conflict_alpha_wool": 0.5, "conflict_alpha_gold": 0.5, "conflict_alpha_walkin": 0.1, "weight_decay": 1e-05
    },
    "y_pure_v10_hNone_a0.5_0.5_5.0_wd1e5": {
        "model": "TARNET_Baseline_PureV10", "c_fusion_mode": "res_moe", "loss_types": ["prior_conflict"],
        "head_hidden_dims": None, "conflict_mode": "all",
        "conflict_alpha_wool": 0.5, "conflict_alpha_gold": 0.5, "conflict_alpha_walkin": 5.0, "weight_decay": 1e-05
    }
}

# 🔵 3. Ours V8 S6 核心差异字典 (共 9 组)
RAW_OURS_S6 = {
    "y_ours_v8s6_h32_a10.0_t1_wd0.01": {
        "model": "TARNET", "c_fusion_mode": "ours_v8_s6", "loss_types": ["prior_conflict"],
        "head_hidden_dims": [32], "conflict_mode": "all",
        "conflict_alpha_wool": 10.0, "conflict_alpha_gold": 10.0, "conflict_alpha_walkin": 10.0, "ours_s6_temp": 1.0, "weight_decay": 0.01
    },
    "y_ours_v8s6_h32_a0.1_t20_wd1e5": {
        "model": "TARNET", "c_fusion_mode": "ours_v8_s6", "loss_types": ["prior_conflict"],
        "head_hidden_dims": [32], "conflict_mode": "all",
        "conflict_alpha_wool": 0.1, "conflict_alpha_gold": 0.1, "conflict_alpha_walkin": 0.1, "ours_s6_temp": 20.0, "weight_decay": 1e-05
    },
    "y_ours_v8s6_h32_a0.5_t20_wd1e5": {
        "model": "TARNET", "c_fusion_mode": "ours_v8_s6", "loss_types": ["prior_conflict"],
        "head_hidden_dims": [32], "conflict_mode": "all",
        "conflict_alpha_wool": 0.5, "conflict_alpha_gold": 0.5, "conflict_alpha_walkin": 0.5, "ours_s6_temp": 20.0, "weight_decay": 1e-05
    },
    "y_ours_v8s6_h32_a10.0_t20_wd0.01": {
        "model": "TARNET", "c_fusion_mode": "ours_v8_s6", "loss_types": ["prior_conflict"],
        "head_hidden_dims": [32], "conflict_mode": "all",
        "conflict_alpha_wool": 10.0, "conflict_alpha_gold": 10.0, "conflict_alpha_walkin": 10.0, "ours_s6_temp": 20.0, "weight_decay": 0.01
    },
    "y_ours_v8s6_hNone_a1.0_t20_wd1e5": {
        "model": "TARNET", "c_fusion_mode": "ours_v8_s6", "loss_types": ["prior_conflict"],
        "head_hidden_dims": None, "conflict_mode": "all",
        "conflict_alpha_wool": 1.0, "conflict_alpha_gold": 1.0, "conflict_alpha_walkin": 1.0, "ours_s6_temp": 20.0, "weight_decay": 1e-05
    },
    "y_ours_v8s6_hNone_a1.0_t1_wd0.001": {
        "model": "TARNET", "c_fusion_mode": "ours_v8_s6", "loss_types": ["prior_conflict"],
        "head_hidden_dims": None, "conflict_mode": "all",
        "conflict_alpha_wool": 1.0, "conflict_alpha_gold": 1.0, "conflict_alpha_walkin": 1.0, "ours_s6_temp": 1.0, "weight_decay": 0.001
    },
    "y_ours_v8s6_h16_a0.1_t1_wd1e5": {
        "model": "TARNET", "c_fusion_mode": "ours_v8_s6", "loss_types": ["prior_conflict"],
        "head_hidden_dims": [16], "conflict_mode": "all",
        "conflict_alpha_wool": 0.1, "conflict_alpha_gold": 0.1, "conflict_alpha_walkin": 0.1, "ours_s6_temp": 1.0, "weight_decay": 1e-05
    },
    "y_ours_v8s6_h32_16_a5.0_t1_wd0.001": {
        "model": "TARNET", "c_fusion_mode": "ours_v8_s6", "loss_types": ["prior_conflict"],
        "head_hidden_dims": [32, 16], "conflict_mode": "all",
        "conflict_alpha_wool": 5.0, "conflict_alpha_gold": 5.0, "conflict_alpha_walkin": 5.0, "ours_s6_temp": 1.0, "weight_decay": 0.001
    },
    "y_ours_v8s6_h16_a10.0_t1_wd0.01": {
        "model": "TARNET", "c_fusion_mode": "ours_v8_s6", "loss_types": ["prior_conflict"],
        "head_hidden_dims": [16], "conflict_mode": "all",
        "conflict_alpha_wool": 10.0, "conflict_alpha_gold": 10.0, "conflict_alpha_walkin": 10.0, "ours_s6_temp": 1.0, "weight_decay": 0.01
    }
}

# 👑 4. 动态执行大合流：将默认底座拷贝，并用具体版本覆盖，完美粉碎残留与丢失隐患
ALL_CUSTOM_SPACES = {}

for version, diff_dict in {**RAW_PURE_V10, **RAW_OURS_S6}.items():
    # 每次都深拷贝一份干净的默认底座
    base_space_copied = DEFAULT_BASE_SPACE.copy()
    # 用差异项去更新覆盖它
    base_space_copied.update(diff_dict)
    # 落盘到总字典
    ALL_CUSTOM_SPACES[version] = base_space_copied