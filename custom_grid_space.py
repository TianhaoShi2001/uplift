# -*- coding: utf-8 -*-
"""
💾 custom_grid_space.py (追加 Baseline 版本)
"""

# 👑 1. 全量默认安全底座 (保持你原有的不变)
DEFAULT_BASE_SPACE = {
    "learning_rate": 1e-3, "weight_decay": 1e-5, "hidden_dims": [128, 64, 32], "dropout_rate": 0.0,
    "batch_size": 262144, "accumulate_steps": 4, "model": "TARNET",
    "head_hidden_dims": None,
    "conflict_alpha_wool": 0.0, "conflict_alpha_gold": 0.0, "conflict_alpha_walkin": 0.0, "conflict_alpha": 0.0,
    "conflict_focal_type": "none", "conflict_gamma": 2.0, "conflict_global_margin": 1.0,
    "conflict_use_ohem": False, "conflict_ohem_pct": 0.10, "conflict_use_weight_clip": False, "conflict_max_weight_thres": 3.0,
    "conflict_two_stage_mode": False, "conflict_stage1_epochs": 0, "conflict_freeze_base_in_stage2": False,
    "ours_s6_use_logit_clamp": False, "ours_s6_clamp_val": 2.0, "ours_s6_temp": None
}

# 🏁 2. 十大 Baseline 纯净核心字典
RAW_BASELINES = {
    # 1. CFRNet
    "y_baseline_cfrnet_2": {
        "model": "CFRNet", "weight_decay": 1e-06, "hidden_dims": [64, 32], "batch_size": 262144, "cfr_weight": 0.01
    },
    # 2. DragonNet
    "y_baseline_dragonnet": {
        "model": "DragonNet", "weight_decay": 0.001, "hidden_dims": [64, 32], "batch_size": 262144, "dragon_alpha": 1.0, "dragon_beta": 1.0
    },
    # 3. ECUP
    "y_ecup": {
        "model": "ECUP", "weight_decay": 1e-05, "hidden_dims": [128, 64, 32], "batch_size": 65536, "loss_types": ["bce"], 
        "d_dim": 32, "tower_h": 128, "tae_h": 128, "num_heads": 2, "gamma": 1.0, "ctcvr_weight": 1.0
    },
    # 4. EUEN
    "y_baseline_euen_academic_ms": {
        "model": "EUEN_Academic", "weight_decay": 0.001, "hidden_dims": [64, 32], "batch_size": 262144, "loss_types": ["bce"]
    },
    # 5. EFIN
    "y_baseline_efin": {
        "model": "EFIN_ours", "weight_decay": 1e-06, "hidden_dims": [64, 32], "batch_size": 262144, "efin_embed_dim": 32, "efin_lambda": 0.1
    },
    # 6. MOTTO
    "y_motto": {
        "model": "MOTTO", "weight_decay": 1e-05, "hidden_dims": [128, 64, 32], "batch_size": 262144, "loss_types": ["bce"], 
        "d_dim": 16, "bottom_dim": 64, "expert_hidden_dims": [128, 64, 32], "tower_dim": 64, "use_specific_experts": True, "alpha_sda": 5.0, "aux_weight": 0.5
    },
    # 7. MTMT MLP
    "y_mtmt_mlp": {
        "model": "MTMT", "weight_decay": 0.0001, "hidden_dims": [128, 64, 32], "dropout_rate": 0.1, "batch_size": 262144, "loss_types": ["bce"],
        "expert_type": "mlp", "expert_hidden_dims": [64, 32], "num_experts": 4, "t_emb_dim": 16, "aux_weight": 0.1
    },
    # 8. S-Learner
    "y_baseline_s_learner": {
        "model": "S_Learner", "weight_decay": 0.001, "hidden_dims": [64], "batch_size": 262144
    },
    # 9. T-Learner
    "y_baseline_t_learner": {
        "model": "T_Learner", "weight_decay": 0.01, "hidden_dims": [128, 64, 32], "batch_size": 262144
    },
    # 10. TARNET (Base)
    "y_v1_base": {
        "model": "TARNET", "c_fusion_mode": "none", "weight_decay": 1e-06, "hidden_dims": [128, 64, 32], "batch_size": 262144, "loss_types": ["bce"]
    }
}

# 🚀 3. 动态执行大合流 (确保你在原本的合流字典里带上 RAW_BASELINES)
ALL_CUSTOM_SPACES = {}

# 假设你保留了之前的 RAW_OURS_S4, RAW_OURS_S6 等，这里直接加入 RAW_BASELINES 一起更新
for version, diff_dict in {**RAW_BASELINES}.items(): # 如果有其他配置字典，用 **RAW_OURS_S4, **RAW_BASELINES 的方式解包
    base_space_copied = DEFAULT_BASE_SPACE.copy()
    base_space_copied.update(diff_dict)
    ALL_CUSTOM_SPACES[version] = base_space_copied


RAW_NEW_TOP3 = {
    "s6_new_top1_auuc9145_hNone_a1.0_t1_wd1e5": {
        "model": "TARNET", "c_fusion_mode": "ours_s6_conflict", "loss_types": ["prior_conflict"],
        "head_hidden_dims": None, "conflict_mode": "all",
        "conflict_alpha_wool": 1.0, "conflict_alpha_gold": 1.0, "conflict_alpha_walkin": 1.0,
        "ours_s6_temp": 1.0, "weight_decay": 1e-05
    },
    "s6_new_top2_auuc9105_hNone_a10.0_t1_wd1e4": {
        "model": "TARNET", "c_fusion_mode": "ours_s6_conflict", "loss_types": ["prior_conflict"],
        "head_hidden_dims": None, "conflict_mode": "all",
        "conflict_alpha_wool": 10.0, "conflict_alpha_gold": 10.0, "conflict_alpha_walkin": 10.0,
        "ours_s6_temp": 1.0, "weight_decay": 0.0001
    },
    "s6_new_top3_auuc9094_hNone_a0.1_t1_wd1e5": {
        "model": "TARNET", "c_fusion_mode": "ours_s6_conflict", "loss_types": ["prior_conflict"],
        "head_hidden_dims": None, "conflict_mode": "all",
        "conflict_alpha_wool": 0.1, "conflict_alpha_gold": 0.1, "conflict_alpha_walkin": 0.1,
        "ours_s6_temp": 1.0, "weight_decay": 1e-05
    }
}

# 确保它被合流入你的 ALL_CUSTOM_SPACES
for version, diff_dict in RAW_NEW_TOP3.items():
    base_space_copied = DEFAULT_BASE_SPACE.copy()
    base_space_copied.update(diff_dict)
    ALL_CUSTOM_SPACES[version] = base_space_copied

# # -*- coding: utf-8 -*-
# """
# 💾 custom_grid_space.py
# 包含最新的 Ours S4、Ours S6、Pure V10 搜索最优配置 (带原始 AUUC 得分命名)
# 🌟 彻底解决参数丢失与残留 Bug：自动融合默认安全底座参数！
# """

# # 👑 1. 全量默认安全底座
# DEFAULT_BASE_SPACE = {
#     "learning_rate": 1e-3, "weight_decay": 1e-5, "hidden_dims": [128, 64, 32], "dropout_rate": 0.0,
#     "batch_size": 65536 * 4, "accumulate_steps": 4, "model": "TARNET",
#     "head_hidden_dims": None,
#     "conflict_alpha_wool": 0.0, "conflict_alpha_gold": 0.0, "conflict_alpha_walkin": 0.0, "conflict_alpha": 0.0,
#     "conflict_focal_type": "none", "conflict_gamma": 2.0, "conflict_global_margin": 1.0,
#     "conflict_use_ohem": False, "conflict_ohem_pct": 0.10, "conflict_use_weight_clip": False, "conflict_max_weight_thres": 3.0,
#     "conflict_two_stage_mode": False, "conflict_stage1_epochs": 0, "conflict_freeze_base_in_stage2": False,
#     "ours_s6_use_logit_clamp": False, "ours_s6_clamp_val": 2.0, "ours_s6_temp": None
# }

# # 🟢 2. Ours S4 (共 7 组，去除了完全重复的第四名)
# RAW_OURS_S4 = {
#     "s4_top1_auuc9102_hNone_a5.0_wd1e4": { "model": "TARNET", "c_fusion_mode": "ours_s4_conflict", "loss_types": ["prior_conflict"], "head_hidden_dims": None, "conflict_mode": "all", "conflict_alpha_wool": 5.0, "conflict_alpha_gold": 5.0, "conflict_alpha_walkin": 5.0, "weight_decay": 0.0001 },
#     "s4_top2_auuc9096_hNone_a1.0_wd1e2": { "model": "TARNET", "c_fusion_mode": "ours_s4_conflict", "loss_types": ["prior_conflict"], "head_hidden_dims": None, "conflict_mode": "all", "conflict_alpha_wool": 1.0, "conflict_alpha_gold": 1.0, "conflict_alpha_walkin": 1.0, "weight_decay": 0.01 },
#     "s4_top3_auuc9094_h64_32_a0.1_wd1e5": { "model": "TARNET", "c_fusion_mode": "ours_s4_conflict", "loss_types": ["prior_conflict"], "head_hidden_dims": [64, 32], "conflict_mode": "all", "conflict_alpha_wool": 0.1, "conflict_alpha_gold": 0.1, "conflict_alpha_walkin": 0.1, "weight_decay": 1e-05 },
#     "s4_top4_auuc9089_hNone_a5.0_wd1e5": { "model": "TARNET", "c_fusion_mode": "ours_s4_conflict", "loss_types": ["prior_conflict"], "head_hidden_dims": None, "conflict_mode": "all", "conflict_alpha_wool": 5.0, "conflict_alpha_gold": 5.0, "conflict_alpha_walkin": 5.0, "weight_decay": 1e-05 },
#     "s4_top5_auuc9086_hNone_a1.0_wd1e3": { "model": "TARNET", "c_fusion_mode": "ours_s4_conflict", "loss_types": ["prior_conflict"], "head_hidden_dims": None, "conflict_mode": "all", "conflict_alpha_wool": 1.0, "conflict_alpha_gold": 1.0, "conflict_alpha_walkin": 1.0, "weight_decay": 0.001 },
#     "s4_top6_auuc9080_hNone_a5.0_wd1e2": { "model": "TARNET", "c_fusion_mode": "ours_s4_conflict", "loss_types": ["prior_conflict"], "head_hidden_dims": None, "conflict_mode": "all", "conflict_alpha_wool": 5.0, "conflict_alpha_gold": 5.0, "conflict_alpha_walkin": 5.0, "weight_decay": 0.01 },
#     "s4_top7_auuc9079_hNone_a10.0_wd1e5": { "model": "TARNET", "c_fusion_mode": "ours_s4_conflict", "loss_types": ["prior_conflict"], "head_hidden_dims": None, "conflict_mode": "all", "conflict_alpha_wool": 10.0, "conflict_alpha_gold": 10.0, "conflict_alpha_walkin": 10.0, "weight_decay": 1e-05 },
# }

# # 🔵 3. Ours S6 核心差异字典 (共 9 组)
# RAW_OURS_S6 = {
#     "s6_top1_auuc9104_hNone_a0.1_t1_wd1e3": { "model": "TARNET", "c_fusion_mode": "ours_s6_conflict", "loss_types": ["prior_conflict"], "head_hidden_dims": None, "conflict_mode": "all", "conflict_alpha_wool": 0.1, "conflict_alpha_gold": 0.1, "conflict_alpha_walkin": 0.1, "ours_s6_temp": 1.0, "weight_decay": 0.001 },
#     "s6_top2_auuc9088_hNone_a5.0_t1_wd1e3": { "model": "TARNET", "c_fusion_mode": "ours_s6_conflict", "loss_types": ["prior_conflict"], "head_hidden_dims": None, "conflict_mode": "all", "conflict_alpha_wool": 5.0, "conflict_alpha_gold": 5.0, "conflict_alpha_walkin": 5.0, "ours_s6_temp": 1.0, "weight_decay": 0.001 },
#     "s6_top3_auuc9086_hNone_a0.1_t1_wd1e5": { "model": "TARNET", "c_fusion_mode": "ours_s6_conflict", "loss_types": ["prior_conflict"], "head_hidden_dims": None, "conflict_mode": "all", "conflict_alpha_wool": 0.1, "conflict_alpha_gold": 0.1, "conflict_alpha_walkin": 0.1, "ours_s6_temp": 1.0, "weight_decay": 1e-05 },
#     "s6_top4_auuc9084_hNone_a5.0_t1_wd1e5": { "model": "TARNET", "c_fusion_mode": "ours_s6_conflict", "loss_types": ["prior_conflict"], "head_hidden_dims": None, "conflict_mode": "all", "conflict_alpha_wool": 5.0, "conflict_alpha_gold": 5.0, "conflict_alpha_walkin": 5.0, "ours_s6_temp": 1.0, "weight_decay": 1e-05 },
#     "s6_top5_auuc9084_hNone_a5.0_t1_wd1e4": { "model": "TARNET", "c_fusion_mode": "ours_s6_conflict", "loss_types": ["prior_conflict"], "head_hidden_dims": None, "conflict_mode": "all", "conflict_alpha_wool": 5.0, "conflict_alpha_gold": 5.0, "conflict_alpha_walkin": 5.0, "ours_s6_temp": 1.0, "weight_decay": 0.0001 },
#     "s6_top6_auuc9078_hNone_a1.0_t20_wd1e4": { "model": "TARNET", "c_fusion_mode": "ours_s6_conflict", "loss_types": ["prior_conflict"], "head_hidden_dims": None, "conflict_mode": "all", "conflict_alpha_wool": 1.0, "conflict_alpha_gold": 1.0, "conflict_alpha_walkin": 1.0, "ours_s6_temp": 20.0, "weight_decay": 0.0001 },
#     "s6_top7_auuc9077_hNone_a1.0_t1_wd1e4": { "model": "TARNET", "c_fusion_mode": "ours_s6_conflict", "loss_types": ["prior_conflict"], "head_hidden_dims": None, "conflict_mode": "all", "conflict_alpha_wool": 1.0, "conflict_alpha_gold": 1.0, "conflict_alpha_walkin": 1.0, "ours_s6_temp": 1.0, "weight_decay": 0.0001 },
#     "s6_top8_auuc9069_hNone_a1.0_t10_wd1e3": { "model": "TARNET", "c_fusion_mode": "ours_s6_conflict", "loss_types": ["prior_conflict"], "head_hidden_dims": None, "conflict_mode": "all", "conflict_alpha_wool": 1.0, "conflict_alpha_gold": 1.0, "conflict_alpha_walkin": 1.0, "ours_s6_temp": 10.0, "weight_decay": 0.001 },
#     "s6_top9_auuc9068_hNone_a5.0_t1_wd1e2": { "model": "TARNET", "c_fusion_mode": "ours_s6_conflict", "loss_types": ["prior_conflict"], "head_hidden_dims": None, "conflict_mode": "all", "conflict_alpha_wool": 5.0, "conflict_alpha_gold": 5.0, "conflict_alpha_walkin": 5.0, "ours_s6_temp": 1.0, "weight_decay": 0.01 },
# }

# # 👑 4. Pure V10 核心差异字典 (共 12 组)
# RAW_PURE_V10 = {
#     "v10_top1_auuc9105_hNone_a0.0_wd1e3": { "model": "TARNET_Baseline_PureV10", "c_fusion_mode": "res_moe", "loss_types": ["prior_conflict"], "head_hidden_dims": None, "conflict_mode": "all", "conflict_alpha_wool": 0.0, "conflict_alpha_gold": 0.0, "conflict_alpha_walkin": 0.0, "weight_decay": 0.001 },
#     "v10_top2_auuc9105_hNone_a0.01_wd1e3": { "model": "TARNET_Baseline_PureV10", "c_fusion_mode": "res_moe", "loss_types": ["prior_conflict"], "head_hidden_dims": None, "conflict_mode": "all", "conflict_alpha_wool": 0.01, "conflict_alpha_gold": 0.01, "conflict_alpha_walkin": 0.01, "weight_decay": 0.001 },
#     "v10_top3_auuc9104_hNone_a0.1_wd1e3": { "model": "TARNET_Baseline_PureV10", "c_fusion_mode": "res_moe", "loss_types": ["prior_conflict"], "head_hidden_dims": None, "conflict_mode": "all", "conflict_alpha_wool": 0.1, "conflict_alpha_gold": 0.1, "conflict_alpha_walkin": 0.1, "weight_decay": 0.001 },
#     "v10_top4_auuc9102_hNone_a0.5_wd1e3": { "model": "TARNET_Baseline_PureV10", "c_fusion_mode": "res_moe", "loss_types": ["prior_conflict"], "head_hidden_dims": None, "conflict_mode": "all", "conflict_alpha_wool": 0.5, "conflict_alpha_gold": 0.5, "conflict_alpha_walkin": 0.5, "weight_decay": 0.001 },
#     "v10_top5_auuc9099_hNone_a0.05_wd1e3": { "model": "TARNET_Baseline_PureV10", "c_fusion_mode": "res_moe", "loss_types": ["prior_conflict"], "head_hidden_dims": None, "conflict_mode": "all", "conflict_alpha_wool": 0.05, "conflict_alpha_gold": 0.05, "conflict_alpha_walkin": 0.05, "weight_decay": 0.001 },
#     "v10_top6_auuc9088_hNone_a0.5_wd1e5": { "model": "TARNET_Baseline_PureV10", "c_fusion_mode": "res_moe", "loss_types": ["prior_conflict"], "head_hidden_dims": None, "conflict_mode": "all", "conflict_alpha_wool": 0.5, "conflict_alpha_gold": 0.5, "conflict_alpha_walkin": 0.5, "weight_decay": 1e-05 },
#     "v10_top7_auuc9087_hNone_a0.1_wd1e5": { "model": "TARNET_Baseline_PureV10", "c_fusion_mode": "res_moe", "loss_types": ["prior_conflict"], "head_hidden_dims": None, "conflict_mode": "all", "conflict_alpha_wool": 0.1, "conflict_alpha_gold": 0.1, "conflict_alpha_walkin": 0.1, "weight_decay": 1e-05 },
#     "v10_top8_auuc9087_hNone_a5.0_wd1e3": { "model": "TARNET_Baseline_PureV10", "c_fusion_mode": "res_moe", "loss_types": ["prior_conflict"], "head_hidden_dims": None, "conflict_mode": "all", "conflict_alpha_wool": 5.0, "conflict_alpha_gold": 5.0, "conflict_alpha_walkin": 5.0, "weight_decay": 0.001 },
#     "v10_top9_auuc9084_hNone_a0.5_wd1e4": { "model": "TARNET_Baseline_PureV10", "c_fusion_mode": "res_moe", "loss_types": ["prior_conflict"], "head_hidden_dims": None, "conflict_mode": "all", "conflict_alpha_wool": 0.5, "conflict_alpha_gold": 0.5, "conflict_alpha_walkin": 0.5, "weight_decay": 0.0001 },
#     "v10_top10_auuc9083_hNone_a10.0_wd1e2": { "model": "TARNET_Baseline_PureV10", "c_fusion_mode": "res_moe", "loss_types": ["prior_conflict"], "head_hidden_dims": None, "conflict_mode": "all", "conflict_alpha_wool": 10.0, "conflict_alpha_gold": 10.0, "conflict_alpha_walkin": 10.0, "weight_decay": 0.01 },
#     "v10_top11_auuc9079_h64_32_a5.0_wd1e2": { "model": "TARNET_Baseline_PureV10", "c_fusion_mode": "res_moe", "loss_types": ["prior_conflict"], "head_hidden_dims": [64, 32], "conflict_mode": "all", "conflict_alpha_wool": 5.0, "conflict_alpha_gold": 5.0, "conflict_alpha_walkin": 5.0, "weight_decay": 0.01 },
#     "v10_top12_auuc9077_hNone_a10.0_wd1e5": { "model": "TARNET_Baseline_PureV10", "c_fusion_mode": "res_moe", "loss_types": ["prior_conflict"], "head_hidden_dims": None, "conflict_mode": "all", "conflict_alpha_wool": 10.0, "conflict_alpha_gold": 10.0, "conflict_alpha_walkin": 10.0, "weight_decay": 1e-05 },
# }

# # 🚀 5. 动态执行大合流
# ALL_CUSTOM_SPACES = {}

# for version, diff_dict in {**RAW_OURS_S4, **RAW_OURS_S6, **RAW_PURE_V10}.items():
#     base_space_copied = DEFAULT_BASE_SPACE.copy()
#     base_space_copied.update(diff_dict)
#     ALL_CUSTOM_SPACES[version] = base_space_copied


# # # -*- coding: utf-8 -*-
# # """
# # 💾 custom_grid_space.py
# # 包含 17 组高分参数的纯净配置文件。
# # 🌟 彻底解决参数丢失与残留 Bug：自动融合默认安全底座参数！
# # """

# # # 👑 1. 定义大盘全量默认安全底座 (对应你给出的默认配置项)
# # DEFAULT_BASE_SPACE = {
# #     "learning_rate": 1e-3,
# #     "weight_decay": 1e-5,
# #     "hidden_dims": [128, 64, 32],
# #     "dropout_rate": 0.0,
# #     "batch_size": 65536 * 4,
# #     "accumulate_steps": 4,
# #     "model": "TARNET",
    
# #     "head_hidden_dims": None,
# #     "conflict_alpha_wool": 0.0,
# #     "conflict_alpha_gold": 0.0,
# #     "conflict_alpha_walkin": 0.0,
# #     "conflict_focal_type": "none",
# #     "conflict_gamma": 2.0,
# #     "conflict_global_margin": 1.0,
# #     "conflict_use_ohem": False,
# #     "conflict_ohem_pct": 0.10,
# #     "conflict_use_weight_clip": False,
# #     "conflict_max_weight_thres": 3.0,
# #     "conflict_two_stage_mode": False,
# #     "conflict_stage1_epochs": 0,
# #     "conflict_freeze_base_in_stage2": False,
# #     "ours_s6_use_logit_clamp": False,
# #     "ours_s6_clamp_val": 2.0,
# #     "ours_s6_temp": None  # 确保初始状态下温度干净
# # }

# # # 🟢 2. Pure V10 核心差异字典 (共 8 组)
# # RAW_PURE_V10 = {
# #     "y_pure_v10_h32_a1.0_wd1e4": {
# #         "model": "TARNET_Baseline_PureV10", "c_fusion_mode": "res_moe", "loss_types": ["prior_conflict"],
# #         "head_hidden_dims": [32], "conflict_mode": "all",
# #         "conflict_alpha_wool": 1.0, "conflict_alpha_gold": 1.0, "conflict_alpha_walkin": 1.0, "weight_decay": 0.0001
# #     },
# #     "y_pure_v10_h16_a5.0_wd1e5": {
# #         "model": "TARNET_Baseline_PureV10", "c_fusion_mode": "res_moe", "loss_types": ["prior_conflict"],
# #         "head_hidden_dims": [16], "conflict_mode": "all",
# #         "conflict_alpha_wool": 5.0, "conflict_alpha_gold": 5.0, "conflict_alpha_walkin": 5.0, "weight_decay": 1e-05
# #     },
# #     "y_pure_v10_h64_32_a0.5_wd1e4": {
# #         "model": "TARNET_Baseline_PureV10", "c_fusion_mode": "res_moe", "loss_types": ["prior_conflict"],
# #         "head_hidden_dims": [64, 32], "conflict_mode": "all",
# #         "conflict_alpha_wool": 0.5, "conflict_alpha_gold": 0.5, "conflict_alpha_walkin": 0.5, "weight_decay": 0.0001
# #     },
# #     "y_pure_v10_h16_a1.0_wd1e4": {
# #         "model": "TARNET_Baseline_PureV10", "c_fusion_mode": "res_moe", "loss_types": ["prior_conflict"],
# #         "head_hidden_dims": [16], "conflict_mode": "all",
# #         "conflict_alpha_wool": 1.0, "conflict_alpha_gold": 1.0, "conflict_alpha_walkin": 1.0, "weight_decay": 0.0001
# #     },
# #     "y_pure_v10_hNone_a0.5_5.0_1.0_wd1e5": {
# #         "model": "TARNET_Baseline_PureV10", "c_fusion_mode": "res_moe", "loss_types": ["prior_conflict"],
# #         "head_hidden_dims": None, "conflict_mode": "all",
# #         "conflict_alpha_wool": 0.5, "conflict_alpha_gold": 5.0, "conflict_alpha_walkin": 1.0, "weight_decay": 1e-05
# #     },
# #     "y_pure_v10_hNone_a0.5_10.0_0.5_wd1e5": {
# #         "model": "TARNET_Baseline_PureV10", "c_fusion_mode": "res_moe", "loss_types": ["prior_conflict"],
# #         "head_hidden_dims": None, "conflict_mode": "all",
# #         "conflict_alpha_wool": 0.5, "conflict_alpha_gold": 10.0, "conflict_alpha_walkin": 0.5, "weight_decay": 1e-05
# #     },
# #     "y_pure_v10_hNone_a0.5_0.5_0.1_wd1e5": {
# #         "model": "TARNET_Baseline_PureV10", "c_fusion_mode": "res_moe", "loss_types": ["prior_conflict"],
# #         "head_hidden_dims": None, "conflict_mode": "all",
# #         "conflict_alpha_wool": 0.5, "conflict_alpha_gold": 0.5, "conflict_alpha_walkin": 0.1, "weight_decay": 1e-05
# #     },
# #     "y_pure_v10_hNone_a0.5_0.5_5.0_wd1e5": {
# #         "model": "TARNET_Baseline_PureV10", "c_fusion_mode": "res_moe", "loss_types": ["prior_conflict"],
# #         "head_hidden_dims": None, "conflict_mode": "all",
# #         "conflict_alpha_wool": 0.5, "conflict_alpha_gold": 0.5, "conflict_alpha_walkin": 5.0, "weight_decay": 1e-05
# #     }
# # }

# # # 🔵 3. Ours V8 S6 核心差异字典 (共 9 组)
# # RAW_OURS_S6 = {
# #     "y_ours_v8s6_h32_a10.0_t1_wd0.01": {
# #         "model": "TARNET", "c_fusion_mode": "ours_v8_s6", "loss_types": ["prior_conflict"],
# #         "head_hidden_dims": [32], "conflict_mode": "all",
# #         "conflict_alpha_wool": 10.0, "conflict_alpha_gold": 10.0, "conflict_alpha_walkin": 10.0, "ours_s6_temp": 1.0, "weight_decay": 0.01
# #     },
# #     "y_ours_v8s6_h32_a0.1_t20_wd1e5": {
# #         "model": "TARNET", "c_fusion_mode": "ours_v8_s6", "loss_types": ["prior_conflict"],
# #         "head_hidden_dims": [32], "conflict_mode": "all",
# #         "conflict_alpha_wool": 0.1, "conflict_alpha_gold": 0.1, "conflict_alpha_walkin": 0.1, "ours_s6_temp": 20.0, "weight_decay": 1e-05
# #     },
# #     "y_ours_v8s6_h32_a0.5_t20_wd1e5": {
# #         "model": "TARNET", "c_fusion_mode": "ours_v8_s6", "loss_types": ["prior_conflict"],
# #         "head_hidden_dims": [32], "conflict_mode": "all",
# #         "conflict_alpha_wool": 0.5, "conflict_alpha_gold": 0.5, "conflict_alpha_walkin": 0.5, "ours_s6_temp": 20.0, "weight_decay": 1e-05
# #     },
# #     "y_ours_v8s6_h32_a10.0_t20_wd0.01": {
# #         "model": "TARNET", "c_fusion_mode": "ours_v8_s6", "loss_types": ["prior_conflict"],
# #         "head_hidden_dims": [32], "conflict_mode": "all",
# #         "conflict_alpha_wool": 10.0, "conflict_alpha_gold": 10.0, "conflict_alpha_walkin": 10.0, "ours_s6_temp": 20.0, "weight_decay": 0.01
# #     },
# #     "y_ours_v8s6_hNone_a1.0_t20_wd1e5": {
# #         "model": "TARNET", "c_fusion_mode": "ours_v8_s6", "loss_types": ["prior_conflict"],
# #         "head_hidden_dims": None, "conflict_mode": "all",
# #         "conflict_alpha_wool": 1.0, "conflict_alpha_gold": 1.0, "conflict_alpha_walkin": 1.0, "ours_s6_temp": 20.0, "weight_decay": 1e-05
# #     },
# #     "y_ours_v8s6_hNone_a1.0_t1_wd0.001": {
# #         "model": "TARNET", "c_fusion_mode": "ours_v8_s6", "loss_types": ["prior_conflict"],
# #         "head_hidden_dims": None, "conflict_mode": "all",
# #         "conflict_alpha_wool": 1.0, "conflict_alpha_gold": 1.0, "conflict_alpha_walkin": 1.0, "ours_s6_temp": 1.0, "weight_decay": 0.001
# #     },
# #     "y_ours_v8s6_h16_a0.1_t1_wd1e5": {
# #         "model": "TARNET", "c_fusion_mode": "ours_v8_s6", "loss_types": ["prior_conflict"],
# #         "head_hidden_dims": [16], "conflict_mode": "all",
# #         "conflict_alpha_wool": 0.1, "conflict_alpha_gold": 0.1, "conflict_alpha_walkin": 0.1, "ours_s6_temp": 1.0, "weight_decay": 1e-05
# #     },
# #     "y_ours_v8s6_h32_16_a5.0_t1_wd0.001": {
# #         "model": "TARNET", "c_fusion_mode": "ours_v8_s6", "loss_types": ["prior_conflict"],
# #         "head_hidden_dims": [32, 16], "conflict_mode": "all",
# #         "conflict_alpha_wool": 5.0, "conflict_alpha_gold": 5.0, "conflict_alpha_walkin": 5.0, "ours_s6_temp": 1.0, "weight_decay": 0.001
# #     },
# #     "y_ours_v8s6_h16_a10.0_t1_wd0.01": {
# #         "model": "TARNET", "c_fusion_mode": "ours_v8_s6", "loss_types": ["prior_conflict"],
# #         "head_hidden_dims": [16], "conflict_mode": "all",
# #         "conflict_alpha_wool": 10.0, "conflict_alpha_gold": 10.0, "conflict_alpha_walkin": 10.0, "ours_s6_temp": 1.0, "weight_decay": 0.01
# #     }
# # }

# # # 👑 4. 动态执行大合流：将默认底座拷贝，并用具体版本覆盖，完美粉碎残留与丢失隐患
# # ALL_CUSTOM_SPACES = {}

# # for version, diff_dict in {**RAW_PURE_V10, **RAW_OURS_S6}.items():
# #     # 每次都深拷贝一份干净的默认底座
# #     base_space_copied = DEFAULT_BASE_SPACE.copy()
# #     # 用差异项去更新覆盖它
# #     base_space_copied.update(diff_dict)
# #     # 落盘到总字典
# #     ALL_CUSTOM_SPACES[version] = base_space_copied