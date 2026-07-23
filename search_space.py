# ==========================================
# 工业级调参弹药库 (纯 Grid Search 穷举版)
# ==========================================

def get_default_hyperparams(task="train_y", version="v1_baseline"):
    """
    默认的超参数配置 (用于本地单次 Debug / 一键验证)
    """
    base_params = {
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "hidden_dims": [64, 32],
        "dropout_rate": 0.1,
        "batch_size": 2048,
        "num_epochs": 50,
        "accumulate_steps": 1, # 👉 Debug 模式默认不累加，方便排错
    }
    
    if task == "train_c":
        # C 的默认兜底
        base_params.update({
            "focal_gamma": 2.0,
            "focal_alpha": 0.25,
            "align_method": "swd",     # 🌟 默认设为 SWD，或者 moment
            "align_weight": 0.1,       # 🌟 替代原来的 mmd_alpha
            "mmd_sigma": 1.0
        })
    elif task == "train_y":
        # Y 的 version 控制 Debug 时的行为
        if version == "v1_baseline":
            base_params.update({
                "c_fusion_mode": "none",
                "loss_types": ["bce"],
            })
        else: # 比如 v2_full_magic
            base_params.update({
                "c_fusion_mode": "joint_emb",
                "c_embedding_dim": 4,
                "loss_types": ["strata_weighted", "pairwise", "variance_reg"],
                "rank_margin": 0.05,
                "rank_alpha": 1.0,
                "never_threshold": 0.9,
                "var_lambda": 1.0
            })
        
    return base_params

def get_ray_search_space(task="train_y", version="v1_baseline"):
    """
    大规模调参空间 (无缝对接 Ray Tune)
    通过 task 和 version 共同决定抛出什么空间，全部使用 grid_search！
    """
    try:
        from ray import tune
    except ImportError:
        print("⚠️ 警告: 未安装 ray[tune]。")
        return {}

    space = {
        "learning_rate": tune.grid_search([1e-3]),
        "weight_decay": tune.grid_search([1e-5, 1e-4, 1e-3, 1e-2]), # , 1e-3
        "hidden_dims": tune.grid_search([[128, 64, 32]]), 
        "dropout_rate": tune.grid_search([0.0]),
        "batch_size": tune.grid_search([65536*4]),
        "accumulate_steps": tune.grid_search([4]),
        'model': tune.grid_search(['TARNET']),
        
        # 👑 [预留高级接口：默认安全底座关闭]
        "head_hidden_dims": tune.grid_search([None]),               # 默认不传 head_hidden_dims，保持旧版本裸线性头向下兼容
        "conflict_alpha_wool": tune.grid_search([0.0]),             # 默认关闭三人群解耦，保持原版 conflict_alpha 控场
        "conflict_alpha_gold": tune.grid_search([0.0]),
        "conflict_alpha_walkin": tune.grid_search([0.0]),
        "conflict_focal_type": tune.grid_search(["none"]),         # 默认不截断，不启动全局保底版 Focal
        "conflict_gamma": tune.grid_search([2.0]),
        "conflict_global_margin": tune.grid_search([1.0]),
        "conflict_use_ohem": tune.grid_search([False]),             # 默认关闭困难样本挖掘
        "conflict_ohem_pct": tune.grid_search([0.10]),
        "conflict_use_weight_clip": tune.grid_search([False]),       # 默认关闭 Loss 侧权重裁剪
        "conflict_max_weight_thres": tune.grid_search([3.0]),
        "conflict_two_stage_mode": tune.grid_search([False]),       # 默认关闭训练时空两阶段流
        "conflict_stage1_epochs": tune.grid_search([0]),
        "conflict_freeze_base_in_stage2": tune.grid_search([False]),
        "ours_s6_use_logit_clamp": tune.grid_search([False]),       # 默认关闭 S6 前向修正 Logit 空间裁剪
        "ours_s6_clamp_val": tune.grid_search([2.0])
    }

    # 1. 公共搜索空间 (全局 Grid)
    # space = {
    #     "learning_rate": tune.grid_search([1e-3]),
    #     "weight_decay": tune.grid_search([1e-6, 1e-5, 1e-4, 1e-3, 1e-2]),
    #     # "hidden_dims": tune.grid_search([[128, 64, 32]]),
    #     "hidden_dims": tune.grid_search([[128, 64, 32]]), # tune.grid_search([[128, 64, 32]]),
    #     # tune.grid_search([[128, 64, 32]]), [64], [64, 32], 
    #     #   tune.grid_search([[64], [64, 32], [128, 64, 32]]),
    #     # tune.grid_search([[128, 64, 32]]), 
    #     # tune.grid_search([[64], [64, 32], [128, 64, 32]]),
    #     #  尝试中间版本的时候删掉64 和 64 32，提速度。
    #     "dropout_rate": tune.grid_search([0.0]),
    #     "batch_size": tune.grid_search([65536*4]),
    #     "accumulate_steps": tune.grid_search([4]),
    #     'model':tune.grid_search(['TARNET'])
    # }

    # ==========================================
    # 2. Stage 1 (C 模型) 探索空间 - 严控版本
    # ==========================================
    if task == "train_c":
        if version == "c_v1_base":
            space.update({
                "focal_gamma": tune.grid_search([0.0]),  # 纯 BCE
                "focal_alpha": tune.grid_search([0.5]),  # 绝对公平
                "align_method": tune.grid_search(["none"]),
            })
        elif version == "c_v2_focal":
            space.update({
                "focal_gamma": tune.grid_search([1.0, 2.0, 3.0]), # 只验证难样本挖掘的力度
                "focal_alpha": tune.grid_search([0.5]),      # 不搞类别偏袒
                "align_method": tune.grid_search(["none"]),
            })

        elif version == "c_v3_swd":
            space.update({
                "focal_gamma": tune.grid_search([0.0]),      # 关掉 Focal，控制变量
                "focal_alpha": tune.grid_search([0.5]),
                "align_method": tune.grid_search(["swd"]),
                "align_weight": tune.grid_search([0.01, 0.1, 1.0]), 
            })
        
        elif version == "c_v4_moment":
            space.update({
                "focal_gamma": tune.grid_search([0.0]),      # 关掉 Focal，控制变量
                "focal_alpha": tune.grid_search([0.5]),
                "align_method": tune.grid_search(["moment"]),
                "align_weight": tune.grid_search([0.01, 0.1, 1.0]), 
            })
        # 🟢 新增的经典基线系列
        elif version == "y_baseline_s_learner":
            space.update({
                "model": tune.grid_search(["S_Learner"]),
            })
        elif version == "y_baseline_t_learner":
            space.update({
                "model": tune.grid_search(["T_Learner"]),
            })
        elif version == "y_baseline_dragonnet":
            space.update({
                "model": tune.grid_search(["DragonNet"]),
                "dragon_alpha": tune.grid_search([1.0]),  # Propensity loss 权重
                "dragon_beta": tune.grid_search([1.0]),   # TMLE Reg loss 权重
            })
        elif version == "y_baseline_euen":
            space.update({
                "model": tune.grid_search(["EUEN"]),
                # EUEN 纯靠显式预估，通常不需要加各种复杂的损失权重
            })
        # elif version == "y_baseline_efin":
        #     space.update({
        #         "model": tune.grid_search(["EFIN"]),
        #         "efin_embed_dim": tune.grid_search([16, 32]), 
        #         "efin_lambda": tune.grid_search([1e-3, 1e-2, 1e-1]), 
            # })

        # elif version == "y_baseline_efin":
        #     space.update({
        #         "model": tune.grid_search(["EFIN"]),
        #     })
        # elif version == "c_v3_mmd":
        #     space.update({
        #         "focal_gamma": tune.grid_search([0.0]),      # 拿上一轮表现不错的值固定住
        #         "focal_alpha": tune.grid_search([0.5]),
        #         "mmd_alpha": tune.grid_search([0.01, 0.1, 0.5]), # 只专心搜对齐的力度！
        #         "mmd_sigma": tune.grid_search([1.0])         # 固定带宽
        #     })
            
    # ==========================================
    # 3. Stage 3 (Y 模型) 探索空间 - 你的原版逻辑
    # ==========================================
    elif task == "train_y":
        # 🟢 版本 1：最纯净的 Baseline (无 C 融合，纯 BCE 损失)




        if version == "y_v1_base":
            space.update({
                "c_fusion_mode": tune.grid_search(["none"]),
                "loss_types": tune.grid_search([["bce"]]),
            })
        # ==========================================================
        # 🌟 纯 V10 突围战 & 循序渐进消融对照大阵列 (修正参数范围 + 裸Linear头版)
        # ==========================================================
        
        # --- [实验 1/6: 单独探索 - 纯羊毛党靶向重击] ---
        elif version == "y_pure_v10_solo_wool":
            space.update({
                "model": tune.grid_search(["TARNET_Baseline_PureV10"]), 
                "c_fusion_mode": tune.grid_search(["res_moe"]),
                "loss_types": tune.grid_search([["prior_conflict"]]),
                "head_hidden_dims": tune.grid_search([None]),           # 🌟 默认 None 走 naked linear 头
                "conflict_mode": tune.grid_search(["wool"]), 
                "conflict_alpha_wool": tune.grid_search([0.1, 0.5, 1.0, 5.0, 10.0]), # 🌟 精准修正搜索范围
            })

        # --- [实验 2/6: 单独探索 - 纯隐藏金子绝对拯救] ---
        elif version == "y_pure_v10_solo_gold":
            space.update({
                "model": tune.grid_search(["TARNET_Baseline_PureV10"]), 
                "c_fusion_mode": tune.grid_search(["res_moe"]),
                "loss_types": tune.grid_search([["prior_conflict"]]),
                "head_hidden_dims": tune.grid_search([None]),           # 🌟 默认 None 走 naked linear 头
                "conflict_mode": tune.grid_search(["gold"]), 
                "conflict_alpha_gold": tune.grid_search([0.1, 0.5, 1.0, 5.0, 10.0]), # 🌟 精准修正搜索范围
            })

        # --- [实验 3/6: 单独探索 - 纯自然进店镜像拦截] ---
        elif version == "y_pure_v10_solo_walkin":
            space.update({
                "model": tune.grid_search(["TARNET_Baseline_PureV10"]), 
                "c_fusion_mode": tune.grid_search(["res_moe"]),
                "loss_types": tune.grid_search([["prior_conflict"]]),
                "head_hidden_dims": tune.grid_search([None]),           # 🌟 默认 None 走 naked linear 头
                "conflict_mode": tune.grid_search(["walkin"]), 
                "conflict_alpha_walkin": tune.grid_search([0.1, 0.5, 1.0, 5.0, 10.0]), # 🌟 精准修正搜索范围
            })

# --- [实验 4/7: 混合全激活 - 三人群(All)共享同一爆破参数] ---
        elif version == "y_pure_v10_mix_all_same_alpha":
            space.update({
                "model": tune.grid_search(["TARNET_Baseline_PureV10"]), 
                "c_fusion_mode": tune.grid_search(["res_moe"]),
                "loss_types": tune.grid_search([["prior_conflict"]]),
                "head_hidden_dims": tune.grid_search([None]),           # 🌟 默认 None 走 naked linear 头
                "conflict_mode": tune.grid_search(["all"]), 
                # 🌟 核心联动：只让 wool 走网格搜索，5 档
                "conflict_alpha_wool": tune.grid_search([1.0, 5.0]),  #  0.1, 0.5, 
                # 🌟 强行视线对齐，绝对不产生直积裂变！
                "conflict_alpha_gold": tune.sample_from(lambda spec: spec.config["conflict_alpha_wool"]),   
                "conflict_alpha_walkin": tune.sample_from(lambda spec: spec.config["conflict_alpha_wool"]), 
            })

        # --- [实验 5/7: 经典双向对齐 - 羊毛+金子(Both)共享同一老版机制参数] ---
        elif version == "y_pure_v10_mix_both_same_alpha":
            space.update({
                "model": tune.grid_search(["TARNET_Baseline_PureV10"]), 
                "c_fusion_mode": tune.grid_search(["res_moe"]),
                "loss_types": tune.grid_search([["prior_conflict"]]),
                "head_hidden_dims": tune.grid_search([None]),           # 🌟 默认 None 走 naked linear 头
                "conflict_mode": tune.grid_search(["wool_gold"]), 
                # 🌟 核心联动：wool 主控 5 档
                "conflict_alpha_wool": tune.grid_search([1.0, 5.0, 10.0]),   # 0.1, 0.5, 
                "conflict_alpha_gold": tune.sample_from(lambda spec: spec.config["conflict_alpha_wool"]),   
            })

        # --- [实验 5/7-Head: 经典双向对齐 + 多档预测头探索] ---
        elif version == "y_pure_v10_mix_both_same_alpha_head":
            space.update({
                "model": tune.grid_search(["TARNET_Baseline_PureV10"]), 
                "c_fusion_mode": tune.grid_search(["res_moe"]),
                "loss_types": tune.grid_search([["prior_conflict"]]),
                # 4 档 head 拓扑 × 2 档 alpha = 严格控制在 8 组试验！
                "head_hidden_dims": tune.grid_search([[32], [64, 32], [128, 64, 32], [32, 32]]),  
                "conflict_mode": tune.grid_search(["wool_gold"]), 
                # 🌟 核心联动：同步 2 档 alpha 变化
                "conflict_alpha_wool": tune.grid_search([1.0, 10.0]),   
                "conflict_alpha_gold": tune.sample_from(lambda spec: spec.config["conflict_alpha_wool"]),   
            })

        elif version == "y_pure_v10_all_same_alpha_head_0710":
            space.update({
                "model": tune.grid_search(["TARNET_Baseline_PureV10"]), 
                "c_fusion_mode": tune.grid_search(["res_moe"]),
                "loss_types": tune.grid_search([["prior_conflict"]]),
                # 4 档 head 拓扑 × 2 档 alpha = 严格控制在 8 组试验！
                "head_hidden_dims": tune.grid_search([[32], [64, 32], [32,16], None, [16]]),  
                "weight_decay": tune.grid_search([1e-5, 1e-4,]), # , 1e-3
                "conflict_mode": tune.grid_search(["all"]), 
                # 🌟 核心联动：同步 2 档 alpha 变化
                "conflict_alpha_wool": tune.grid_search([1.0, 5, 0.5]),   
                "conflict_alpha_gold": tune.sample_from(lambda spec: spec.config["conflict_alpha_wool"]),  
                "conflict_alpha_walkin": tune.sample_from(lambda spec: spec.config["conflict_alpha_wool"]),  
            })

        # --- [实验 6/7: 特定双向组合 A - 羊毛+自然进店(Wool+Walkin)共享同一参数] ---
        elif version == "y_pure_v10_mix_wool_walkin_same_alpha":
            space.update({
                "model": tune.grid_search(["TARNET_Baseline_PureV10"]), 
                "c_fusion_mode": tune.grid_search(["res_moe"]),
                "loss_types": tune.grid_search([["prior_conflict"]]),
                "head_hidden_dims": tune.grid_search([None]),           # 🌟 默认 None 走 naked linear 头
                "conflict_mode": tune.grid_search(["wool_walkin"]), 
                # 🌟 核心联动：wool 主控 5 档
                "conflict_alpha_wool": tune.grid_search([ 1.0, 5.0, 10.0]),  #  0.1, 0.5,
                "conflict_alpha_walkin": tune.sample_from(lambda spec: spec.config["conflict_alpha_wool"]), 
            })

        # --- [实验 7/7: 🌟 新增特定双向组合 B - 自然进店+隐藏金子(Walkin+Gold)共享同一参数] ---
        elif version == "y_pure_v10_mix_walkin_gold_same_alpha":
            space.update({
                "model": tune.grid_search(["TARNET_Baseline_PureV10"]), 
                "c_fusion_mode": tune.grid_search(["res_moe"]),
                "loss_types": tune.grid_search([["prior_conflict"]]),
                "head_hidden_dims": tune.grid_search([None]),           # 🌟 默认 None 走 naked linear 头
                "conflict_mode": tune.grid_search(["walkin_gold"]), 
                # 🌟 核心联动：由 walkin 充当主变阻器 5 档
                "conflict_alpha_walkin": tune.grid_search([1.0, 5.0, 10.0]),   # 0.1, 0.5, 
                "conflict_alpha_gold": tune.sample_from(lambda spec: spec.config["conflict_alpha_walkin"]), 
            })
        elif version == "y_pure_v10_debug_arena":
            space.update({
                "model": tune.grid_search(["TARNET_Baseline_PureV10"]), # 物理上在 Residual Base 下突围
                "c_fusion_mode": tune.grid_search(["res_moe"]),
                "loss_types": tune.grid_search([["prior_conflict"]]),
                "head_hidden_dims": tune.grid_search([[32]]),           # 🌟 激活公平加深 Head！
                "conflict_mode": tune.grid_search(["all"]),
                
                # 🧪 分离实验：想调哪个就把哪个参数解冻写进 grid_search 列表即可，其余会稳稳走上面的公共 0.0 关闭默认值！
                "conflict_alpha_wool": tune.grid_search([0.0, 2.0, 5.0]),   
                "conflict_alpha_gold": tune.grid_search([0.0, 5.0]),  
                "conflict_alpha_walkin": tune.grid_search([0.0, 2.0]),
                
                "conflict_focal_type": tune.grid_search(["none", "global_bounded"]), # 测实验 3: 保底 Focal 
                "conflict_use_ohem": tune.grid_search([False, True]),               # 测实验 4: OHEM
                "conflict_use_weight_clip": tune.grid_search([False, True]),         # 测实验 6: Loss侧裁剪
                "conflict_two_stage_mode": tune.grid_search([False, True]),          # 测两阶段训练流
                "conflict_stage1_epochs": tune.grid_search([5]),
                "conflict_freeze_base_in_stage2": tune.grid_search([True])
            })
        elif version == "y_ours_s4_conflict_0710":
            space.update({
                "model": tune.grid_search(["TARNET"]), 
                "c_fusion_mode": tune.grid_search(["ours_s4_conflict"]),
                "loss_types": tune.grid_search([["prior_conflict"]]),
                
                # 预测头容量消融：覆盖 4 档典型非线性漏斗拓扑
                "head_hidden_dims": tune.grid_search([None]), 
                
                # 三向人群错位激活模式：羊毛+金子经典双向对齐
                "conflict_mode": tune.grid_search(["all"]), 
                "weight_decay": tune.grid_search([1e-5, 1e-4, 1e-3]),
                
                "conflict_alpha_wool": tune.grid_search([1.0, 5, 0.5]),   
                "conflict_alpha_gold": tune.sample_from(lambda spec: spec.config["conflict_alpha_wool"]),  
                "conflict_alpha_walkin": tune.sample_from(lambda spec: spec.config["conflict_alpha_wool"]),  
                

            })

        elif version == "y_ours_s4_conflict_0710_alpha0":
            space.update({
                "model": tune.grid_search(["TARNET"]), 
                "c_fusion_mode": tune.grid_search(["ours_s4_conflict"]),
                "loss_types": tune.grid_search([["prior_conflict"]]),
                
                # 预测头容量消融：覆盖 4 档典型非线性漏斗拓扑
                "head_hidden_dims": tune.grid_search([[32], [64, 32], [32,16], None, [16]]), 
                
                # 三向人群错位激活模式：羊毛+金子经典双向对齐
                "conflict_mode": tune.grid_search(["all"]), 
                "weight_decay": tune.grid_search([1e-5, 1e-4, 1e-3, 1e-2]),
                
                "conflict_alpha_wool": tune.grid_search([0]),   
                "conflict_alpha_gold": tune.sample_from(lambda spec: spec.config["conflict_alpha_wool"]),  
                "conflict_alpha_walkin": tune.sample_from(lambda spec: spec.config["conflict_alpha_wool"]),  
            })
        elif version == "y_ours_s4_conflict_0710_search_alpha_res_2_search_head":
            space.update({
                "model": tune.grid_search(["TARNET"]), 
                "c_fusion_mode": tune.grid_search(["ours_s4_conflict"]),
                "loss_types": tune.grid_search([["prior_conflict"]]),
                
                # 预测头容量消融：覆盖 4 档典型非线性漏斗拓扑
                "head_hidden_dims": tune.grid_search([ [64, 32] ]), #  , [64, 32[32,16], [16], [32]]
                
                # 三向人群错位激活模式：羊毛+金子经典双向对齐
                "conflict_mode": tune.grid_search(["all"]), 
                "weight_decay": tune.grid_search([1e-5, 1e-4, 1e-3, 1e-2]),
                
                "conflict_alpha_wool": tune.grid_search([1, 0.5, 5, 10 , 0.1]),   
                "conflict_alpha_gold": tune.sample_from(lambda spec: spec.config["conflict_alpha_wool"]),  
                "conflict_alpha_walkin": tune.sample_from(lambda spec: spec.config["conflict_alpha_wool"]),  
            })
        elif version == "y_ours_s6_conflict_0710_alpha_search_temp10":
            space.update({
                "model": tune.grid_search(["TARNET"]), 
                "c_fusion_mode": tune.grid_search(["ours_s6_conflict"]),
                "loss_types": tune.grid_search([["prior_conflict"]]),
                
                # 预测头容量消融：覆盖 4 档典型非线性漏斗拓扑
                "head_hidden_dims": tune.grid_search([None]), 
                
                # 三向人群错位激活模式：羊毛+金子经典双向对齐
                "conflict_mode": tune.grid_search(["all"]), 
                "weight_decay": tune.grid_search([1e-5, 1e-4, 1e-3, 1e-2]),
                "ours_s6_temp": tune.grid_search([10]), 
                
                "conflict_alpha_wool": tune.grid_search([1, 0.5, 5, 10 , 0.1]),   
                "conflict_alpha_gold": tune.sample_from(lambda spec: spec.config["conflict_alpha_wool"]),  
                "conflict_alpha_walkin": tune.sample_from(lambda spec: spec.config["conflict_alpha_wool"]),  
            })
        elif version == "y_ours_s6_conflict_0710_alpha_search_temp1_20":
            space.update({
                "model": tune.grid_search(["TARNET"]), 
                "c_fusion_mode": tune.grid_search(["ours_s6_conflict"]),
                "loss_types": tune.grid_search([["prior_conflict"]]),
                
                # 预测头容量消融：覆盖 4 档典型非线性漏斗拓扑
                "head_hidden_dims": tune.grid_search([[32],[32,16],[64,32]]), 
                
                # 三向人群错位激活模式：羊毛+金子经典双向对齐
                "conflict_mode": tune.grid_search(["all"]), 
                "weight_decay": tune.grid_search([1e-5, 1e-4, 1e-3, 1e-2]),
                "ours_s6_temp": tune.grid_search([1,20]), 
                
                "conflict_alpha_wool": tune.grid_search([1, 0.5, 5, 10 , 0.1]),   
                "conflict_alpha_gold": tune.sample_from(lambda spec: spec.config["conflict_alpha_wool"]),  
                "conflict_alpha_walkin": tune.sample_from(lambda spec: spec.config["conflict_alpha_wool"]),  
            })
        elif version == "y_ours_s6_conflict_0710_alpha0_temp10":
            space.update({
                "model": tune.grid_search(["TARNET"]), 
                "c_fusion_mode": tune.grid_search(["ours_s6_conflict"]),
                "loss_types": tune.grid_search([["prior_conflict"]]),
                
                # 预测头容量消融：覆盖 4 档典型非线性漏斗拓扑
                "head_hidden_dims": tune.grid_search([[32], [64, 32], [32,16], None, [16]]), 
                
                # 三向人群错位激活模式：羊毛+金子经典双向对齐
                "conflict_mode": tune.grid_search(["all"]), 
                "weight_decay": tune.grid_search([1e-5, 1e-4, 1e-3, 1e-2]),
                "ours_s6_temp": tune.grid_search([10]), 
                
                "conflict_alpha_wool": tune.grid_search([0]),   
                "conflict_alpha_gold": tune.sample_from(lambda spec: spec.config["conflict_alpha_wool"]),  
                "conflict_alpha_walkin": tune.sample_from(lambda spec: spec.config["conflict_alpha_wool"]),  
            })
        elif version == "y_ours_s6_conflict_0710_alpha0_temp20":
            space.update({
                "model": tune.grid_search(["TARNET"]), 
                "c_fusion_mode": tune.grid_search(["ours_s6_conflict"]),
                "loss_types": tune.grid_search([["prior_conflict"]]),
                
                # 预测头容量消融：覆盖 4 档典型非线性漏斗拓扑
                "head_hidden_dims": tune.grid_search([[32], [64, 32], [32,16], None, [16]]), 
                
                # 三向人群错位激活模式：羊毛+金子经典双向对齐
                "conflict_mode": tune.grid_search(["all"]), 
                "weight_decay": tune.grid_search([1e-5, 1e-4, 1e-3, 1e-2]),
                "ours_s6_temp": tune.grid_search([20]), 
                
                "conflict_alpha_wool": tune.grid_search([0]),   
                "conflict_alpha_gold": tune.sample_from(lambda spec: spec.config["conflict_alpha_wool"]),  
                "conflict_alpha_walkin": tune.sample_from(lambda spec: spec.config["conflict_alpha_wool"]),  
            })
        elif version == "y_ours_s6_conflict_0710_alpha0_temp1":
            space.update({
                "model": tune.grid_search(["TARNET"]), 
                "c_fusion_mode": tune.grid_search(["ours_s6_conflict"]),
                "loss_types": tune.grid_search([["prior_conflict"]]),
                
                # 预测头容量消融：覆盖 4 档典型非线性漏斗拓扑
                "head_hidden_dims": tune.grid_search([[32], [64, 32], [32,16], None, [16]]), 
                
                # 三向人群错位激活模式：羊毛+金子经典双向对齐
                "conflict_mode": tune.grid_search(["all"]), 
                "weight_decay": tune.grid_search([1e-5, 1e-4, 1e-3, 1e-2]),
                "ours_s6_temp": tune.grid_search([1]), 
                
                "conflict_alpha_wool": tune.grid_search([0]),   
                "conflict_alpha_gold": tune.sample_from(lambda spec: spec.config["conflict_alpha_wool"]),  
                "conflict_alpha_walkin": tune.sample_from(lambda spec: spec.config["conflict_alpha_wool"]),  
            })    
        elif version == "y_v8_s6_temp_10_20_v10_all_mix_original_code_search_head_wd_alpha_same":
            space.update({"c_fusion_mode": tune.grid_search(["v8_evolution_moe"]), 
            "head_hidden_dims": tune.grid_search([[32], None]), 
            "v8_scheme": tune.grid_search([6]),
                           "align_temp": tune.grid_search([10,20]),
                           "weight_decay": tune.grid_search([1e-4]), # , 1e-4
                           "conflict_mode": tune.grid_search(["all"]), 
                "conflict_alpha_wool": tune.grid_search([1, 0.5, 5, 10 , 0.1, 0]),   
                "conflict_alpha_gold": tune.sample_from(lambda spec: spec.config["conflict_alpha_wool"]),  
                "conflict_alpha_walkin": tune.sample_from(lambda spec: spec.config["conflict_alpha_wool"]),  
                         "loss_types": tune.grid_search([["prior_conflict"]])})
        elif version == "y_v8_s4_v10_all_mix_original_code_search_head_wd_alpha_same":
            space.update({"c_fusion_mode": tune.grid_search(["v8_evolution_moe"]), 
            "head_hidden_dims": tune.grid_search([[32], None]), 
            "v8_scheme": tune.grid_search([6]),
                           "weight_decay": tune.grid_search([1e-4]), # , 1e-4
                           "conflict_mode": tune.grid_search(["all"]), 
                "conflict_alpha_wool": tune.grid_search([1, 0.5, 5, 10 , 0.1, 0]),   
                "conflict_alpha_gold": tune.sample_from(lambda spec: spec.config["conflict_alpha_wool"]),  
                "conflict_alpha_walkin": tune.sample_from(lambda spec: spec.config["conflict_alpha_wool"]),  
                         "loss_types": tune.grid_search([["prior_conflict"]])})

        elif version == "y_ours_s4_conflict_0710_1e-2":
            space.update({
                "model": tune.grid_search(["TARNET"]), 
                "c_fusion_mode": tune.grid_search(["ours_s4_conflict"]),
                "loss_types": tune.grid_search([["prior_conflict"]]),
                
                # 预测头容量消融：覆盖 4 档典型非线性漏斗拓扑
                "head_hidden_dims": tune.grid_search([None]), 
                
                # 三向人群错位激活模式：羊毛+金子经典双向对齐
                "conflict_mode": tune.grid_search(["all"]), 
                "weight_decay": tune.grid_search([1e-2]),
                
                "conflict_alpha_wool": tune.grid_search([1.0, 5, 0.5]),   
                "conflict_alpha_gold": tune.sample_from(lambda spec: spec.config["conflict_alpha_wool"]),  
                "conflict_alpha_walkin": tune.sample_from(lambda spec: spec.config["conflict_alpha_wool"]),  

            })

        # ==========================================================
        # 🌟 Ours 核心架构与历史演进兼容 (保持原本老逻辑，额外多塞入消融调试接口)
        # ==========================================================
        elif version == "y_ours_s4_conflict":
            space.update({
                "c_fusion_mode": tune.grid_search(["ours_s4_conflict"]),
                "loss_types": tune.grid_search([["prior_conflict"]]),
                "conflict_alpha": tune.grid_search([0, 0.01, 0.1, 0.5, 1.0, 5.0, 10]),
                "conflict_mode": tune.grid_search(["both"]),
                # 👇 下面这些留给你要做 debug 或分离时一键启动，目前都是 tune.grid_search([默认]) 绝不破坏老版本执行结果
                "head_hidden_dims": tune.grid_search([[32]]) # Ours 架构默认同步开起 Head 公平对齐
            })
        elif version == "y_ours_s4_conflict_0717_h_None_search_alpha":
            space.update({
                "c_fusion_mode": tune.grid_search(["ours_s4_conflict"]),
                "loss_types": tune.grid_search([["prior_conflict"]]),
                "conflict_alpha_wool": tune.grid_search([ 1, 5, 10, 0.1, 0.5]),  # ([1, 0.5, 5, 10 , 0.1]), 0.5, 5,   0.05, 0, 0.01,
                "hidden_dims": tune.grid_search([[128, 64, 32]]), 
                "conflict_alpha_gold": tune.sample_from(lambda spec: spec.config["conflict_alpha_wool"]),  
                "conflict_alpha_walkin": tune.sample_from(lambda spec: spec.config["conflict_alpha_wool"]),  
                "weight_decay": tune.grid_search([1e-3, 1e-2]), # 1e-3, 1e-2
                "conflict_mode": tune.grid_search(["all"]),
                # 👇 下面这些留给你要做 debug 或分离时一键启动，目前都是 tune.grid_search([默认]) 绝不破坏老版本执行结果
                "head_hidden_dims": tune.grid_search([None]) # Ours 架构默认同步开起 Head 公平对齐
            })
        elif version == "y_ours_s6_conflict_0717_hNone_search_alpha":
            space.update({
                "model": tune.grid_search(["TARNET"]), 
                "c_fusion_mode": tune.grid_search(["ours_s6_conflict"]),
                "loss_types": tune.grid_search([["prior_conflict"]]),
                
                # 预测头容量消融：覆盖 4 档典型非线性漏斗拓扑
                "head_hidden_dims": tune.grid_search([None]), 
                
                # 三向人群错位激活模式：羊毛+金子经典双向对齐
                "conflict_mode": tune.grid_search(["all"]), 
                "weight_decay": tune.grid_search([1e-3,1e-2,1e-4,1e-5]),
                "hidden_dims": tune.grid_search([[64,32],[64],[32],[16],[32,16]]), 
                "ours_s6_temp": tune.grid_search([1]), 
                
                "conflict_alpha_wool": tune.grid_search([1, 5, 10, 0.1, 0.5]),  # ([0.05, 0, 0.01, 1, 0.5, 5, 10 , 0.1]),  # ([1, 0.5, 5, 10 , 0.1]),   
                "conflict_alpha_gold": tune.sample_from(lambda spec: spec.config["conflict_alpha_wool"]),  
                "conflict_alpha_walkin": tune.sample_from(lambda spec: spec.config["conflict_alpha_wool"]),  
            })
        elif version == "y_ours_s6_conflict_0717_hNone_search_alpha_temp":
            space.update({
                "model": tune.grid_search(["TARNET"]), 
                "c_fusion_mode": tune.grid_search(["ours_s6_conflict"]),
                "loss_types": tune.grid_search([["prior_conflict"]]),
                
                # 预测头容量消融：覆盖 4 档典型非线性漏斗拓扑
                "head_hidden_dims": tune.grid_search([None]), 
                
                # 三向人群错位激活模式：羊毛+金子经典双向对齐
                "conflict_mode": tune.grid_search(["all"]), 
                "weight_decay": tune.grid_search([1e-3,1e-2,1e-4,1e-5]),
                "hidden_dims": tune.grid_search([[128, 64, 32]]), 
                "ours_s6_temp": tune.grid_search([0.1,5,10,20]), 
                
                "conflict_alpha_wool": tune.grid_search([1, 10, 0.1]),  # ([0.05, 0, 0.01, 1, 0.5, 5, 10 , 0.1]),  # ([1, 0.5, 5, 10 , 0.1]),   
                "conflict_alpha_gold": tune.sample_from(lambda spec: spec.config["conflict_alpha_wool"]),  
                "conflict_alpha_walkin": tune.sample_from(lambda spec: spec.config["conflict_alpha_wool"]),  
            })
        # ----------------------------------------------------------
        # Criteo · V8_s6 + H20 OFAT（傻瓜单轴）· 基于最新 prior_conflict 键名
        # 对照工业：rnc=关renorm+clip · k50=ohem · pw=羊毛α · bl=下保底
        # ----------------------------------------------------------

        elif version == "y_v8_s6_cf_b0":
            # Loss 对照尺：conflict 开 · renorm=mean · 无 clip/ohem/focal
            space.update({
                "c_fusion_mode": tune.grid_search(["v8_evolution_moe"]),
                "v8_scheme": tune.grid_search([6]),
                "loss_types": tune.grid_search([["prior_conflict"]]),
                "weight_decay": tune.grid_search([1e-3,1e-2]),
                "hidden_dims": tune.grid_search([[64,32]]), 
                "conflict_mode": tune.grid_search(["all"]),
                "conflict_alpha_wool": tune.grid_search([1, 5, 10]),  # ([0.05, 0, 0.01, 1, 0.5, 5, 10 , 0.1]),  # ([1, 0.5, 5, 10 , 0.1]),   
                "conflict_alpha_gold": tune.sample_from(lambda spec: spec.config["conflict_alpha_wool"]),  
                "conflict_alpha_walkin": tune.sample_from(lambda spec: spec.config["conflict_alpha_wool"]),  
                "conflict_renorm": tune.grid_search(["mean"]),
                "conflict_focal_type": tune.grid_search(["none"]),
                "conflict_use_ohem": tune.grid_search([False]),
                "conflict_use_weight_clip": tune.grid_search([False]),
            })

        elif version == "y_v8_s6_rnc":
            # 主轴：关均值抹平 + 硬帽（工业 rnc3/5/8）
            space.update({
                "c_fusion_mode": tune.grid_search(["v8_evolution_moe"]),
                "v8_scheme": tune.grid_search([6]),
                "loss_types": tune.grid_search([["prior_conflict"]]),
                "weight_decay": tune.grid_search([1e-3,1e-2]),
                "hidden_dims": tune.grid_search([[64,32]]), 
                "conflict_mode": tune.grid_search(["all"]),
                "conflict_alpha_wool": tune.grid_search([1, 5, 10]),  # ([0.05, 0, 0.01, 1, 0.5, 5, 10 , 0.1]),  # ([1, 0.5, 5, 10 , 0.1]),   
                "conflict_alpha_gold": tune.sample_from(lambda spec: spec.config["conflict_alpha_wool"]),  
                "conflict_alpha_walkin": tune.sample_from(lambda spec: spec.config["conflict_alpha_wool"]),  
                "conflict_renorm": tune.grid_search(["none"]),
                "conflict_focal_type": tune.grid_search(["none"]),
                "conflict_use_ohem": tune.grid_search([False]),
                "conflict_use_weight_clip": tune.grid_search([True]),
                "conflict_max_weight_thres": tune.grid_search([3.0, 5.0, 8.0]),
            })
        elif version == "y_v8_s6_c":
            # 主轴：关均值抹平 + 硬帽（工业 rnc3/5/8）
            space.update({
                "c_fusion_mode": tune.grid_search(["v8_evolution_moe"]),
                "v8_scheme": tune.grid_search([6]),
                "loss_types": tune.grid_search([["prior_conflict"]]),
                "weight_decay": tune.grid_search([1e-3,1e-2]),
                "hidden_dims": tune.grid_search([[64,32]]), 
                "conflict_mode": tune.grid_search(["all"]),
                "conflict_alpha_wool": tune.grid_search([1, 5, 10]),  # ([0.05, 0, 0.01, 1, 0.5, 5, 10 , 0.1]),  # ([1, 0.5, 5, 10 , 0.1]),   
                "conflict_alpha_gold": tune.sample_from(lambda spec: spec.config["conflict_alpha_wool"]),  
                "conflict_alpha_walkin": tune.sample_from(lambda spec: spec.config["conflict_alpha_wool"]),  
                "conflict_renorm": tune.grid_search(["mean"]),
                "conflict_focal_type": tune.grid_search(["none"]),
                "conflict_use_ohem": tune.grid_search([False]),
                "conflict_use_weight_clip": tune.grid_search([True]),
                "conflict_max_weight_thres": tune.grid_search([3.0, 5.0, 8.0]),
            })

        elif version == "y_v8_s6_ohem":
            # 次轴：renorm 仍开 · 只保留最难 ohem_pct（工业 k50 ≈ 0.5）
            space.update({
                "c_fusion_mode": tune.grid_search(["v8_evolution_moe"]),
                "v8_scheme": tune.grid_search([6]),
                "loss_types": tune.grid_search([["prior_conflict"]]),
                "weight_decay": tune.grid_search([1e-3,1e-2]),
                "hidden_dims": tune.grid_search([[64,32]]), 
                "conflict_mode": tune.grid_search(["all"]),
                "conflict_alpha_wool": tune.grid_search([1, 5, 10]),  # ([0.05, 0, 0.01, 1, 0.5, 5, 10 , 0.1]),  # ([1, 0.5, 5, 10 , 0.1]),   
                "conflict_alpha_gold": tune.sample_from(lambda spec: spec.config["conflict_alpha_wool"]),  
                "conflict_alpha_walkin": tune.sample_from(lambda spec: spec.config["conflict_alpha_wool"]),  
                "conflict_renorm": tune.grid_search(["mean"]),
                "conflict_focal_type": tune.grid_search(["none"]),
                "conflict_use_ohem": tune.grid_search([True]),
                "conflict_ohem_pct": tune.grid_search([0.2,0.3, 0.4, 0.5, 0.6, 0.7,0.8,0.1,0.9]),
                "conflict_use_weight_clip": tune.grid_search([False]),
            })


        elif version == "y_v8_s6_bl":
            # 弱轴：下保底（工业 bl*）
            space.update({
                "c_fusion_mode": tune.grid_search(["v8_evolution_moe"]),
                "v8_scheme": tune.grid_search([6]),
                "loss_types": tune.grid_search([["prior_conflict"]]),
                "weight_decay": tune.grid_search([1e-3,1e-2]),
                "hidden_dims": tune.grid_search([[64,32]]), 
                "conflict_mode": tune.grid_search(["all"]),
                "conflict_alpha_wool": tune.grid_search([1, 5, 10]),  # ([0.05, 0, 0.01, 1, 0.5, 5, 10 , 0.1]),  # ([1, 0.5, 5, 10 , 0.1]),   
                "conflict_alpha_gold": tune.sample_from(lambda spec: spec.config["conflict_alpha_wool"]),  
                "conflict_alpha_walkin": tune.sample_from(lambda spec: spec.config["conflict_alpha_wool"]),  
                "conflict_renorm": tune.grid_search(["mean"]),
                "conflict_focal_type": tune.grid_search(["none"]),
                "conflict_use_ohem": tune.grid_search([False]),
                "conflict_use_weight_clip": tune.grid_search([False]),
                "conflict_min_weight_thres": tune.grid_search([0.1, 0.25, 0.5]),
            })

        elif version == "y_pure_v10_0717_h_None_search_alpha":
            space.update({
                "model": tune.grid_search(["TARNET_Baseline_PureV10"]), 
                "c_fusion_mode": tune.grid_search(["res_moe"]),
                "loss_types": tune.grid_search([["prior_conflict"]]),
                # 4 档 head 拓扑 × 2 档 alpha = 严格控制在 8 组试验！
                "head_hidden_dims": tune.grid_search([None]),  
                "weight_decay": tune.grid_search([1e-5, 1e-4, 1e-3, 1e-2]), # , 1e-3
                "hidden_dims": tune.grid_search([[128, 64, 32]]), 
                "conflict_mode": tune.grid_search(["all"]), 
                # 🌟 核心联动：同步 2 档 alpha 变化
                "conflict_alpha_wool": tune.grid_search([1, 5, 10, 0.1, 0.5]),  # ([1, 0.5, 5, 10 , 0.1]),   
                "conflict_alpha_gold": tune.sample_from(lambda spec: spec.config["conflict_alpha_wool"]),  
                "conflict_alpha_walkin": tune.sample_from(lambda spec: spec.config["conflict_alpha_wool"]),  
            })
        elif version == "y_pure_v10_0713_h32_search_search_alpha_independt":
            space.update({
                "model": tune.grid_search(["TARNET_Baseline_PureV10"]), 
                "c_fusion_mode": tune.grid_search(["res_moe"]),
                "loss_types": tune.grid_search([["prior_conflict"]]),
                # 4 档 head 拓扑 × 2 档 alpha = 严格控制在 8 组试验！
                "head_hidden_dims": tune.grid_search([None]),  
                "weight_decay": tune.grid_search([1e-5, 1e-4, 1e-3, 1e-2]), # , 1e-3
                "hidden_dims": tune.grid_search([[128, 64, 32]]), 
                "conflict_mode": tune.grid_search(["all"]), 
                # 🌟 核心联动：同步 2 档 alpha 变化
                "conflict_alpha_wool": tune.grid_search([1, 0.5, 5  , 0.1]),  # ([1, 0.5, 5, 10 , 0.1]),   
                "conflict_alpha_gold": tune.grid_search([1, 0.5, 5 , 0.1]),
                "conflict_alpha_walkin": tune.grid_search([1, 0.5, 5 , 0.1]),
            })
            
        # 🟢 在所有 S6 的独立温度分支中，同步塞入加深 Head、Logit裁剪等全新可调空间入口
        elif version == "y_ours_s6_conflict_temp1.0":
            space.update({
                "c_fusion_mode": tune.grid_search(["ours_s6_conflict"]), 
                "conflict_mode": tune.grid_search(["both"]), 
                "loss_types": tune.grid_search([["prior_conflict"]]), 
                "ours_s6_temp": tune.grid_search([1.0]), 
                "conflict_alpha": tune.grid_search([1.0, 3.0, 5.0, 10]),
                "head_hidden_dims": tune.grid_search([[32]]), # 🌟 开启高容量 Head
                "ours_s6_use_logit_clamp": tune.grid_search([False, True]) # 🌟 留好一键 Logit 空间裁剪防数值黑洞接口！
            })
            
        elif version == "y_ours_s6_conflict_temp5.0":
            space.update({
                "c_fusion_mode": tune.grid_search(["ours_s6_conflict"]), 
                "conflict_mode": tune.grid_search(["both"]), 
                "loss_types": tune.grid_search([["prior_conflict"]]), 
                "ours_s6_temp": tune.grid_search([5.0]), 
                "conflict_alpha": tune.grid_search([1.0, 3.0, 5.0, 10]),
                "head_hidden_dims": tune.grid_search([[32]]),
                "ours_s6_use_logit_clamp": tune.grid_search([False, True])
            })
            
        elif version == "y_ours_s6_conflict_temp10.0":
            space.update({
                "c_fusion_mode": tune.grid_search(["ours_s6_conflict"]), 
                "conflict_mode": tune.grid_search(["both"]), 
                "loss_types": tune.grid_search([["prior_conflict"]]), 
                "ours_s6_temp": tune.grid_search([10.0]), 
                "conflict_alpha": tune.grid_search([1.0, 3.0, 5.0, 10]),
                "head_hidden_dims": tune.grid_search([[32]]),
                "ours_s6_use_logit_clamp": tune.grid_search([False, True])
            })
            
        elif version == "y_ours_s6_conflict_temp20.0":
            space.update({
                "c_fusion_mode": tune.grid_search(["ours_s6_conflict"]), 
                "conflict_mode": tune.grid_search(["both"]), 
                "loss_types": tune.grid_search([["prior_conflict"]]), 
                "ours_s6_temp": tune.grid_search([20.0]), 
                "conflict_alpha": tune.grid_search([1.0, 3.0, 5.0, 10]),
                "head_hidden_dims": tune.grid_search([[32]]),
                "ours_s6_use_logit_clamp": tune.grid_search([False, True])
            })
            
        elif version == "y_ours_s6_conflict_temp5.0": # 对应原代码里的重复定义项
            space.update({
                "c_fusion_mode": tune.grid_search(["ours_s6_conflict"]), 
                "conflict_mode": tune.grid_search(["both"]), 
                "loss_types": tune.grid_search([["prior_conflict"]]), 
                "ours_s6_temp": tune.grid_search([5.0]), 
                "conflict_alpha": tune.grid_search([1.0, 3.0, 5.0]),
                "head_hidden_dims": tune.grid_search([[32]]),
                "ours_s6_use_logit_clamp": tune.grid_search([False, True])
            })


        elif version == "y_baseline_s_learner":
            space.update({
                "model": tune.grid_search(["S_Learner"]),
            })
        elif version == "y_baseline_t_learner":
            space.update({
                "model": tune.grid_search(["T_Learner"]),
            })
        elif version == "y_baseline_dragonnet":
            space.update({
                "model": tune.grid_search(["DragonNet"]),
                "dragon_alpha": tune.grid_search([1.0]),  # Propensity loss 权重
                "dragon_beta": tune.grid_search([1.0]),   # TMLE Reg loss 权重
            })
        elif version == "y_baseline_euen":
            space.update({
                "model": tune.grid_search(["EUEN"]),
                # EUEN 纯靠显式预估，通常不需要加各种复杂的损失权重
            })
        elif version == "y_baseline_efin_32":
            space.update({
                "model": tune.grid_search(["EFIN"]),
                "efin_embed_dim": tune.grid_search([32]), # tune.grid_search([32, 64, 128]), # 
                "weight_decay": tune.grid_search([1e-6, 1e-5, 1e-4, 1e-3, 1e-2]),
                # "efin_lambda": tune.grid_search([1e-3, 1e-2, 1e-1]), # Eq.2 的 tradeoff 参数 \lambda
            })
        elif version == "y_baseline_efin_64":
            space.update({
                "model": tune.grid_search(["EFIN"]),
                "efin_embed_dim": tune.grid_search([64]), # tune.grid_search([32, 64, 128]), # 
                "weight_decay": tune.grid_search([1e-6, 1e-5, 1e-4, 1e-3, 1e-2]),
                # "efin_lambda": tune.grid_search([1e-3, 1e-2, 1e-1]), # Eq.2 的 tradeoff 参数 \lambda
            })
        elif version == "y_baseline_efin_128":
            space.update({
                "model": tune.grid_search(["EFIN"]),
                "efin_embed_dim": tune.grid_search([128]), # tune.grid_search([32, 64, 128]), #
                "weight_decay": tune.grid_search([1e-6, 1e-5, 1e-4, 1e-3, 1e-2]), 
                # "efin_lambda": tune.grid_search([1e-3, 1e-2, 1e-1]), # Eq.2 的 tradeoff 参数 \lambda
            })
        # 🟢 经典基线: CFRNet

        elif version == "y_baseline_cfrnet":
            space.update({
                "model": tune.grid_search(["CFRNet"]),
                # 搜一下 SWD 对齐的力度，这个参数对 CFRNet 的效果影响极其致命
                "cfr_weight": tune.grid_search([ 0.01, 0.1, 1.0]), 
            })

        elif version == "y_baseline_cfrnet_2":
            space.update({
                "model": tune.grid_search(["CFRNet"]),
                # 搜一下 SWD 对齐的力度，这个参数对 CFRNet 的效果影响极其致命
                "cfr_weight": tune.grid_search([ 0.001, 0.01, 0.1, 1.0]), 
                "hidden_dims":tune.grid_search([[64], [64, 32], [128, 64, 32]]),
            })
            # "hidden_dims": tune.grid_search([[128, 64, 32]]), 
        elif version == "y_baseline_euen_academic_ms":
            space.update({
                "model": tune.grid_search(["EUEN_Academic"]),
                # "hidden_dims": tune.grid_search([[128, 64]]), # 显式锚定 TF 四档最稳核心配置
                "loss_types": tune.grid_search([["bce"]]),    # 纯净二值 BCE，不回灌辅助 loss
            })
        
        # 🟢 全空间联合建模: DESCN (KDD'22)
        elif version == "y_baseline_descn":
            space.update({
                "model": tune.grid_search(["DESCN"]),
                # "hidden_dims": tune.grid_search([[128]]), # 遵从论文 4.4 节设定
                # DESCN 的多任务权重，先保持 1.0 Baseline，gamma 可以轻微扰动
                "descn_alpha": tune.grid_search([0.5]), 
                "descn_beta1": tune.grid_search([1.0]),
                "descn_beta0": tune.grid_search([1.0]),
                
                # 🌟 X-Network 交叉权重 (对应 config: h1_w, h0_w)
                # 直接复用官方泄露的不对称权重：0.5 和 0.1
                "descn_gamma1": tune.grid_search([0.5, 1.0]), 
                "descn_gamma0": tune.grid_search([0.1, 0.5]),
            })

        elif version == "y_v2_emb":
            space.update({
                "c_fusion_mode": tune.grid_search(["joint_emb"]),
                "c_embedding_dim": tune.grid_search([1,4,16]), # EMB 维度
                "loss_types": tune.grid_search([["bce"]]), # Loss 退回 Base
            })
            
        # 🟡 版本 3：Base + MoE 融合 (只用 C 做专家门控，不改 Loss)
        elif version == "y_v3_moe":
            space.update({
                "c_fusion_mode": tune.grid_search(["moe"]),
                "loss_types": tune.grid_search([["bce"]]), # Loss 退回 Base
            })
        
        
            
        # 🔵 版本 4：Base + 损失分层 (不加 C 融合，纯靠 Strata Loss)
        elif version == "y_v4_loss_strata":
            space.update({
                "c_fusion_mode": tune.grid_search(["none"]), # 融合退回 Base
                "loss_types": tune.grid_search([["strata_weighted"]]), 
            })
            
        # 🔵 版本 5：Base + 损失分层 + 方差正则化 (不加 C 融合，专攻复杂 Loss)
        elif version == "y_v5_loss_var":
            space.update({
                "c_fusion_mode": tune.grid_search(["none"]), # 融合退回 Base
                "loss_types": tune.grid_search([["variance_reg"]]),
                "var_lambda": tune.grid_search([0.1, 1.0, 10]), # 穷举一下方差正则化的惩罚力度
            })
        elif version == "y_v6_res_moe":
            space.update({
                "c_fusion_mode": tune.grid_search(["res_moe"]),
                "loss_types": tune.grid_search([["bce"]]), # 先用纯 BCE 验证架构优越性
            })
        elif version == "y_v7_hard_top5":
            space.update({
                "c_fusion_mode": tune.grid_search(["v7_truncated_moe"]),
                "loss_types": tune.grid_search([["bce"]]), 
                "truncation_mode": tune.grid_search(["hard"]),
                "truncation_pct": tune.grid_search([0.05]), 
            })
        elif version == "y_v7_hard_top1":
            space.update({
                "c_fusion_mode": tune.grid_search(["v7_truncated_moe"]),
                "loss_types": tune.grid_search([["bce"]]), 
                "truncation_mode": tune.grid_search(["hard"]),
                "truncation_pct": tune.grid_search([0.01]), 
            })   
        # 2. 硬截断 Top 10% (探底测试：放宽 5% 看看会不会开始引入羊毛党毒药)
        elif version == "y_v7_hard_top10":
            space.update({
                "c_fusion_mode": tune.grid_search(["v7_truncated_moe"]),
                "loss_types": tune.grid_search([["bce"]]), 
                "truncation_mode": tune.grid_search(["hard"]),
                "truncation_pct": tune.grid_search([0.10]),
            })

        # 3. 软截断 Top 5% (平滑版：防止硬截断在边界处产生梯度突变)
        elif version == "y_v7_soft_top5":
            space.update({
                "c_fusion_mode": tune.grid_search(["v7_truncated_moe"]),
                "loss_types": tune.grid_search([["bce"]]), 
                "truncation_mode": tune.grid_search(["soft"]),
                "truncation_pct": tune.grid_search([0.05]), 
                "truncation_temp": tune.grid_search([10.0]), # 软截断专属温度
            })
        elif version == "y_v7_soft_top1":
            space.update({
                "c_fusion_mode": tune.grid_search(["v7_truncated_moe"]),
                "loss_types": tune.grid_search([["bce"]]), 
                "truncation_mode": tune.grid_search(["soft"]),
                "truncation_pct": tune.grid_search([0.01]), 
                "truncation_temp": tune.grid_search([10.0]), # 软截断专属温度
            })
        # 4. 软截断 Top 10%
        elif version == "y_v7_soft_top10":
            space.update({
                "c_fusion_mode": tune.grid_search(["v7_truncated_moe"]),
                "loss_types": tune.grid_search([["bce"]]), 
                "truncation_mode": tune.grid_search(["soft"]),
                "truncation_pct": tune.grid_search([0.10]),
                "truncation_temp": tune.grid_search([10.0]),
            })
        # 5. 对照组：硬截断 Top 30% (模拟退化回 V3 的状态，用于论文反证法)
        elif version == "y_v7_hard_top30":
            space.update({
                "c_fusion_mode": tune.grid_search(["v7_truncated_moe"]),
                "loss_types": tune.grid_search([["bce"]]), 
                "truncation_mode": tune.grid_search(["hard"]),
                "truncation_pct": tune.grid_search([0.30]),
            })


        # ==========================================
        # 🟣 V8 演进方案 2: 单门控特征驱动
        # ==========================================
        elif version == "y_v8_s1_t5":
            space.update({
                "c_fusion_mode": tune.grid_search(["v8_evolution_moe"]),
                "v8_scheme": tune.grid_search([1]),
                "truncation_pct": tune.grid_search([0.05]), 
                "loss_types": tune.grid_search([["bce"]]),
            })
        elif version == "y_v8_s1_t10":
            space.update({
                "c_fusion_mode": tune.grid_search(["v8_evolution_moe"]),
                "v8_scheme": tune.grid_search([1]),
                "truncation_pct": tune.grid_search([0.10]), 
                "loss_types": tune.grid_search([["bce"]]),
            })
        elif version == "y_v8_s1_t50":
            space.update({
                "c_fusion_mode": tune.grid_search(["v8_evolution_moe"]),
                "v8_scheme": tune.grid_search([1]),
                "truncation_pct": tune.grid_search([0.50]), 
                "loss_types": tune.grid_search([["bce"]]),
            })

        # ==========================================
        # 🟣 V8 演进方案 2: 单门控特征驱动
        # ==========================================
        elif version == "y_v8_s2_t5":
            space.update({
                "c_fusion_mode": tune.grid_search(["v8_evolution_moe"]),
                "v8_scheme": tune.grid_search([2]),
                "truncation_pct": tune.grid_search([0.05]),
                "loss_types": tune.grid_search([["bce"]]),
            })
        elif version == "y_v8_s2_t10":
            space.update({
                "c_fusion_mode": tune.grid_search(["v8_evolution_moe"]),
                "v8_scheme": tune.grid_search([2]),
                "truncation_pct": tune.grid_search([0.10]),
                "loss_types": tune.grid_search([["bce"]]),
            })
        elif version == "y_v8_s2_t50":
            space.update({
                "c_fusion_mode": tune.grid_search(["v8_evolution_moe"]),
                "v8_scheme": tune.grid_search([2]),
                "truncation_pct": tune.grid_search([0.50]),
                "loss_types": tune.grid_search([["bce"]]),
            })

        # ==========================================
        # 🟣 V8 演进方案 3: 独立多门控特征驱动 (🌟 最推荐 Baseline)
        # ==========================================
        elif version == "y_v8_s3_t5":
            space.update({
                "c_fusion_mode": tune.grid_search(["v8_evolution_moe"]),
                "v8_scheme": tune.grid_search([3]),
                "truncation_pct": tune.grid_search([0.05]),
                "loss_types": tune.grid_search([["bce"]]),
            })
        elif version == "y_v8_s3_t10":
            space.update({
                "c_fusion_mode": tune.grid_search(["v8_evolution_moe"]),
                "v8_scheme": tune.grid_search([3]),
                "truncation_pct": tune.grid_search([0.10]),
                "loss_types": tune.grid_search([["bce"]]),
            })
        elif version == "y_v8_s3_t50":
            space.update({
                "c_fusion_mode": tune.grid_search(["v8_evolution_moe"]),
                "v8_scheme": tune.grid_search([3]),
                "truncation_pct": tune.grid_search([0.50]),
                "loss_types": tune.grid_search([["bce"]]),
            })

        # ==========================================
        # 🟣 V8 演进方案 4: 纯特征 MLP Sigmoid (抛弃先验锚点)
        # ==========================================
        elif version == "y_v8_s4":
            space.update({
                "c_fusion_mode": tune.grid_search(["v8_evolution_moe"]),
                "v8_scheme": tune.grid_search([4]),
                # 方案 4 不需要 truncation_pct
                "loss_types": tune.grid_search([["bce"]]),
            })

        # ==========================================
        # 🟣 V8 演进方案 5: 纯 Softmax MMoE (抛弃先验锚点)
        # ==========================================
        elif version == "y_v8_s5":
            space.update({
                "c_fusion_mode": tune.grid_search(["v8_evolution_moe"]),
                "v8_scheme": tune.grid_search([5]),
                # 方案 5 同样不需要 truncation_pct
                "loss_types": tune.grid_search([["bce"]]),
            })

        # ==========================================    
        # 🟢 V9: 靶向 Group DRO 消融实验矩阵 (基于 V6 骨架)
        # 目标：解决 5%-10% 的羊毛党特征崩溃
        # ==========================================
        elif version == "y_v9_dro_a_coarse":
            # 方案 A: 1D 先验粗切 4 组 (0-5%, 5-10%, 10-20%, 20-100%)
            space.update({
                "c_fusion_mode": tune.grid_search(["res_moe"]), # 基于 V6
                "loss_types": tune.grid_search([["group_dro"]]),
                "dro_grouping_mode": tune.grid_search(["1d_coarse"]),
                "dro_ng": tune.grid_search([0.01, 0.05, 0.1]),
                "dro_ema": tune.grid_search([0.1, 0.5, 0.9]),
                "dro_clip": tune.grid_search([0.5])
            })
        elif version == "y_v9_dro_b_coarse":
            # 方案 B: 2D 因果交叉 16 组 (4组 先验 x 4种 T/Y) -> 🌟 核心绝杀版
            space.update({
                "c_fusion_mode": tune.grid_search(["res_moe"]),
                "loss_types": tune.grid_search([["group_dro"]]),
                "dro_grouping_mode": tune.grid_search(["2d_coarse"]),
                "dro_ng": tune.grid_search([0.01, 0.05, 0.1]),
                "dro_ema": tune.grid_search([0.1, 0.5, 0.9]),
                "dro_clip": tune.grid_search([0.5])
            })
        elif version == "y_v9_dro_a_fine":
            # 方案 A_fine: 1D 先验细切 10 组 (十分位)
            space.update({
                "c_fusion_mode": tune.grid_search(["res_moe"]),
                "loss_types": tune.grid_search([["group_dro"]]),
                "dro_grouping_mode": tune.grid_search(["1d_fine"]),
                "dro_ng": tune.grid_search([0.01, 0.05, 0.1]),
                "dro_ema": tune.grid_search([0.1, 0.5, 0.9]),
                "dro_clip": tune.grid_search([0.5])
            })
        elif version == "y_v9_dro_b_fine":
            # 方案 B_fine: 2D 因果细切 40 组 (对比项：用作反证法证明细切会导致过拟合)
            space.update({
                "c_fusion_mode": tune.grid_search(["res_moe"]),
                "loss_types": tune.grid_search([["group_dro"]]),
                "dro_grouping_mode": tune.grid_search(["2d_fine"]),
                "dro_ng": tune.grid_search([0.01, 0.05, 0.1]),
                "dro_ema": tune.grid_search([0.1, 0.5, 0.9]),
                "dro_clip": tune.grid_search([0.5])
            })
        elif version == "y_v10_conflict_wool":
            space.update({
                "c_fusion_mode": tune.grid_search(["res_moe"]), # 基于 V6 强力底座
                "loss_types": tune.grid_search([["prior_conflict"]]),
                "conflict_mode": tune.grid_search(["wool_only"]),
                "conflict_alpha": tune.grid_search([1.0, 3.0, 5.0, 10]), # 探索惩罚力度
            })
        elif version == "y_v10_conflict_gold":
            space.update({
                "c_fusion_mode": tune.grid_search(["res_moe"]),
                "loss_types": tune.grid_search([["prior_conflict"]]),
                "conflict_mode": tune.grid_search(["gold_only"]),
                "conflict_alpha": tune.grid_search([1.0, 3.0, 5.0, 10]),
            })
        elif version == "y_v10_conflict_both":
            space.update({
                "c_fusion_mode": tune.grid_search(["res_moe"]),
                "loss_types": tune.grid_search([["prior_conflict"]]),
                "conflict_mode": tune.grid_search(["both"]),
                "conflict_alpha": tune.grid_search([1.0, 3.0, 5.0, 10]),
            })
        elif version == "y_v10_conflict_all_walkin0705":
            space.update({
                "c_fusion_mode": tune.grid_search(["res_moe"]),
                "loss_types": tune.grid_search([["prior_conflict"]]),
                "conflict_mode": tune.grid_search(["both"]),
                "conflict_alpha": tune.grid_search([0.01, 0.1, 0.5, 1.0, 5.0, 10]),
            })
        elif version == "y_ours_s4_conflict_dim64":
            space.update({"c_fusion_mode": tune.grid_search(["ours_s4_conflict"]), "loss_types": tune.grid_search([["prior_conflict"]]), "hidden_dims": tune.grid_search([[64]]), "conflict_mode": tune.grid_search(["both"]), "conflict_alpha": tune.grid_search([0, 0.01, 0.1, 0.5, 1.0, 5.0, 10])})
        elif version == "y_ours_s4_conflict_dim64_32":
            space.update({"c_fusion_mode": tune.grid_search(["ours_s4_conflict"]), "loss_types": tune.grid_search([["prior_conflict"]]), "hidden_dims": tune.grid_search([[64, 32]]), "conflict_mode": tune.grid_search(["both"]), "conflict_alpha": tune.grid_search([0, 0.01, 0.1, 0.5, 1.0, 5.0, 10])})
        elif version == "y_ours_s4_conflict_dim128_64_32":
            space.update({"c_fusion_mode": tune.grid_search(["ours_s4_conflict"]), "loss_types": tune.grid_search([["prior_conflict"]]), "hidden_dims": tune.grid_search([[128, 64, 32]]), "conflict_mode": tune.grid_search(["both"]), "conflict_alpha": tune.grid_search([0, 0.01, 0.1, 0.5, 1.0, 5.0, 10])})
        elif version == "y_v7_conflict_wool":
            space.update({
                "c_fusion_mode": tune.grid_search(["v7_truncated_moe"]), # 🌟 开启 V7 结构
                "truncation_mode": tune.grid_search(["soft"]),           # V7 专属: 硬截断
                "truncation_temp": tune.grid_search([10.0]),
                "truncation_pct": tune.grid_search([0.05]),              # V7 专属: 锁定前 5%
                "loss_types": tune.grid_search([["prior_conflict"]]),    # 🌟 开启 V10 Loss
                "conflict_mode": tune.grid_search(["wool_only"]),        # V10 专属: 仅惩罚羊毛党
                "conflict_alpha": tune.grid_search([1.0, 3.0, 5.0]),     # 探索惩罚力度
            })
            
        elif version == "y_v7_conflict_gold":
            space.update({
                "c_fusion_mode": tune.grid_search(["v7_truncated_moe"]),
                "truncation_mode": tune.grid_search(["soft"]),           # V7 专属: 硬截断
                "truncation_temp": tune.grid_search([10.0]),
                "truncation_pct": tune.grid_search([0.05]), 
                "loss_types": tune.grid_search([["prior_conflict"]]),
                "conflict_mode": tune.grid_search(["gold_only"]),        # V10 专属: 仅奖励隐藏金子
                "conflict_alpha": tune.grid_search([1.0, 3.0, 5.0,10]),
            })
            
        elif version == "y_v7_conflict_both":
            space.update({
                "c_fusion_mode": tune.grid_search(["v7_truncated_moe"]),
                "truncation_mode": tune.grid_search(["soft"]),           # V7 专属: 硬截断
                "truncation_temp": tune.grid_search([10.0]),
                "truncation_pct": tune.grid_search([0.05]), 
                "loss_types": tune.grid_search([["prior_conflict"]]),
                "conflict_mode": tune.grid_search(["both"]),             # V10 专属: 双向全面纠偏
                "conflict_alpha": tune.grid_search([1.0, 3.0, 5.0,10]),
            })
        elif version == "y_v7_dro_a_coarse":
            space.update({
                "c_fusion_mode": tune.grid_search(["v7_truncated_moe"]), # 🌟 V7 结构
                "truncation_mode": tune.grid_search(["soft"]),           # 🌟 V7 软截断
                "truncation_pct": tune.grid_search([0.05]),              # 🌟 V7 Top 5%
                "truncation_temp": tune.grid_search([10.0]),             
                "loss_types": tune.grid_search([["group_dro"]]),         # 🌟 V9 DRO Loss
                "dro_grouping_mode": tune.grid_search(["1d_coarse"]),
                "dro_ng": tune.grid_search([0.01, 0.05, 0.1]),
                "dro_ema": tune.grid_search([0.1, 0.5, 0.9]),
                "dro_clip": tune.grid_search([0.5])
            })
        elif version == "y_v7_dro_b_coarse":
            space.update({
                "c_fusion_mode": tune.grid_search(["v7_truncated_moe"]),
                "truncation_mode": tune.grid_search(["soft"]),
                "truncation_pct": tune.grid_search([0.05]),
                "truncation_temp": tune.grid_search([10.0]),
                "loss_types": tune.grid_search([["group_dro"]]),
                "dro_grouping_mode": tune.grid_search(["2d_coarse"]),
                "dro_ng": tune.grid_search([0.01, 0.05, 0.1]),
                "dro_ema": tune.grid_search([0.1, 0.5, 0.9]),
                "dro_clip": tune.grid_search([0.5])
            })
        elif version == "y_v7_dro_a_fine":
            space.update({
                "c_fusion_mode": tune.grid_search(["v7_truncated_moe"]),
                "truncation_mode": tune.grid_search(["soft"]),
                "truncation_pct": tune.grid_search([0.05]),
                "truncation_temp": tune.grid_search([10.0]),
                "loss_types": tune.grid_search([["group_dro"]]),
                "dro_grouping_mode": tune.grid_search(["1d_fine"]),
                "dro_ng": tune.grid_search([0.01, 0.05, 0.1]),
                "dro_ema": tune.grid_search([0.1, 0.5, 0.9]),
                "dro_clip": tune.grid_search([0.5])
            })
        elif version == "y_v7_dro_b_fine":
            space.update({
                "c_fusion_mode": tune.grid_search(["v7_truncated_moe"]),
                "truncation_mode": tune.grid_search(["soft"]),
                "truncation_pct": tune.grid_search([0.05]),
                "truncation_temp": tune.grid_search([10.0]),
                "loss_types": tune.grid_search([["group_dro"]]),
                "dro_grouping_mode": tune.grid_search(["2d_fine"]),
                "dro_ng": tune.grid_search([0.01, 0.05, 0.1]),
                "dro_ema": tune.grid_search([0.1, 0.5, 0.9]),
                "dro_clip": tune.grid_search([0.5])
            })
        # ==========================================
        # 🟣 V8: 原始概率空间融合流 (S4, S6, S7, S8)
        # ==========================================
        # elif version == "y_v8_s4":
        #     space.update({"c_fusion_mode": tune.grid_search(["v8_evolution_moe"]), 
        #                   "v8_scheme": tune.grid_search([4]),
        #                     "loss_types": tune.grid_search([["bce"]])})
        elif version == "y_v8_s6":
            space.update({"c_fusion_mode": tune.grid_search(["v8_evolution_moe"]), 
                          "v8_scheme": tune.grid_search([6]), 
                          "loss_types": tune.grid_search([["bce"]])})
        elif version == "y_v8_s7":
            space.update({"c_fusion_mode": tune.grid_search(["v8_evolution_moe"]),
                           "v8_scheme": tune.grid_search([7]), 
                           "loss_types": tune.grid_search([["bce"]])})
        elif version == "y_v8_s8":
            space.update({"c_fusion_mode": tune.grid_search(["v8_evolution_moe"]), 
                          "v8_scheme": tune.grid_search([8]), 
                          "loss_types": tune.grid_search([["bce"]])})

        # ==========================================
        # 🌟 V11: Logit 空间加法融合流 (S4, S6, S7, S8)
        # ==========================================
        elif version == "y_v11_s4":
            space.update({"c_fusion_mode": tune.grid_search(["v11_aligned_moe"]), 
                          "v11_scheme": tune.grid_search([4]), 
                          "align_type": tune.grid_search(["lift"]), 
                          "loss_types": tune.grid_search([["bce"]])})
        elif version == "y_v11_s6":
            space.update({"c_fusion_mode": tune.grid_search(["v11_aligned_moe"]), 
                          "v11_scheme": tune.grid_search([6]), 
                          "align_type": tune.grid_search(["lift"]), 
                          "loss_types": tune.grid_search([["bce"]])})
        elif version == "y_v11_s7":
            space.update({"c_fusion_mode": tune.grid_search(["v11_aligned_moe"]), 
                          "v11_scheme": tune.grid_search([7]), 
                          "align_type": tune.grid_search(["lift"]), 
                          "loss_types": tune.grid_search([["bce"]])})
        elif version == "y_v11_s8":
            space.update({"c_fusion_mode": tune.grid_search(["v11_aligned_moe"]), 
                          "v11_scheme": tune.grid_search([8]), 
                          "align_type": tune.grid_search(["lift"]), 
                          "loss_types": tune.grid_search([["bce"]])})

        elif version == "y_v8_s6_temp0.2":
            space.update({"c_fusion_mode": tune.grid_search(["v8_evolution_moe"]), "v8_scheme": tune.grid_search([6]), "align_temp": tune.grid_search([0.2]), "loss_types": tune.grid_search([["bce"]])})
        elif version == "y_v8_s6_temp0.5":
            space.update({"c_fusion_mode": tune.grid_search(["v8_evolution_moe"]), "v8_scheme": tune.grid_search([6]), "align_temp": tune.grid_search([0.5]), "loss_types": tune.grid_search([["bce"]])})
        elif version == "y_v8_s6_temp1.0":
            space.update({"c_fusion_mode": tune.grid_search(["v8_evolution_moe"]), "v8_scheme": tune.grid_search([6]), "align_temp": tune.grid_search([1.0]), "loss_types": tune.grid_search([["bce"]])})
        elif version == "y_v8_s6_temp2.0":
            space.update({"c_fusion_mode": tune.grid_search(["v8_evolution_moe"]), "v8_scheme": tune.grid_search([6]), "align_temp": tune.grid_search([2.0]), "loss_types": tune.grid_search([["bce"]])})
        elif version == "y_v8_s6_temp5.0":
            space.update({"c_fusion_mode": tune.grid_search(["v8_evolution_moe"]), "v8_scheme": tune.grid_search([6]), "align_temp": tune.grid_search([5.0]), "loss_types": tune.grid_search([["bce"]])})

        elif version == "y_v11_s6_lift_temp0.2":
            space.update({"c_fusion_mode": tune.grid_search(["v11_aligned_moe"]), "v11_scheme": tune.grid_search([6]), "align_type": tune.grid_search(["lift"]), "align_temp": tune.grid_search([0.2]), "loss_types": tune.grid_search([["bce"]])})
        elif version == "y_v11_s6_lift_temp0.5":
            space.update({"c_fusion_mode": tune.grid_search(["v11_aligned_moe"]), "v11_scheme": tune.grid_search([6]), "align_type": tune.grid_search(["lift"]), "align_temp": tune.grid_search([0.5]), "loss_types": tune.grid_search([["bce"]])})
        elif version == "y_v11_s6_lift_temp1.0":
            space.update({"c_fusion_mode": tune.grid_search(["v11_aligned_moe"]), "v11_scheme": tune.grid_search([6]), "align_type": tune.grid_search(["lift"]), "align_temp": tune.grid_search([1.0]), "loss_types": tune.grid_search([["bce"]])})
        elif version == "y_v11_s6_lift_temp2.0":
            space.update({"c_fusion_mode": tune.grid_search(["v11_aligned_moe"]), "v11_scheme": tune.grid_search([6]), "align_type": tune.grid_search(["lift"]), "align_temp": tune.grid_search([2.0]), "loss_types": tune.grid_search([["bce"]])})
        elif version == "y_v11_s6_lift_temp5.0":
            space.update({"c_fusion_mode": tune.grid_search(["v11_aligned_moe"]), "v11_scheme": tune.grid_search([6]), "align_type": tune.grid_search(["lift"]), "align_temp": tune.grid_search([5.0]), "loss_types": tune.grid_search([["bce"]])})
        
        # 2. Z-Score 空间组
        elif version == "y_v11_s6_zscore_temp0.2":
            space.update({"c_fusion_mode": tune.grid_search(["v11_aligned_moe"]), "v11_scheme": tune.grid_search([6]), "align_type": tune.grid_search(["z_score"]), "align_temp": tune.grid_search([0.2]), "loss_types": tune.grid_search([["bce"]])})
        elif version == "y_v11_s6_zscore_temp0.5":
            space.update({"c_fusion_mode": tune.grid_search(["v11_aligned_moe"]), "v11_scheme": tune.grid_search([6]), "align_type": tune.grid_search(["z_score"]), "align_temp": tune.grid_search([0.5]), "loss_types": tune.grid_search([["bce"]])})
        elif version == "y_v11_s6_zscore_temp1.0":
            space.update({"c_fusion_mode": tune.grid_search(["v11_aligned_moe"]), "v11_scheme": tune.grid_search([6]), "align_type": tune.grid_search(["z_score"]), "align_temp": tune.grid_search([1.0]), "loss_types": tune.grid_search([["bce"]])})
        elif version == "y_v11_s6_zscore_temp2.0":
            space.update({"c_fusion_mode": tune.grid_search(["v11_aligned_moe"]), "v11_scheme": tune.grid_search([6]), "align_type": tune.grid_search(["z_score"]), "align_temp": tune.grid_search([2.0]), "loss_types": tune.grid_search([["bce"]])})
        elif version == "y_v11_s6_zscore_temp5.0":
            space.update({"c_fusion_mode": tune.grid_search(["v11_aligned_moe"]), "v11_scheme": tune.grid_search([6]), "align_type": tune.grid_search(["z_score"]), "align_temp": tune.grid_search([5.0]), "loss_types": tune.grid_search([["bce"]])})

        # 3. Rank 空间组
        elif version == "y_v11_s6_rank_temp0.2":
            space.update({"c_fusion_mode": tune.grid_search(["v11_aligned_moe"]), "v11_scheme": tune.grid_search([6]), "align_type": tune.grid_search(["rank"]), "align_temp": tune.grid_search([0.2]), "loss_types": tune.grid_search([["bce"]])})
        elif version == "y_v11_s6_rank_temp0.5":
            space.update({"c_fusion_mode": tune.grid_search(["v11_aligned_moe"]), "v11_scheme": tune.grid_search([6]), "align_type": tune.grid_search(["rank"]), "align_temp": tune.grid_search([0.5]), "loss_types": tune.grid_search([["bce"]])})
        elif version == "y_v11_s6_rank_temp1.0":
            space.update({"c_fusion_mode": tune.grid_search(["v11_aligned_moe"]), "v11_scheme": tune.grid_search([6]), "align_type": tune.grid_search(["rank"]), "align_temp": tune.grid_search([1.0]), "loss_types": tune.grid_search([["bce"]])})
        elif version == "y_v11_s6_rank_temp2.0":
            space.update({"c_fusion_mode": tune.grid_search(["v11_aligned_moe"]), "v11_scheme": tune.grid_search([6]), "align_type": tune.grid_search(["rank"]), "align_temp": tune.grid_search([2.0]), "loss_types": tune.grid_search([["bce"]])})
        elif version == "y_v11_s6_rank_temp5.0":
            space.update({"c_fusion_mode": tune.grid_search(["v11_aligned_moe"]), "v11_scheme": tune.grid_search([6]), "align_type": tune.grid_search(["rank"]), "align_temp": tune.grid_search([5.0]), "loss_types": tune.grid_search([["bce"]])})



# ==========================================================
        # 🔥 V8 温度消融组: 超高温区 (10.0, 20.0, 50.0, 100.0)
        # ==========================================================
        elif version == "y_v8_s6_temp10.0":
            space.update({"c_fusion_mode": tune.grid_search(["v8_evolution_moe"]), "v8_scheme": tune.grid_search([6]), "align_temp": tune.grid_search([10.0]), "loss_types": tune.grid_search([["bce"]])})
        elif version == "y_v8_s6_temp20.0":
            space.update({"c_fusion_mode": tune.grid_search(["v8_evolution_moe"]), "v8_scheme": tune.grid_search([6]), "align_temp": tune.grid_search([20.0]), "loss_types": tune.grid_search([["bce"]])})
        elif version == "y_v8_s6_temp50.0":
            space.update({"c_fusion_mode": tune.grid_search(["v8_evolution_moe"]), "v8_scheme": tune.grid_search([6]), "align_temp": tune.grid_search([50.0]), "loss_types": tune.grid_search([["bce"]])})
        elif version == "y_v8_s6_temp100.0":
            space.update({"c_fusion_mode": tune.grid_search(["v8_evolution_moe"]), "v8_scheme": tune.grid_search([6]), "align_temp": tune.grid_search([100.0]), "loss_types": tune.grid_search([["bce"]])})

        # ==========================================================
        # 🔥 V11 空间与温度消融组: S6 超高温核心战区
        # ==========================================================
        # 1. Lift 空间超高温组
        elif version == "y_v11_s6_lift_temp10.0":
            space.update({"c_fusion_mode": tune.grid_search(["v11_aligned_moe"]), "v11_scheme": tune.grid_search([6]), "align_type": tune.grid_search(["lift"]), "align_temp": tune.grid_search([10.0]), "loss_types": tune.grid_search([["bce"]])})
        elif version == "y_v11_s6_lift_temp20.0":
            space.update({"c_fusion_mode": tune.grid_search(["v11_aligned_moe"]), "v11_scheme": tune.grid_search([6]), "align_type": tune.grid_search(["lift"]), "align_temp": tune.grid_search([20.0]), "loss_types": tune.grid_search([["bce"]])})
        elif version == "y_v11_s6_lift_temp50.0":
            space.update({"c_fusion_mode": tune.grid_search(["v11_aligned_moe"]), "v11_scheme": tune.grid_search([6]), "align_type": tune.grid_search(["lift"]), "align_temp": tune.grid_search([50.0]), "loss_types": tune.grid_search([["bce"]])})
        elif version == "y_v11_s6_lift_temp100.0":
            space.update({"c_fusion_mode": tune.grid_search(["v11_aligned_moe"]), "v11_scheme": tune.grid_search([6]), "align_type": tune.grid_search(["lift"]), "align_temp": tune.grid_search([100.0]), "loss_types": tune.grid_search([["bce"]])})
        
        # 2. Z-Score 空间超高温组
        elif version == "y_v11_s6_zscore_temp10.0":
            space.update({"c_fusion_mode": tune.grid_search(["v11_aligned_moe"]), "v11_scheme": tune.grid_search([6]), "align_type": tune.grid_search(["z_score"]), "align_temp": tune.grid_search([10.0]), "loss_types": tune.grid_search([["bce"]])})
        elif version == "y_v11_s6_zscore_temp20.0":
            space.update({"c_fusion_mode": tune.grid_search(["v11_aligned_moe"]), "v11_scheme": tune.grid_search([6]), "align_type": tune.grid_search(["z_score"]), "align_temp": tune.grid_search([20.0]), "loss_types": tune.grid_search([["bce"]])})
        elif version == "y_v11_s6_zscore_temp50.0":
            space.update({"c_fusion_mode": tune.grid_search(["v11_aligned_moe"]), "v11_scheme": tune.grid_search([6]), "align_type": tune.grid_search(["z_score"]), "align_temp": tune.grid_search([50.0]), "loss_types": tune.grid_search([["bce"]])})
        elif version == "y_v11_s6_zscore_temp100.0":
            space.update({"c_fusion_mode": tune.grid_search(["v11_aligned_moe"]), "v11_scheme": tune.grid_search([6]), "align_type": tune.grid_search(["z_score"]), "align_temp": tune.grid_search([100.0]), "loss_types": tune.grid_search([["bce"]])})

        # 3. Rank 空间超高温组
        elif version == "y_v11_s6_rank_temp10.0":
            space.update({"c_fusion_mode": tune.grid_search(["v11_aligned_moe"]), "v11_scheme": tune.grid_search([6]), "align_type": tune.grid_search(["rank"]), "align_temp": tune.grid_search([10.0]), "loss_types": tune.grid_search([["bce"]])})
        elif version == "y_v11_s6_rank_temp20.0":
            space.update({"c_fusion_mode": tune.grid_search(["v11_aligned_moe"]), "v11_scheme": tune.grid_search([6]), "align_type": tune.grid_search(["rank"]), "align_temp": tune.grid_search([20.0]), "loss_types": tune.grid_search([["bce"]])})
        elif version == "y_v11_s6_rank_temp50.0":
            space.update({"c_fusion_mode": tune.grid_search(["v11_aligned_moe"]), "v11_scheme": tune.grid_search([6]), "align_type": tune.grid_search(["rank"]), "align_temp": tune.grid_search([50.0]), "loss_types": tune.grid_search([["bce"]])})
        elif version == "y_v11_s6_rank_temp100.0":
            space.update({"c_fusion_mode": tune.grid_search(["v11_aligned_moe"]), "v11_scheme": tune.grid_search([6]), "align_type": tune.grid_search(["rank"]), "align_temp": tune.grid_search([100.0]), "loss_types": tune.grid_search([["bce"]])})



        # --------------------baselines
        elif version == "y_mtmt_mlp":
            space.update({
                "model": tune.grid_search(["MTMT"]),
                "loss_types": tune.grid_search([["bce"]]), 
                # 🌟 MLP 核心搜索矩阵
                "expert_type": tune.grid_search(["mlp"]),
                "expert_hidden_dims": tune.grid_search([
                    [128, 64, 32], # 深且漏斗
                    [64, 32],      # 浅层漏斗
                    [64]           # 极简单层
                ]),
                "dropout_rate": tune.grid_search([0.1]),
                "num_experts": tune.grid_search([4]),    
                "t_emb_dim": tune.grid_search([16]), 
                "aux_weight": tune.grid_search([0.1, 0.5, 1.0]) 
            })
        elif version == "y_mtmt_resnet":
            space.update({
                "model": tune.grid_search(["MTMT"]),
                "loss_types": tune.grid_search([["bce"]]), 
                "expert_type": tune.grid_search(["resnet18"]),
                "expert_hidden_dims": tune.grid_search([[32], [64], [128]]),
                "num_experts": tune.grid_search([4]),    
                "t_emb_dim": tune.grid_search([16]), 
                "batch_size": tune.grid_search([15360]),
                "aux_weight": tune.grid_search([0.5, 1.0, 0.1]) 
            })
        elif version == "y_ecup":
            space.update({
                "model": tune.grid_search(["ECUP"]),
                "loss_types": tune.grid_search([["bce"]]), # ECUP 内部接管了 Loss，留空即可
                
                # ------------------------------------------
                # 🌟 核心 1: 严格对齐 Table 2 的网格搜索空间
                # ------------------------------------------
                # "learning_rate": tune.grid_search([1e-4, 1e-3, 1e-2]), # 对应论文 lr
                "d_dim": tune.grid_search([8, 16, 32]),                # 对应论文 d (Embedding dimension)
                "tower_h": tune.grid_search([128]), # ([128, 256, 512]),          # 对应论文 h (Task's hidden units of first layer)
                "tae_h": tune.grid_search([128]), # ([64, 128, 256, 512]),        # 对应论文 h_gate (TAEGate's hidden units)
                
                # ------------------------------------------
                # 🌟 核心 2: 严格对齐 5.1.2 节的固定网络结构
                # ------------------------------------------
                "num_heads": tune.grid_search([2]),      # 原文: number of heads of the multi-head attention is 2
                
                # ------------------------------------------
                # 基础训练参数
                # ------------------------------------------
                "batch_size": tune.grid_search([65536]),  # 原文: batch_size is 2^11
                # "num_epochs": tune.grid_search([50]),
                "gamma": tune.grid_search([1.0]),
                "ctcvr_weight": tune.grid_search([1.0])  # 默认 1:1 联合优化
            })
        elif version == "y_motto":
            space.update({
                "model": tune.grid_search(["MOTTO"]),
                "loss_types": tune.grid_search([["bce"]]), 
                
                # "learning_rate": tune.grid_search([1e-3]),
                # "batch_size": tune.grid_search([4096]), 
                # "num_epochs": tune.grid_search([50]),
                
                # 🌟 加入初始特征投影维度
                "d_dim": tune.grid_search([16]), 
                "bottom_dim": tune.grid_search([64]),
                # 🌟 这里你可以自由探索专家的“漏斗形状”或“直筒形状”了
                "expert_hidden_dims": tune.grid_search([[64], [64, 32], [128, 64, 32]]),
                "tower_dim": tune.grid_search([64]),
                
                "use_specific_experts": tune.grid_search([True]),
                "alpha_sda": tune.grid_search([0.1, 0.5, 1.0, 5.0]), 
                "aux_weight": tune.grid_search([0.1, 0.5, 1.0])
            })
        elif version == "y_efin_0717new":
            space.update({
                "model": tune.grid_search(["EFIN_0717new"]),
                "efin_hu_dim": tune.grid_search([128,64,32]),
                "efin_hc_dim": tune.sample_from(lambda spec: spec.config["efin_hu_dim"]),  
                # "efin_hc_dim": tune.grid_search([128,64,32]),
                "batch_size": tune.grid_search([66536]),
                "efin_is_self": tune.grid_search([False]),
                "efin_dropout": tune.grid_search([0.0]),
                "weight_decay": tune.grid_search([1e-5, 1e-4, 1e-3, 1e-2]),
            })
            
        elif version == "y_ecup_0717new":
            space.update({
                "model": tune.grid_search(["ECUP_0717new"]),
                "loss_types": tune.grid_search([["bce"]]),
                "learning_rate": tune.grid_search([1e-3]),
                "d_dim": tune.grid_search([32]), # 16 8 32
                "tower_h": tune.grid_search([128]), # 128 64 32
                "tae_h": tune.grid_search([128]),
                "num_heads": tune.grid_search([2]), # 1 2 4
                "batch_size": tune.grid_search([66536]),
                "gamma": tune.grid_search([1.0]),
                "ctcvr_weight": tune.grid_search([0.1, 0.5, 1.0, 2,  10]),
                "weight_decay": tune.grid_search([1e-5, 1e-4, 1e-3, 1e-2]),
            })
            
        elif version == "y_mtmt_0717new":
            space.update({
                "model": tune.grid_search(["MTMT_0717new"]),
                "loss_types": tune.grid_search([["bce"]]),
                "expert_type": tune.grid_search(["mlp", "resnet18"]),
                "expert_hidden_dims": tune.grid_search([[64, 32],[128,64,32],[32]]),  # resnet18 只取 [0]=64 当输出维
                "num_experts": tune.grid_search([4]), # 2 4 8
                "t_emb_dim": tune.grid_search([16]), # 16 8 32
                "dropout_rate": tune.grid_search([0]),
                "aux_weight": tune.grid_search([0.01, 0.1, 1.0, 10]),
            })

        # 🟢 版本 4：TARNET Naive Multitask (主任务 Y + 辅任务 C 联合优化)
        elif version == "y_v1_naive_mt_small_c_weight":
            space.update({
                "model": tune.grid_search(["TARNET_MT"]),  # 强制顶掉默认的 TARNET
                "c_fusion_mode": tune.grid_search(["none"]), 
                "mt_c_weight": tune.grid_search([0.1, 0.3, 0.5]), # 探索辅助任务拉扯的力度
                "loss_types": tune.grid_search([["bce"]]),        # Baseline 保持纯净
            })
        elif version == "y_v1_naive_mt_larger_c_weight":
            space.update({
                "model": tune.grid_search(["TARNET_MT"]),  # 强制顶掉默认的 TARNET
                "c_fusion_mode": tune.grid_search(["none"]), 
                "mt_c_weight": tune.grid_search([1.0, 2.0, 5.0]), # 探索辅助任务拉扯的力度
                "loss_types": tune.grid_search([["bce"]]),        # Baseline 保持纯净
            })
# ==========================================================
        # 🟢 [前4组] PureV10 核心蛊王战区
        # ==========================================================
# ==========================================================
        # 🟢 [1-8 组] PureV10 核心参数化强灌战区 (全面重构去 group 命名)
        # ==========================================================
        elif version == "y_pure_v10_h32_a1.0_wd1e4":
            space.update({
                "model": tune.grid_search(["TARNET_Baseline_PureV10"]), "c_fusion_mode": tune.grid_search(["res_moe"]),
                "loss_types": tune.grid_search([["prior_conflict"]]), "head_hidden_dims": tune.grid_search([[32]]),
                "conflict_mode": tune.grid_search(["all"]), 
                "conflict_alpha_wool": tune.grid_search([1.0]), "conflict_alpha_gold": tune.grid_search([1.0]), "conflict_alpha_walkin": tune.grid_search([1.0]), 
                "weight_decay": tune.grid_search([0.0001]),
            })
        elif version == "y_pure_v10_h16_a5.0_wd1e5":
            space.update({
                "model": tune.grid_search(["TARNET_Baseline_PureV10"]), "c_fusion_mode": tune.grid_search(["res_moe"]),
                "loss_types": tune.grid_search([["prior_conflict"]]), "head_hidden_dims": tune.grid_search([[16]]),
                "conflict_mode": tune.grid_search(["all"]), 
                "conflict_alpha_wool": tune.grid_search([5.0]), "conflict_alpha_gold": tune.grid_search([5.0]), "conflict_alpha_walkin": tune.grid_search([5.0]), 
                "weight_decay": tune.grid_search([1e-05]),
            })
        elif version == "y_pure_v10_h64_32_a0.5_wd1e4":
            space.update({
                "model": tune.grid_search(["TARNET_Baseline_PureV10"]), "c_fusion_mode": tune.grid_search(["res_moe"]),
                "loss_types": tune.grid_search([["prior_conflict"]]), "head_hidden_dims": tune.grid_search([[64, 32]]),
                "conflict_mode": tune.grid_search(["all"]), 
                "conflict_alpha_wool": tune.grid_search([0.5]), "conflict_alpha_gold": tune.grid_search([0.5]), "conflict_alpha_walkin": tune.grid_search([0.5]), 
                "weight_decay": tune.grid_search([0.0001]),
            })
        elif version == "y_pure_v10_h16_a1.0_wd1e4":
            space.update({
                "model": tune.grid_search(["TARNET_Baseline_PureV10"]), "c_fusion_mode": tune.grid_search(["res_moe"]),
                "loss_types": tune.grid_search([["prior_conflict"]]), "head_hidden_dims": tune.grid_search([[16]]),
                "conflict_mode": tune.grid_search(["all"]), 
                "conflict_alpha_wool": tune.grid_search([1.0]), "conflict_alpha_gold": tune.grid_search([1.0]), "conflict_alpha_walkin": tune.grid_search([1.0]), 
                "weight_decay": tune.grid_search([0.0001]),
            })
        elif version == "y_pure_v10_hNone_a0.5_5.0_1.0_wd1e5":
            space.update({
                "model": tune.grid_search(["TARNET_Baseline_PureV10"]), "c_fusion_mode": tune.grid_search(["res_moe"]),
                "loss_types": tune.grid_search([["prior_conflict"]]), "head_hidden_dims": tune.grid_search([None]),
                "conflict_mode": tune.grid_search(["all"]), 
                "conflict_alpha_wool": tune.grid_search([0.5]), "conflict_alpha_gold": tune.grid_search([5.0]), "conflict_alpha_walkin": tune.grid_search([1.0]), 
                "weight_decay": tune.grid_search([1e-05]),
            })
        elif version == "y_pure_v10_hNone_a0.5_10.0_0.5_wd1e5":
            space.update({
                "model": tune.grid_search(["TARNET_Baseline_PureV10"]), "c_fusion_mode": tune.grid_search(["res_moe"]),
                "loss_types": tune.grid_search([["prior_conflict"]]), "head_hidden_dims": tune.grid_search([None]),
                "conflict_mode": tune.grid_search(["all"]), 
                "conflict_alpha_wool": tune.grid_search([0.5]), "conflict_alpha_gold": tune.grid_search([10.0]), "conflict_alpha_walkin": tune.grid_search([0.5]), 
                "weight_decay": tune.grid_search([1e-05]),
            })
        elif version == "y_pure_v10_hNone_a0.5_0.5_0.1_wd1e5":
            space.update({
                "model": tune.grid_search(["TARNET_Baseline_PureV10"]), "c_fusion_mode": tune.grid_search(["res_moe"]),
                "loss_types": tune.grid_search([["prior_conflict"]]), "head_hidden_dims": tune.grid_search([None]),
                "conflict_mode": tune.grid_search(["all"]), 
                "conflict_alpha_wool": tune.grid_search([0.5]), "conflict_alpha_gold": tune.grid_search([0.5]), "conflict_alpha_walkin": tune.grid_search([0.1]), 
                "weight_decay": tune.grid_search([1e-05]),
            })
        elif version == "y_pure_v10_hNone_a0.5_0.5_5.0_wd1e5":
            space.update({
                "model": tune.grid_search(["TARNET_Baseline_PureV10"]), "c_fusion_mode": tune.grid_search(["res_moe"]),
                "loss_types": tune.grid_search([["prior_conflict"]]), "head_hidden_dims": tune.grid_search([None]),
                "conflict_mode": tune.grid_search(["all"]), 
                "conflict_alpha_wool": tune.grid_search([0.5]), "conflict_alpha_gold": tune.grid_search([0.5]), "conflict_alpha_walkin": tune.grid_search([5.0]), 
                "weight_decay": tune.grid_search([1e-05]),
            })

        # ==========================================================
        # 🔵 [9-17 组] Ours V8 S6 参数化强灌战区 (精准挂载 v8_s6 拓扑)
        # ==========================================================
        elif version == "y_ours_v8s6_h32_a10.0_t1_wd0.01":
            space.update({
                "model": tune.grid_search(["TARNET"]), "c_fusion_mode": tune.grid_search(["ours_v8_s6"]), 
                "loss_types": tune.grid_search([["prior_conflict"]]), "head_hidden_dims": tune.grid_search([[32]]),
                "conflict_mode": tune.grid_search(["all"]), 
                "conflict_alpha_wool": tune.grid_search([10.0]), "conflict_alpha_gold": tune.grid_search([10.0]), "conflict_alpha_walkin": tune.grid_search([10.0]), 
                "ours_s6_temp": tune.grid_search([1.0]), "weight_decay": tune.grid_search([0.01]),
            })
        elif version == "y_ours_v8s6_h32_a0.1_t20_wd1e5":
            space.update({
                "model": tune.grid_search(["TARNET"]), "c_fusion_mode": tune.grid_search(["ours_v8_s6"]), 
                "loss_types": tune.grid_search([["prior_conflict"]]), "head_hidden_dims": tune.grid_search([[32]]),
                "conflict_mode": tune.grid_search(["all"]), 
                "conflict_alpha_wool": tune.grid_search([0.1]), "conflict_alpha_gold": tune.grid_search([0.1]), "conflict_alpha_walkin": tune.grid_search([0.1]), 
                "ours_s6_temp": tune.grid_search([20.0]), "weight_decay": tune.grid_search([1e-05]),
            })
        elif version == "y_ours_v8s6_h32_a0.5_t20_wd1e5":
            space.update({
                "model": tune.grid_search(["TARNET"]), "c_fusion_mode": tune.grid_search(["ours_v8_s6"]), 
                "loss_types": tune.grid_search([["prior_conflict"]]), "head_hidden_dims": tune.grid_search([[32]]),
                "conflict_mode": tune.grid_search(["all"]), 
                "conflict_alpha_wool": tune.grid_search([0.5]), "conflict_alpha_gold": tune.grid_search([0.5]), "conflict_alpha_walkin": tune.grid_search([0.5]), 
                "ours_s6_temp": tune.grid_search([20.0]), "weight_decay": tune.grid_search([1e-05]),
            })
        elif version == "y_ours_v8s6_h32_a10.0_t20_wd0.01":
            space.update({
                "model": tune.grid_search(["TARNET"]), "c_fusion_mode": tune.grid_search(["ours_v8_s6"]), 
                "loss_types": tune.grid_search([["prior_conflict"]]), "head_hidden_dims": tune.grid_search([[32]]),
                "conflict_mode": tune.grid_search(["all"]), 
                "conflict_alpha_wool": tune.grid_search([10.0]), "conflict_alpha_gold": tune.grid_search([10.0]), "conflict_alpha_walkin": tune.grid_search([10.0]), 
                "ours_s6_temp": tune.grid_search([20.0]), "weight_decay": tune.grid_search([0.01]),
            })
        elif version == "y_ours_v8s6_hNone_a1.0_t20_wd1e5":
            space.update({
                "model": tune.grid_search(["TARNET"]), "c_fusion_mode": tune.grid_search(["ours_v8_s6"]), 
                "loss_types": tune.grid_search([["prior_conflict"]]), "head_hidden_dims": tune.grid_search([None]),
                "conflict_mode": tune.grid_search(["all"]), 
                "conflict_alpha_wool": tune.grid_search([1.0]), "conflict_alpha_gold": tune.grid_search([1.0]), "conflict_alpha_walkin": tune.grid_search([1.0]), 
                "ours_s6_temp": tune.grid_search([20.0]), "weight_decay": tune.grid_search([1e-05]),
            })
        elif version == "y_ours_v8s6_hNone_a1.0_t1_wd0.001":
            space.update({
                "model": tune.grid_search(["TARNET"]), "c_fusion_mode": tune.grid_search(["ours_v8_s6"]), 
                "loss_types": tune.grid_search([["prior_conflict"]]), "head_hidden_dims": tune.grid_search([None]),
                "conflict_mode": tune.grid_search(["all"]), 
                "conflict_alpha_wool": tune.grid_search([1.0]), "conflict_alpha_gold": tune.grid_search([1.0]), "conflict_alpha_walkin": tune.grid_search([1.0]), 
                "ours_s6_temp": tune.grid_search([1.0]), "weight_decay": tune.grid_search([0.001]),
            })
        elif version == "y_ours_v8s6_h16_a0.1_t1_wd1e5":
            space.update({
                "model": tune.grid_search(["TARNET"]), "c_fusion_mode": tune.grid_search(["ours_v8_s6"]), 
                "loss_types": tune.grid_search([["prior_conflict"]]), "head_hidden_dims": tune.grid_search([[16]]),
                "conflict_mode": tune.grid_search(["all"]), 
                "conflict_alpha_wool": tune.grid_search([0.1]), "conflict_alpha_gold": tune.grid_search([0.1]), "conflict_alpha_walkin": tune.grid_search([0.1]), 
                "ours_s6_temp": tune.grid_search([1.0]), "weight_decay": tune.grid_search([1e-05]),
            })
        elif version == "y_ours_v8s6_h32_16_a5.0_t1_wd0.001":
            space.update({
                "model": tune.grid_search(["TARNET"]), "c_fusion_mode": tune.grid_search(["ours_v8_s6"]), 
                "loss_types": tune.grid_search([["prior_conflict"]]), "head_hidden_dims": tune.grid_search([[32, 16]]),
                "conflict_mode": tune.grid_search(["all"]), 
                "conflict_alpha_wool": tune.grid_search([5.0]), "conflict_alpha_gold": tune.grid_search([5.0]), "conflict_alpha_walkin": tune.grid_search([5.0]), 
                "ours_s6_temp": tune.grid_search([1.0]), "weight_decay": tune.grid_search([0.001]),
            })
        elif version == "y_ours_v8s6_h16_a10.0_t1_wd0.01":
            space.update({
                "model": tune.grid_search(["TARNET"]), "c_fusion_mode": tune.grid_search(["ours_v8_s6"]), 
                "loss_types": tune.grid_search([["prior_conflict"]]), "head_hidden_dims": tune.grid_search([[16]]),
                "conflict_mode": tune.grid_search(["all"]), 
                "conflict_alpha_wool": tune.grid_search([10.0]), "conflict_alpha_gold": tune.grid_search([10.0]), "conflict_alpha_walkin": tune.grid_search([10.0]), 
                "ours_s6_temp": tune.grid_search([1.0]), "weight_decay": tune.grid_search([0.01]),
            })
        elif version == "y_baseline_canniuplift_ms":
    # 默认单点配置 —— 先验证跑通（forward/backward/loss 下降），再跑下面几个网格。
            space.update({
                "model": tune.grid_search(["CanniUplift"]),
                "canniuplift_use_treat_attn": tune.grid_search([True]),
                "canniuplift_attn_d_dim": tune.grid_search([16]),
                "canniuplift_attn_num_heads": tune.grid_search([2]),
                "canniuplift_seller_w": tune.grid_search([1.0]),
                "canniuplift_rdd_w": tune.grid_search([1.0]),
                "canniuplift_redem_w": tune.grid_search([1.0]),
                "canniuplift_iptw_w": tune.grid_search([1.0]),
            })
        elif version == "y_baseline_canniuplift_paper_grid":
            # ------------------------------------------
            # 🌟 严格对齐论文 §4.3 Experimental Setup 公开的搜索空间：
            #   "The search space includes learning rate η ∈ {5e-4, 1e-4, 5e-5}
            #    and hidden dimension dh ∈ {128, 256, 512}."
            # 论文用 Optuna 40 trials/自动调参；本仓是穷举 grid，故这里老实穷举 3×3=9 点。
            # 论文只给了单个 dh（塔第一层宽度），这里按论文语义把塔统一设成 [dh, dh//2]
            # 两层（第二层减半，贴近论文 "hidden dimension" 单旋钮、而不是我们自己那套
            # 四档 cap_grid 的多层写法）。batch_size/L2 论文写死 4096 / 1e-4，不在这个
            # version 里穷举（避免和 _baseline_fixed_common 的批大小口径打架，见下方注释）。
            # ------------------------------------------
            space.update({
                "model": tune.grid_search(["CanniUplift"]),
                "learning_rate": tune.grid_search([1e-3]),      # 论文 η
                "hidden_dims": tune.grid_search([[256, 128], [128, 64],  [128,64,32],[64,32],[64]]),  # 论文 dh
                "weight_decay": tune.grid_search([1e-4,1e-5,1e-3,1e-2]),                   # 论文 L2 λ=1e-4
                "canniuplift_use_treat_attn": tune.grid_search([True]),
                "canniuplift_attn_d_dim": tune.grid_search([16]),
                "canniuplift_attn_num_heads": tune.grid_search([2]),
                "canniuplift_seller_w": tune.grid_search([1.0]),
                "canniuplift_rdd_w": tune.grid_search([1.0,0.5,2]),
                # "canniuplift_redem_w": tune.grid_search([1.0]),
                "canniuplift_iptw_w": tune.grid_search([1.0]),
                "canniuplift_redem_w": tune.sample_from(lambda spec: spec.config["canniuplift_rdd_w"]),
            })
        elif version == "y_baseline_canniuplift_cap_grid":
            # 容量四档扫（同 rankzoo air_zhongjie_pytorch/search_space.py 及 TF 版
            # config/st_baseline_cap_units.py 的 l1c05/l2c05/l2c1/l3c1 四档口径），
            # 其余超参固定默认值 —— 定位 Seller/RDD 塔宽度对 wAUUC 的影响。
            # 注意：论文 §4.3 只搜了单层宽度 {128,256,512}（见上面 paper_grid 版）；
            # 这四档「层数+宽度」组合是 rankzoo/本仓的自定义规模档位，论文未公开。
            space.update({
                "model": tune.grid_search(["CanniUplift"]),
                "hidden_dims": tune.grid_search([[64], [64, 32], [128, 64], [128, 64, 32]]),
                "canniuplift_use_treat_attn": tune.grid_search([True]),
                "canniuplift_attn_d_dim": tune.grid_search([16]),
                "canniuplift_attn_num_heads": tune.grid_search([2]),
                "canniuplift_seller_w": tune.grid_search([1.0]),
                "canniuplift_rdd_w": tune.grid_search([1.0]),
                "canniuplift_redem_w": tune.grid_search([1.0]),
                "canniuplift_iptw_w": tune.grid_search([1.0]),
            })
        elif version == "y_baseline_canniuplift_ablation":
            # Treat-Attn 开关 ablation —— 定位 attn 收益是否来自字段交互本身。
            # 提醒：这不是论文 Table 2 的模块消融（论文消融的是 PGA / RDD 是否启用，
            # baseline=EUEN）。本仓没实现 PGA（无多 seller 候选集结构），所以做不了
            # 论文那张表；这里消融的是我们自己为了适配表格数据而新增的「字段版
            # Treat-Attention」，验证它本身是否比不用 attn（纯 encoder 拼接）更好。
            # 若要贴近论文 Table 2 的叙事，应额外把本仓已有的 EUEN baseline 拉来同条件
            # 对比（RDD 有/无 = CanniUplift vs EUEN），而不是只看 attn 开关。
            space.update({
                "model": tune.grid_search(["CanniUplift"]),
                "canniuplift_use_treat_attn": tune.grid_search([True, False]),
                "canniuplift_attn_d_dim": tune.grid_search([16]),
                "canniuplift_attn_num_heads": tune.grid_search([2]),
                "canniuplift_seller_w": tune.grid_search([1.0]),
                "canniuplift_rdd_w": tune.grid_search([1.0]),
                "canniuplift_redem_w": tune.grid_search([1.0]),
                "canniuplift_iptw_w": tune.grid_search([1.0]),
            })
        elif version == "y_baseline_canniuplift_loss_weight_grid":
            # L_rdd / L_redem 相对 L_seller 的权重扫（容量固定默认 [128,64]）。
            # 论文 Eq.17 直接把三/两项 loss 系数设为 1（Ltotal = LGMV_Tw + Lredem，
            # 未公开做过权重扫描），这里的 {0.5,1.0,2.0} 网格是本仓自定义补充，
            # 不是论文数值。
            space.update({
                "model": tune.grid_search(["CanniUplift"]),
                "canniuplift_use_treat_attn": tune.grid_search([True]),
                "canniuplift_rdd_w": tune.grid_search([0.5, 1.0, 2.0]),
                "canniuplift_redem_w": tune.grid_search([0.5, 1.0, 2.0]),
            })
        

        else:
            space.update({
                "c_fusion_mode": tune.grid_search(["none"]),
                "loss_types": tune.grid_search([
                    ["bce"]]),
            })
            
    return space

if __name__ == "__main__":
    import json
    print("🚀 开始验证 search_space.py 模块...\n")
    default_params = get_default_hyperparams(task="train_y", version="v2_full_magic")
    print("🧪 验证 version=v2_full_magic 的 Y 阶段默认参数：")
    print(json.dumps(default_params, indent=4, ensure_ascii=False))