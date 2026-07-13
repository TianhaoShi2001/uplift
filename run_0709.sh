#!/bin/bash
# =========================================================================
# 🚀 Uplift Modeling 纯 V10 突围战：8组全量联动消融大盘 A/B 组并发脚本 (Exp 均值对齐版)
# 物理逻辑：两路大军后台异步并发（&），分流到指定 GPU，exp_name 统一追加 _mean 后缀
# =========================================================================
ulimit -n 65536

# -------------------------------------------------------------------------
# 🛠️ 自由指定 GPU 与卡槽分配（在这里改卡号）
# -------------------------------------------------------------------------
NUM_GPU=0.06     # 每个 Trial 分配的 GPU 份额
CUDA_A="3,0"       # 🎯 流水线 A 绑定的 GPU 卡号 (负责 3 组 Solo 实验 + 1 组 Head 拓扑实验)
CUDA_B="0"       # 🎯 流水线 B 绑定的 GPU 卡号 (负责 4 组人群同参数 Mix 基础实验)
LOG_DIR="experiment_logs_pure_v10"
mkdir -p $LOG_DIR

echo "========================================================"
echo "🚀 启动 纯 V10 (Naked Head) A/B 双轴流并发大阅兵"
echo "▶️ [流水线 A] 将独占 GPU: ${CUDA_A} (处理 Solo 实验与 Head 拓扑探索实验)"
echo "▶️ [流水线 B] 将独占 GPU: ${CUDA_B} (处理多人群 Mix 同参数同步长实验)"
echo "📄 所有任务的详细输出将保存在 $LOG_DIR/ 目录下"
echo "========================================================"

# --------------------------------------------------------------------------+
# 🟢 [流水线 A] 核心单兵与多档预测头战区 (独占 GPU_A)
# -------------------------------------------------------------------------
(
    # echo "🟢 [流水线 A] 已在 GPU ${CUDA_A} 正式起航！"
    
    # echo "▶️ [流水线 A] 正在训练 [1/8]: 纯羊毛党靶向重击战 (y_pure_v10_solo_wool)..."
    # python main.py --mode tune --task train_y --model TARNET_Baseline_PureV10 --version y_pure_v10_solo_wool --exp_name exp_v10_solo_wool_mean --cuda $CUDA_A --num_per_gpu $NUM_GPU > $LOG_DIR/log_v10_solo_wool.txt 2>&1
    
    # echo "▶️ [流水线 A] 正在训练 [2/8]: 纯隐藏金子绝对拯救战 (y_pure_v10_solo_gold)..."
    # python main.py --mode tune --task train_y --model TARNET_Baseline_PureV10 --version y_pure_v10_solo_gold --exp_name exp_v10_solo_gold_mean --cuda $CUDA_A --num_per_gpu $NUM_GPU > $LOG_DIR/log_v10_solo_gold.txt 2>&1
    
    # echo "▶️ [流水线 A] 正在训练 [3/8]: 纯自然进店镜像拦截战 (y_pure_v10_solo_walkin)..."
    # python main.py --mode tune --task train_y --model TARNET_Baseline_PureV10 --version y_pure_v10_solo_walkin --exp_name exp_v10_solo_walkin_mean --cuda $CUDA_A --num_per_gpu $NUM_GPU > $LOG_DIR/log_v10_solo_walkin.txt 2>&1

    echo "👑 [流水线 A] 正在训练 [4/8]: 经典双向对齐+多档预测头探索战 (y_pure_v10_mix_both_same_alpha_head)..."
    python main.py --mode tune --task train_y --model TARNET_Baseline_PureV10 --version y_pure_v10_all_same_alpha_head_0710 --exp_name exp_y_pure_v10_all_same_alpha_head_0710_res_2_layer_head_search --cuda $CUDA_A --num_per_gpu $NUM_GPU > $LOG_DIR/log_y_pure_v10_all_same_alpha_head_0710.txt 2>&1

    echo "✅ [流水线 A] 4 组靶向与预测头消融实验全部收网！"
) 

# # -------------------------------------------------------------------------
# # 🔵 [流水线 B] 多人群同步长联动战区 (独占 GPU_B)
# # -------------------------------------------------------------------------
# (
#     echo "🔵 [流水线 B] 已在 GPU ${CUDA_B} 正式起航！"
    
#     echo "▶️ [流水线 B] 正在训练 [5/8]: 三人群(All)共享同一参数联动战 (y_pure_v10_mix_all_same_alpha)..."
#     python main.py --mode tune --task train_y --model TARNET_Baseline_PureV10 --version y_pure_v10_mix_all_same_alpha --exp_name exp_v10_mix_all_mean --cuda $CUDA_B --num_per_gpu $NUM_GPU > $LOG_DIR/log_v10_mix_all.txt 2>&1
    
#     echo "▶️ [流水线 B] 正在训练 [6/8]: 经典双向对称对齐战 (y_pure_v10_mix_both_same_alpha)..."
#     python main.py --mode tune --task train_y --model TARNET_Baseline_PureV10 --version y_pure_v10_mix_both_same_alpha --exp_name exp_v10_mix_both_mean --cuda $CUDA_B --num_per_gpu $NUM_GPU > $LOG_DIR/log_v10_mix_both.txt 2>&1
    
#     echo "▶️ [流水线 B] 正在训练 [7/8]: 特定组合 A-羊毛+自然进店 (y_pure_v10_mix_wool_walkin_same_alpha)..."
#     python main.py --mode tune --task train_y --model TARNET_Baseline_PureV10 --version y_pure_v10_mix_wool_walkin_same_alpha --exp_name exp_v10_mix_wool_walkin_mean --cuda $CUDA_B --num_per_gpu $NUM_GPU > $LOG_DIR/log_v10_mix_wool_walkin.txt 2>&1

#     echo "▶️ [流水线 B] 正在训练 [8/8]: 特定组合 B-自然进店+隐藏金子 (y_pure_v10_mix_walkin_gold_same_alpha)..."
#     python main.py --mode tune --task train_y --model TARNET_Baseline_PureV10 --version y_pure_v10_mix_walkin_gold_same_alpha --exp_name exp_v10_mix_walkin_gold_mean --cuda $CUDA_B --num_per_gpu $NUM_GPU > $LOG_DIR/log_v10_mix_walkin_gold.txt 2>&1

#     echo "✅ [流水线 B] 4 组基础多人群联动消融实验全部收网！"
# ) &

# 🛑 核心物理阻断：死死按住主 shell 进程，等待 A/B 两路大军合流
wait

echo "========================================================"
echo "🎉 终极捷报：大盘 8 组（含多档 Head）纯 V10 消融试验全部平稳落幕！"
echo "========================================================"