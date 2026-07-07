#!/bin/bash
# =========================================================================
# 🚀 Uplift Modeling 纯 V10 突围战：6 组单变量分离与循序渐进消融全自动流水线
# 包含：单人群靶向探索 (Solo) + 相同步长全人群联动 (Mix All / Both / Wool+Walkin)
# =========================================================================
ulimit -n 65536

# -------------------------------------------------------------------------
# 🛠️ 全局资源与硬件配置 (支持双核并发，避免多任务抢卡爆显存)
# -------------------------------------------------------------------------
NUM_GPU=0.06
NUM_GPU_B=0.1
CUDA_A="3"
CUDA_B="0"
LOG_DIR="experiment_logs_pure_v10"
mkdir -p $LOG_DIR

echo "========================================================"
echo "🚀 启动 纯 V10 (Residual Base + 裸 Linear Head) 循序渐进大阅兵"
echo "▶️ [流水线 A] 将使用 GPU: ${CUDA_A} (负责前 3 组单人群靶向 Solo 实验)"
echo "▶️ [流水线 B] 将使用 GPU: ${CUDA_B} (负责后 3 组多人群同步长 Mix 实验)"
echo "📄 所有任务的详细输出将保存在 $LOG_DIR/ 目录下"
echo "========================================================"

# -------------------------------------------------------------------------
# 🟢 [流水线 A] 核心单兵作战战区 (Solo Area)
# -------------------------------------------------------------------------
(
    echo "🟢 [流水线 A] 已启动！"
    
    echo "▶️ [流水线 A] 正在训练 [1/6]: 纯羊毛党靶向重击战"
    python main.py --mode tune --task train_y --model TARNET_Baseline_PureV10 --version y_pure_v10_solo_wool --exp_name exp_v10_solo_wool --cuda $CUDA_A --num_per_gpu $NUM_GPU > $LOG_DIR/log_v10_solo_wool.txt 2>&1
    
    echo "▶️ [流水线 A] 正在训练 [2/6]: 纯隐藏金子绝对拯救战"
    python main.py --mode tune --task train_y --model TARNET_Baseline_PureV10 --version y_pure_v10_solo_gold --exp_name exp_v10_solo_gold --cuda $CUDA_A --num_per_gpu $NUM_GPU > $LOG_DIR/log_v10_solo_gold.txt 2>&1
    
    echo "▶️ [流水线 A] 正在训练 [3/6]: 纯自然进店镜像拦截战"
    python main.py --mode tune --task train_y --model TARNET_Baseline_PureV10 --version y_pure_v10_solo_walkin --exp_name exp_v10_solo_walkin --cuda $CUDA_A --num_per_gpu $NUM_GPU > $LOG_DIR/log_v10_solo_walkin.txt 2>&1

    echo "▶️ [流水线 A] 正在训练 [1/6]: 纯羊毛党靶向重击战"
    python main.py --mode tune --task train_y --model TARNET_Baseline_PureV10 --version y_pure_v10_mix_both_same_alpha_head --exp_name exp_y_pure_v10_mix_both_same_alpha_head --cuda $CUDA_A --num_per_gpu $NUM_GPU > $LOG_DIR/log_y_pure_v10_mix_both_same_alpha_head.txt 2>&1
    

    echo "✅ [流水线 A] 所有单人群 Solo 靶向消融实验全部收网！"
) &

# -------------------------------------------------------------------------
# 🔵 [流水线 B] 混合多兵种群战区 (Mix Area)
# -------------------------------------------------------------------------
(
    echo "🔵 [流水线 B] 已启动！"
    
    echo "▶️ [流水线 B] 正在训练 [4/6]: 三人群(All)共享同一爆破参数联动战"
    python main.py --mode tune --task train_y --model TARNET_Baseline_PureV10 --version y_pure_v10_mix_all_same_alpha --exp_name exp_v10_mix_all --cuda $CUDA_B --num_per_gpu $NUM_GPU_B > $LOG_DIR/log_v10_mix_all.txt 2>&1
    
    echo "▶️ [流水线 B] 正在训练 [5/6]: 经典双向对称对齐战 (Both 共享老机制参数)"
    python main.py --mode tune --task train_y --model TARNET_Baseline_PureV10 --version y_pure_v10_mix_both_same_alpha --exp_name exp_v10_mix_both --cuda $CUDA_B --num_per_gpu $NUM_GPU_B > $LOG_DIR/log_v10_mix_both.txt 2>&1
    
    echo "▶️ [流水线 B] 正在训练 [6/6]: 特定双向组合战 (Wool + Walkin 共享同一参数)"
    python main.py --mode tune --task train_y --model TARNET_Baseline_PureV10 --version y_pure_v10_mix_wool_walkin_same_alpha --exp_name exp_v10_mix_wool_walkin --cuda $CUDA_B --num_per_gpu $NUM_GPU_B > $LOG_DIR/log_v10_mix_wool_walkin.txt 2>&1

    echo "✅ [流水线 B] 所有多人群同参数联动消融实验全部收网！"
) &

# 强行阻断等待两路大军合流
wait

echo "========================================================"
echo "🎉 终极捷报：大盘 6 组纯 V10 精细化一维消融试验全部平稳落幕！"
echo "========================================================"