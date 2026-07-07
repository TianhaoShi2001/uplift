#!/bin/bash

# ========================================================
# 🚀 Uplift Modeling 全宇宙终极自动化流水线 (Master Script)
# 包含：经典因果基线 + 新 V10 (xinv10 三向人群) 独立温度消融矩阵
# ========================================================

ulimit -n 65536

# ---------------------------------------------------------
# 🛠️ 全局资源配置
# ---------------------------------------------------------
NUM_GPU=0.05
CUDA_A="3" 
CUDA_B="1" 

C_CKPT_PATH="/NAS/shith/uplift/ckpts/criteo/train_c/TARNET/c_v1_base/exp_c_explore/best_model.pth"
LOG_DIR="experiment_logs_0415"
mkdir -p $LOG_DIR

echo "========================================================"
echo "🚀 启动 Uplift 全链路双核并发大阅兵"
echo "▶️ [流水线 A] 将使用 GPU: ${CUDA_A} (全卡全力并行)"
echo "▶️ [流水线 B] 将使用 GPU: ${CUDA_B}"
echo "📄 所有任务的详细输出将保存在 $LOG_DIR/ 目录下"
echo "========================================================"

(
echo "🟢 [流水线 A] 已启动！"

    # echo "▶️ [流水线 A] 正在训练: EFIN"
    # python main.py --mode tune --task train_y --model EFIN --version y_baseline_efin --exp_name run_efin --cuda $CUDA_A --num_per_gpu $NUM_GPU > $LOG_DIR/log_efin.txt 2>&1

    # echo "▶️ [流水线 A] 正在训练: DESCN"
    # python main.py --mode tune --task train_y --model DESCN --version y_baseline_descn --exp_name run_descn --cuda $CUDA_A --num_per_gpu $NUM_GPU > $LOG_DIR/log_descn.txt 2>&1

    # echo "▶️ [流水线 A] 正在训练: EUEN_Academic"
    # python main.py --mode tune --task train_y --model EUEN_Academic --version y_baseline_euen_academic_ms --exp_name run_euen_academic --cuda $CUDA_A --num_per_gpu $NUM_GPU > $LOG_DIR/log_euen_academic.txt 2>&1

    echo "▶️ [流水线 A] 正在训练: Ours S4 + Conflict Both (容量: [64])"
    python main.py --mode tune --task train_y --model TARNET --version y_ours_s4_conflict_dim64 --exp_name run_ours_s4_dim64_2 --cuda $CUDA_A --num_per_gpu $NUM_GPU > $LOG_DIR/log_ours_s4_dim64.txt 2>&1

    # echo "▶️ [流水线 A] 正在训练: Ours S4 + Conflict Both (容量: [64, 32])"
    # python main.py --mode tune --task train_y --model TARNET --version y_ours_s4_conflict_dim64_32 --exp_name run_ours_s4_dim64_32 --cuda $CUDA_A --num_per_gpu $NUM_GPU > $LOG_DIR/log_ours_s4_dim64_32.txt 2>&1

    # echo "▶️ [流水线 A] 正在训练: Ours S4 + Conflict Both (容量: [128, 64, 32])"
    # python main.py --mode tune --task train_y --model TARNET --version y_ours_s4_conflict_dim128_64_32 --exp_name run_ours_s4_dim128_64_32 --cuda $CUDA_A --num_per_gpu $NUM_GPU > $LOG_DIR/log_ours_s4_dim128_64_32.txt 2>&1

    # echo "▶️ [流水线 A] 正在训练: 新 V10 + S6 (Temp 10.0)"
    # python main.py --mode tune --task train_y --model TARNET --version y_v10_conflict_all_walkin0705 --exp_name run_ours_v10_all --cuda $CUDA_A --num_per_gpu $NUM_GPU > $LOG_DIR/log_ours_xinv10.txt 2>&1

    # # =========================================================================
    # # 🌟 新 V10 (xinv10) 战区：五档独立解耦温度阵列 (强制锁定最大容量)
    # # =========================================================================
    # echo "▶️ [流水线 A] 正在训练: 新 V10 + S6 (Temp 10.0)"
    # python main.py --mode tune --task train_y --model TARNET --version y_xinv10_conflict_walkin_temp10.0 --exp_name run_ours_xinv10_temp10 --cuda $CUDA_A --num_per_gpu $NUM_GPU > $LOG_DIR/log_ours_xinv10_temp10.txt 2>&1

    # echo "▶️ [流水线 A] 正在训练: 新 V10 + S6 (Temp 20.0)"
    # python main.py --mode tune --task train_y --model TARNET --version y_xinv10_conflict_walkin_temp20.0 --exp_name run_ours_xinv10_temp20 --cuda $CUDA_A --num_per_gpu $NUM_GPU > $LOG_DIR/log_ours_xinv10_temp20.txt 2>&1

    # echo "▶️ [流水线 A] 正在训练: 新 V10 + S6 (Temp 5.0)"
    # python main.py --mode tune --task train_y --model TARNET --version y_xinv10_conflict_walkin_temp5.0 --exp_name run_ours_xinv10_temp5 --cuda $CUDA_A --num_per_gpu $NUM_GPU > $LOG_DIR/log_ours_xinv10_temp5.txt 2>&1

    # echo "▶️ [流水线 A] 正在训练: 新 V10 + S6 (Temp 1.0)"
    # python main.py --mode tune --task train_y --model TARNET --version y_xinv10_conflict_walkin_temp1.0 --exp_name run_ours_xinv10_temp1 --cuda $CUDA_A --num_per_gpu $NUM_GPU > $LOG_DIR/log_ours_xinv10_temp1.txt 2>&1

    # echo "▶️ [流水线 A] 正在训练: 新 V10 + S6 (Temp 50.0)"
    # python main.py --mode tune --task train_y --model TARNET --version y_xinv10_conflict_walkin_temp50.0 --exp_name run_ours_xinv10_temp50 --cuda $CUDA_A --num_per_gpu $NUM_GPU > $LOG_DIR/log_ours_xinv10_temp50.txt 2>&1

    echo "✅ [流水线 A] 基线、拆解版 S4 与全新 xinv10 矩阵并发运行中！"
) 

wait
echo "========================================================"
echo "🎉 终极捷报：大盘所有指定并发调参试验全部平稳落幕！"
echo "========================================================"