#!/bin/bash

# ========================================================
# 🚀 CUDA 5 全量并行白盒数据收割机 (Full Parallel Mode)
# ========================================================

ulimit -n 65536

# ---------------------------------------------------------
# 🛠️ 资源配置 (全量压入 CUDA 5)
# ---------------------------------------------------------
TARGET_CUDA="3"
NUM_GPU="0.1"  # 9个任务共计占用 ~0.9 资源
C_CKPT_PATH="/NAS/shith/uplift/ckpts/criteo/train_c/TARNET/c_v1_base/exp_c_explore/best_model.pth"

# 创建日志文件夹
LOG_DIR="ablation_logs/eval_batch"
mkdir -p "$LOG_DIR"

echo "========================================================"
echo "⚡ 启动 CUDA 5 全并行收割模式"
echo "🎯 目标设备: GPU ${TARGET_CUDA}"
echo "📦 正在并发启动 9 个 Eval 任务..."
echo "📄 日志监控: tail -f ${LOG_DIR}/*.txt"
echo "========================================================"

# 定义执行函数
run_eval() {
    local version=$1
    local exp_name=$2
    echo "▶️ [In Flight] $version ($exp_name)"
    python main.py \
        --mode eval \
        --task train_y \
        --model TARNET \
        --version "$version" \
        --exp_name "$exp_name" \
        --c_ckpt_path "$C_CKPT_PATH" \
        --cuda "$TARGET_CUDA" \
        --num_per_gpu "$NUM_GPU" > "${LOG_DIR}/log_${version}.txt" 2>&1 &
}

# ---------------------------------------------------------
# 🌊 任务阵列投送 (后台并行)
# ---------------------------------------------------------

# # Phase 1: 早期方案
# run_eval "y_v1_base" "run_v1_base"
# run_eval "y_v3_moe" "run_v3_moe"
# run_eval "y_v4_loss_strata" "run_v4_strata"


# run_eval "y_v8" "run_v6_res_moe"
# run_eval "y_v7_soft_top5" "run_v7_soft_top5"

# Phase 3: V8 演进家族
# --- S1 方案: 动态阈值 ---
run_eval "y_v8_s1_t5" "run_v8_s1_t5"
# run_eval "y_v8_s1_t10" "run_v8_s1_t10"
run_eval "y_v8_s1_t50" "run_v8_s1_t50"

# --- S2 方案: 单门控特征驱动 ---
run_eval "y_v8_s2_t5" "run_v8_s2_t5"
run_eval "y_v8_s2_t10" "run_v8_s2_t10"
run_eval "y_v8_s2_t50" "run_v8_s2_t50"

# --- S3 方案: 独立多门控特征驱动 ---
run_eval "y_v8_s3_t5" "run_v8_s3_t5"
run_eval "y_v8_s3_t10" "run_v8_s3_t10"
run_eval "y_v8_s3_t50" "run_v8_s3_t50"

# --- S4 & S5 方案: 纯特征网络 ---
# run_eval "y_v8_s4" "run_v8_s4"
# run_eval "y_v8_s5" "run_v8_s5"

# # # Phase 2: 残差与截断
# run_eval "y_v8" "run_v6_res_moe"
# run_eval "y_v7_soft_top5" "run_v7_soft_top5"

# # Phase 3: V8 演进家族
# run_eval "y_v8_s1_t5" "run_v8_s1_t5"
# run_eval "y_v8_s4" "run_v8_s4"
# run_eval "y_v8_s5" "run_v8_s5"

# # Phase 4: 惩罚 Loss 方案
# run_eval "y_v10_conflict_both" "run_v10_both"

# ---------------------------------------------------------
# 总控台：等待收割完毕
# ---------------------------------------------------------
echo "--------------------------------------------------------"
echo "⏳ 所有任务已投产后台，正在同步提取数据..."
echo "📊 使用 'nvidia-smi -l 1' 观察 CUDA 5 的神经脉冲。"

wait

echo "========================================================"
echo "🎉 任务全线报捷！9个版本的 .csv 和 .npz 已乖乖躺在 results 目录下了。"
echo "========================================================"