#!/bin/bash
# ========================================================
# 🚀 CanniUplift (权重全为1) Y_AUUC Top 5 多 Seed 并发验证
# ========================================================

ulimit -n 65536
MAX_JOBS=5

LOG_DIR="no_ray_canni_w1_top5_logs"
mkdir -p $LOG_DIR

SEEDS=(42 1042 2042 3042)

echo "========================================================"
echo "🚀 启动 CanniUplift [权重全为1的 Top 5] 4-Seed 并发验证"
echo "📄 日志输出目录: $LOG_DIR/"
echo "========================================================"

MODELS_GPU2=(
    "canni_w1_top1_h128_64_wd0.01"
    "canni_w1_top2_h128_64_wd1e-05"
)

MODELS_GPU6=(
    "canni_w1_top3_h128_64_wd0.0001"
    "canni_w1_top4_h128_64_32_wd0.0001"
)

MODELS_GPU3=(
    "canni_w1_top5_h128_64_wd0.001"
)

run_dynamic_queue() {
    local GPU_ID=$1
    local MAX_CONCURRENCY=$2
    shift 2
    local MODELS=("$@")

    for VERSION_NAME in "${MODELS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            
            EXP_NAME="${VERSION_NAME}_seed_${SEED}"
            
            python main.py \
                --mode single_grid \
                --task train_y \
                --version "$VERSION_NAME" \
                --exp_name "$EXP_NAME" \
                --seed "$SEED" \
                --cuda "$GPU_ID" > "$LOG_DIR/log_${EXP_NAME}.txt" 2>&1 &
            
            echo "   -> 🚀 [GPU $GPU_ID] 已投递 | 队列: $(jobs -pr | wc -l)/${MAX_CONCURRENCY} | ${EXP_NAME}"

            while [ $(jobs -pr | wc -l) -ge "$MAX_CONCURRENCY" ]; do
                sleep 1
            done
            
        done
    done
    wait
    echo "✅ [GPU $GPU_ID] 收网下班！"
}

( run_dynamic_queue "2" "$MAX_JOBS" "${MODELS_GPU2[@]}" ) &
( run_dynamic_queue "6" "$MAX_JOBS" "${MODELS_GPU6[@]}" ) &
( run_dynamic_queue "3" "$MAX_JOBS" "${MODELS_GPU3[@]}" ) &

wait

echo "========================================================"
echo "🎉 终极捷报：全为 1 权重组的 Top 5 已全部跑完！"
echo "========================================================"