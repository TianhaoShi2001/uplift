#!/bin/bash
# =========================================================================
# 🚀 Uplift 免 Ray 纯 Python 参数化全局动态调度器 (Master Controller - 物理隔离版)
# =========================================================================
ulimit -n 65536
LOG_DIR="no_ray_tune_matrix_logs"
mkdir -p $LOG_DIR

SEEDS=(42 1042)

TASKS_PURE_V10=(
    "y_pure_v10_h32_a1.0_wd1e4"
    "y_pure_v10_h16_a5.0_wd1e5"
    "y_pure_v10_h64_32_a0.5_wd1e4"
    "y_pure_v10_h16_a1.0_wd1e4"
    "y_pure_v10_hNone_a0.5_5.0_1.0_wd1e5"
    "y_pure_v10_hNone_a0.5_10.0_0.5_wd1e5"
    "y_pure_v10_hNone_a0.5_0.5_0.1_wd1e5"
    "y_pure_v10_hNone_a0.5_0.5_5.0_wd1e5"
)

TASKS_OURS_S6=(
    "y_ours_v8s6_h32_a10.0_t1_wd0.01"
    "y_ours_v8s6_h32_a0.1_t20_wd1e5"
    "y_ours_v8s6_h32_a0.5_t20_wd1e5"
    "y_ours_v8s6_h32_a10.0_t20_wd0.01"
    "y_ours_v8s6_hNone_a1.0_t20_wd1e5"
    "y_ours_v8s6_hNone_a1.0_t1_wd0.001"
    "y_ours_v8s6_h16_a0.1_t1_wd1e5"
    "y_ours_v8s6_h32_16_a5.0_t1_wd0.001"
    "y_ours_v8s6_h16_a10.0_t1_wd0.01"
)

echo "========================================================"
echo "🚀 启动 17 组纯 Python 免 Ray 矩阵消融大盘"
echo "📄 过程重定向日志存放在: $LOG_DIR/"
echo "========================================================"

run_no_ray_dispatcher() {
    local GPU_ID=$1
    local MAX_CONCURRENCY=$2
    shift 2
    local ARR_TASKS=("$@")

    for VERSION_NAME in "${ARR_TASKS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            EXP_NAME="${VERSION_NAME}_mean_seed_${SEED}"
            
            # 免Ray单步单进程强训通道
            python main.py \
                --mode single_grid \
                --task train_y \
                --version "$VERSION_NAME" \
                --exp_name "$EXP_NAME" \
                --seed "$SEED" \
                --cuda "$GPU_ID" > "$LOG_DIR/log_${EXP_NAME}.txt" 2>&1 &
                
            # 🌟 这里的 jobs 统计现在被小括号限制在当前显卡独立沙盒内，精准不越界！
            echo "🚀 [GPU $GPU_ID] 挂载新进程: ${EXP_NAME} | 卡槽占用: $(jobs -pr | wc -l)/${MAX_CONCURRENCY}"
            
            while [ $(jobs -pr | wc -l) -ge "$MAX_CONCURRENCY" ]; do
                sleep 2
            done
        done
    done
    wait
    echo "✅ [GPU $GPU_ID] 所有消融组合顺利出关！"
}

# -------------------------------------------------------------------------
# 🚀 物理集群异构分配大阅兵 (🌟 核心加固：加 () 形成物理隔离的子进程环境)
# -------------------------------------------------------------------------
( run_no_ray_dispatcher "2" 1 "${TASKS_PURE_V10[@]}" )  &

( run_no_ray_dispatcher "4" 1 "${TASKS_OURS_S6[@]}" ) &
( run_no_ray_dispatcher "5" 1 "${TASKS_OURS_S6[@]}" ) &
( run_no_ray_dispatcher "6" 1 "${TASKS_OURS_S6[@]}" ) &

( run_no_ray_dispatcher "7" 2 "${TASKS_PURE_V10[@]}" ) &

( run_no_ray_dispatcher "3" 15 "${TASKS_PURE_V10[@]}" ) &
( run_no_ray_dispatcher "0" 12 "${TASKS_OURS_S6[@]}" ) &


wait
echo "========================================================"
echo "🎉 🎉 🎉 终极大捷：大盘参数化消融大盘完全平稳安全收网！"
echo "========================================================"