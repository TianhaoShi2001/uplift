#!/bin/bash
# =========================================================================
# 🚀 Uplift 免 Ray 纯 Python 参数化全局动态调度器 (Master Controller - 物理隔离版)
# 🌟 本次大盘新增特征：
# 1. 实验名全系挂载 "TopX_AUUC" 后缀，方便查验稳定性 Gap！
# 2. 注入全新 4 种子大满贯 (42, 1042, 2042, 3042)
# =========================================================================
ulimit -n 65536
LOG_DIR="no_ray_tune_matrix_logs"
mkdir -p $LOG_DIR

# 🌟 4 组大满贯保底随机种子池
SEEDS=(42 1042 2042 3042)

# 🟢 S4 组 (7 个任务)
TASKS_OURS_S4=(
    "s4_top1_auuc9102_hNone_a5.0_wd1e4"
    "s4_top2_auuc9096_hNone_a1.0_wd1e2"
    "s4_top3_auuc9094_h64_32_a0.1_wd1e5"
    "s4_top4_auuc9089_hNone_a5.0_wd1e5"
    "s4_top5_auuc9086_hNone_a1.0_wd1e3"
    "s4_top6_auuc9080_hNone_a5.0_wd1e2"
    "s4_top7_auuc9079_hNone_a10.0_wd1e5"
)

# 🔵 S6 组 (9 个任务)
TASKS_OURS_S6=(
    "s6_top1_auuc9104_hNone_a0.1_t1_wd1e3"
    "s6_top2_auuc9088_hNone_a5.0_t1_wd1e3"
    "s6_top3_auuc9086_hNone_a0.1_t1_wd1e5"
    "s6_top4_auuc9084_hNone_a5.0_t1_wd1e5"
    "s6_top5_auuc9084_hNone_a5.0_t1_wd1e4"
    "s6_top6_auuc9078_hNone_a1.0_t20_wd1e4"
    "s6_top7_auuc9077_hNone_a1.0_t1_wd1e4"
    "s6_top8_auuc9069_hNone_a1.0_t10_wd1e3"
    "s6_top9_auuc9068_hNone_a5.0_t1_wd1e2"
)

# 👑 Pure V10 组 (12 个任务)
TASKS_PURE_V10=(
    "v10_top1_auuc9105_hNone_a0.0_wd1e3"
    "v10_top2_auuc9105_hNone_a0.01_wd1e3"
    "v10_top3_auuc9104_hNone_a0.1_wd1e3"
    "v10_top4_auuc9102_hNone_a0.5_wd1e3"
    "v10_top5_auuc9099_hNone_a0.05_wd1e3"
    "v10_top6_auuc9088_hNone_a0.5_wd1e5"
    "v10_top7_auuc9087_hNone_a0.1_wd1e5"
    "v10_top8_auuc9087_hNone_a5.0_wd1e3"
    "v10_top9_auuc9084_hNone_a0.5_wd1e4"
    "v10_top10_auuc9083_hNone_a10.0_wd1e2"
    "v10_top11_auuc9079_h64_32_a5.0_wd1e2"
    "v10_top12_auuc9077_hNone_a10.0_wd1e5"
)

echo "========================================================"
echo "🚀 启动 S4, S6, V10 巅峰参数带分带排名矩阵挂机消融大盘"
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
                
            # 🌟 小括号局部独立沙盒
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
# 🚀 物理集群异构分配大阅兵 
# -------------------------------------------------------------------------
# 为了不让 0、3 卡把任务全抢了导致其它卡闲置，我们把 3 个任务大数组分散一下

# ( run_no_ray_dispatcher "2" 1 "${TASKS_PURE_V10[@]}" ) &

# ( run_no_ray_dispatcher "4" 1 "${TASKS_OURS_S6[@]}" ) &
# ( run_no_ray_dispatcher "5" 1 "${TASKS_OURS_S6[@]}" ) &
# ( run_no_ray_dispatcher "6" 1 "${TASKS_OURS_S4[@]}" ) &   # 👉 丢给 6 卡跑点 S4

( run_no_ray_dispatcher "7" 1 "${TASKS_PURE_V10[@]}" ) &

# 核爆区：3 卡跑 15 并发，0 卡跑 12 并发
( run_no_ray_dispatcher "3" 20 "${TASKS_PURE_V10[@]}" ) &
( run_no_ray_dispatcher "0" 13 "${TASKS_OURS_S6[@]}" ) &

wait
echo "========================================================"
echo "🎉 🎉 🎉 终极大捷：大盘带分参数化消融 4-seed 完全平稳安全收网！"
echo "========================================================"