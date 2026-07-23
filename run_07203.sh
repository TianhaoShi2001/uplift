#!/bin/bash
# ========================================================
# 🚀 Uplift 15 大高分配置 (4-Seed) 动态滑动队列并发脚本
# 策略：3卡火力全开，每卡承载 5 个配置，稳吃 5 并发
# ========================================================

ulimit -n 65536
MAX_JOBS=5

LOG_DIR="no_ray_top15_logs"
mkdir -p $LOG_DIR

SEEDS=(42 1042 2042 3042 4042)

echo "========================================================"
echo "🚀 启动 Top 15 霸榜组合 (4-Seed) 并发验证大盘"
echo "▶️ [GPU 2, 6, 3] 队列容量: ${MAX_JOBS} 并发进程"
echo "📄 日志输出目录: $LOG_DIR/"
echo "========================================================"
C_CKPT_PATH="/NAS/shith/uplift/ckpts/criteo/train_c/TARNET/c_v1_base/exp_c_explore/best_model.pth"

# ---------------------------------------------------------
# 📦 任务清单拆分 (15个配置，均分给3张卡)
# ---------------------------------------------------------
MODELS_GPU2=(
    "s6_new_top3_auuc9094_hNone_a0.1_t1_wd1e5"
)

MODELS_GPU6=(
    "s6_new_top1_auuc9145_hNone_a1.0_t1_wd1e5"
)

MODELS_GPU3=(
    "s6_new_top2_auuc9105_hNone_a10.0_t1_wd1e4"
)

# ---------------------------------------------------------
# ⚙️ 核心：滑动窗口动态队列控制器
# ---------------------------------------------------------
run_dynamic_queue() {
    local GPU_ID=$1
    local MAX_CONCURRENCY=$2
    shift 2
    local MODELS=("$@")

    echo "🌊 [GPU $GPU_ID] 启动动态滑动队列 (容量: $MAX_CONCURRENCY 进程) ..."

    for VERSION_NAME in "${MODELS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            
            EXP_NAME="${VERSION_NAME}_mean_seed_${SEED}"
            
            # 1. 把任务挂入后台运行
            python main.py \
                --mode single_grid \
                --task train_y \
                --version "$VERSION_NAME" \
                --exp_name "$EXP_NAME" \
                --seed "$SEED" \
                --c_ckpt_path "$C_CKPT_PATH" --cuda "$GPU_ID" > "$LOG_DIR/log_${EXP_NAME}.txt" 2>&1 &
            
            echo "   -> 🚀 [GPU $GPU_ID] 成功投递任务 | 队列状态: $(jobs -pr | wc -l)/${MAX_CONCURRENCY} | 模型: ${VERSION_NAME} (Seed: ${SEED})"

            # 2. 🌟 核心控制阀门：防爆破拦截
            while [ $(jobs -pr | wc -l) -ge "$MAX_CONCURRENCY" ]; do
                sleep 1
            done
            
        done
    done
    
    echo "⏳ [GPU $GPU_ID] 所有任务投递完毕，等待最后冲线..."
    wait
    echo "✅ [GPU $GPU_ID] 队列清空，圆满收网！"
}

# ---------------------------------------------------------
# 🚀 启动三队列独立运行
# ---------------------------------------------------------
(
    run_dynamic_queue "2" "$MAX_JOBS" "${MODELS_GPU2[@]}"
) &

(
    run_dynamic_queue "6" "$MAX_JOBS" "${MODELS_GPU6[@]}"
) &

(
    run_dynamic_queue "3" "$MAX_JOBS" "${MODELS_GPU3[@]}"
) &

wait

echo "========================================================"
echo "🎉 终极捷报：Top 15 高分配置 (4-Seed) 已全部验证完毕！"
echo "========================================================"