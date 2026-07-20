#!/bin/bash
# ========================================================
# 🚀 Uplift 新王炸 Top3 (4-Seed) 动态滑动队列并发测试
# ========================================================

ulimit -n 65536

# ---------------------------------------------------------
# 🛠️ 显卡与队列配置 
# ---------------------------------------------------------
GPU_ID="2"       # 👈 你可以随便换成当前空闲的卡 (比如 0, 2, 6)
MAX_JOBS=8       # 👈 保持 5 个并发

LOG_DIR="no_ray_newtop3_logs"
mkdir -p $LOG_DIR

SEEDS=(42 1042 2042 3042)

echo "========================================================"
echo "🚀 启动新王炸 Top3 (4-Seed) 动态队列并发大盘"
echo "▶️ [GPU $GPU_ID] 队列容量: ${MAX_JOBS} 并发进程"
echo "📄 日志输出目录: $LOG_DIR/"
echo "========================================================"

# ---------------------------------------------------------
# 📦 任务清单 (刚刚解析出来的 3 个配置名)
# ---------------------------------------------------------
MODELS_NEW_TOP3=(
    # "s6_new_top1_auuc9145_hNone_a1.0_t1_wd1e5"
    "s6_new_top2_auuc9105_hNone_a10.0_t1_wd1e4"
    "s6_new_top3_auuc9094_hNone_a0.1_t1_wd1e5"
)

# ---------------------------------------------------------
# ⚙️ 核心：滑动窗口动态队列控制器 (绝对不越界的 jobs 拦截)
# ---------------------------------------------------------
run_dynamic_queue() {
    local TARGET_GPU=$1
    local MAX_CONCURRENCY=$2
    shift 2
    local MODELS=("$@")

    echo "🌊 [GPU $TARGET_GPU] 启动动态滑动队列 (容量: $MAX_CONCURRENCY 进程) ..."

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
                --cuda "$TARGET_GPU" > "$LOG_DIR/log_${EXP_NAME}.txt" 2>&1 &
            
            echo "   -> 🚀 [GPU $TARGET_GPU] 成功投递任务 | 队列状态: $(jobs -pr | wc -l)/${MAX_CONCURRENCY} | 模型: ${VERSION_NAME} (Seed: ${SEED})"

            # 2. 🌟 核心控制阀门：如果当前子 shell 的后台进程数达到上限，就死循环等待！
            while [ $(jobs -pr | wc -l) -ge "$MAX_CONCURRENCY" ]; do
                sleep 1  
            done
            
        done
    done
    
    echo "⏳ [GPU $TARGET_GPU] 所有任务已投递完毕，等待队列中最后几名选手冲线..."
    wait
    echo "✅ [GPU $TARGET_GPU] 队列清空，完美下班！"
}

# ---------------------------------------------------------
# 🚀 启动单流水线运行
# ---------------------------------------------------------
(
    run_dynamic_queue "$GPU_ID" "$MAX_JOBS" "${MODELS_NEW_TOP3[@]}"
) &

wait

echo "========================================================"
echo "🎉 终极捷报：新王炸 Top 3 (4-Seed) 验证已全部投递并跑完！"
echo "========================================================"