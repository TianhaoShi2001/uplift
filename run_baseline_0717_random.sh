#!/bin/bash
# ========================================================
# 🚀 Uplift Baseline 极限榨取计划：动态滑动队列并发 (Sliding Window)
# 策略：按你最稳的写法，任务拆分静态入场，空出一个补一个！告别花里胡哨的锁！
# ========================================================

ulimit -n 65536

# ---------------------------------------------------------
# 🛠️ 显卡与队列配置 (按你的要求：2、6、3 卡，每卡 5 并发)
# ---------------------------------------------------------
MAX_JOBS=5

LOG_DIR="no_ray_baseline_logs"
mkdir -p $LOG_DIR

SEEDS=(42 1042 2042 3042)

echo "========================================================"
echo "🚀 启动 10 大 Baseline (4-Seed) 动态队列并发大盘"
echo "▶️ [GPU 2, 6, 3] 队列容量: ${MAX_JOBS} 并发进程"
echo "📄 日志输出目录: $LOG_DIR/"
echo "========================================================"

# ---------------------------------------------------------
# 📦 任务清单拆分 (10个Baseline，均分给3张卡)
# ---------------------------------------------------------
MODELS_GPU2=(
    "y_baseline_cfrnet_2"
    "y_baseline_dragonnet"
    "y_ecup"
    "y_baseline_euen_academic_ms"
)

MODELS_GPU6=(
    "y_baseline_efin"
    "y_motto"
    "y_mtmt_mlp"
)

MODELS_GPU3=(
    "y_baseline_s_learner"
    "y_baseline_t_learner"
    "y_v1_base"
)

# ---------------------------------------------------------
# ⚙️ 核心：滑动窗口动态队列控制器 (原汁原味的 jobs 拦截)
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
                --cuda "$GPU_ID" > "$LOG_DIR/log_${EXP_NAME}.txt" 2>&1 &
            
            echo "   -> 🚀 [GPU $GPU_ID] 成功投递任务 | 队列状态: $(jobs -pr | wc -l)/${MAX_CONCURRENCY} | 模型: ${VERSION_NAME} (Seed: ${SEED})"

            # 2. 🌟 核心控制阀门：如果当前子 shell 的后台进程数达到上限，就死循环等待！
            while [ $(jobs -pr | wc -l) -ge "$MAX_CONCURRENCY" ]; do
                sleep 1  # 睡 1 秒钟再检查，跑完一个立马跳出循环，补发下一个！
            done
            
        done
    done
    
    # 所有的任务都投递完了，等队列里最后几个收尾
    echo "⏳ [GPU $GPU_ID] 所有任务已投递完毕，等待队列中最后几名选手冲线..."
    wait
    echo "✅ [GPU $GPU_ID] 队列清空，完美下班！"
}

# ---------------------------------------------------------
# 🚀 启动三队列独立运行
# ---------------------------------------------------------

# 开启流水线 1 (挂入后台)
(
    run_dynamic_queue "2" "$MAX_JOBS" "${MODELS_GPU2[@]}"
) &

# 开启流水线 2 (挂入后台)
(
    run_dynamic_queue "6" "$MAX_JOBS" "${MODELS_GPU6[@]}"
) &

# 开启流水线 3 (挂入后台)
(
    run_dynamic_queue "3" "$MAX_JOBS" "${MODELS_GPU3[@]}"
) &

# 挂起终端，等待三大集群全部完工
wait

echo "========================================================"
echo "🎉 终极捷报：三卡动态滑动队列已全部清空！Baseline 稳稳收网！"
echo "========================================================"