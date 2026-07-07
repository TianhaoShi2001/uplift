#!/bin/bash

# ========================================================
# 🚀 Uplift 极限榨取计划：EFIN 专属多随机种子动态滑动队列并发
# 策略：空出一个补一个，时刻保持指定卡满载，告别显存闲置！
# ========================================================

ulimit -n 65536

# ---------------------------------------------------------
# 🛠️ 全局环境与【差异化并发容量】配置
# ---------------------------------------------------------
NUM_GPU=0.05 

# GPU A 的配置 (专攻跑 EFIN 这一条学术/基线大队列)
CUDA_A="0"
MAX_JOBS_A=5  

# GPU B 的配置 (闲置或分流，目前全量集中在 A)
CUDA_B="6"
MAX_JOBS_B=5  

PROJECT_ROOT="/NAS/shith/uplift"
DATASET="criteo"
SEEDS=(1042 2042 3042 4042 5042)

LOG_DIR="reproduce_dynamic_logs"
mkdir -p $LOG_DIR

echo "========================================================"
echo "🚀 启动 Uplift 动态队列并发复现 (EFIN 专用模式)"
echo "▶️ [GPU ${CUDA_A}] 队列容量: ${MAX_JOBS_A} 并发进程"
echo "📄 日志输出目录: $LOG_DIR/"
echo "========================================================"

# ---------------------------------------------------------
# 📦 模型清单：只锁定 EFIN 进行极限随机种子轰鸣
# ---------------------------------------------------------
MODELS_A=(
    "EFIN:y_baseline_efin:run_efin"
)

MODELS_B=(
    # 目前只专注跑 A 流水线的 EFIN，B 队列留空
)

# ---------------------------------------------------------
# ⚙️ 核心：滑动窗口动态队列控制器
# ---------------------------------------------------------
run_dynamic_queue() {
    local GPU_ID=$1
    local MAX_CONCURRENCY=$2
    shift 2
    local MODELS=("$@")

    if [ ${#MODELS[@]} -eq 0 ]; then
        return
    fi

    echo "🌊 [GPU $GPU_ID] 启动动态滑动队列 (容量: $MAX_CONCURRENCY 进程) ..."

    for ITEM in "${MODELS[@]}"; do
        IFS=":" read -r MODEL VERSION EXP_NAME <<< "${ITEM}"
        CONFIG_PATH="${PROJECT_ROOT}/ckpts/${DATASET}/train_y/${MODEL}/${VERSION}/${EXP_NAME}/best_config.json"
        
        if [ ! -f "$CONFIG_PATH" ]; then
            echo "⚠️ [GPU $GPU_ID] 找不到图纸跳过: $CONFIG_PATH"
            continue
        fi

        for SEED in "${SEEDS[@]}"; do
            
            # 1. 把任务挂入后台运行
            python main.py \
                --mode reproduce \
                --task train_y \
                --model "$MODEL" \
                --version "$VERSION" \
                --exp_name "$EXP_NAME" \
                --config_path "$CONFIG_PATH" \
                --seed "$SEED" \
                --cuda "$GPU_ID" \
                --num_per_gpu $NUM_GPU > "$LOG_DIR/log_rep_${VERSION}_seed_${SEED}.txt" 2>&1 # &
            
            echo "   -> 🚀 [GPU $GPU_ID] 成功投递任务 | 队列状态: $(jobs -pr | wc -l)/${MAX_CONCURRENCY} | 模型: ${MODEL} (Seed: ${SEED})"

            # 2. 核心控制阀门：跑完一个空出来立马补齐下一个
            while [ $(jobs -pr | wc -l) -ge "$MAX_CONCURRENCY" ]; do
                sleep 1  
            done
            
        done
    done
    
    echo "⏳ [GPU $GPU_ID] 所有任务已投递完毕，等待队列中最后几名选手冲线..."
    wait
    echo "✅ [GPU $GPU_ID] 队列清空，完美下班！"
}

# ---------------------------------------------------------
# 🚀 启动独立运行
# ---------------------------------------------------------
(
    run_dynamic_queue "$CUDA_A" "$MAX_JOBS_A" "${MODELS_A[@]}"
) &

wait

echo "========================================================"
echo "🎉 终极捷报：EFIN 多随机种子动态滑动队列已全部安全清空！"
echo "========================================================"