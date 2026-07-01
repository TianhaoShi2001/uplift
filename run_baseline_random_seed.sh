#!/bin/bash

# ========================================================
# 🚀 Uplift 极限榨取计划：动态滑动队列并发 (Sliding Window)
# 策略：空出一个补一个，时刻保持指定卡满载，告别显存闲置！
# ========================================================

ulimit -n 65536

# ---------------------------------------------------------
# 🛠️ 全局环境与【差异化并发容量】配置
# ---------------------------------------------------------
NUM_GPU=0.05 

# GPU A 的配置 (假设显存小)
CUDA_A="1"
MAX_JOBS_A=5  # 👈 动态队列容量：这块卡最多同时跑 15 个

# GPU B 的配置 (假设显存大)
CUDA_B="6"
MAX_JOBS_B=5  # 👈 动态队列容量：这块卡最多同时跑 20 个

PROJECT_ROOT="/NAS/shith/uplift"
DATASET="criteo"
SEEDS=(1042 2042 3042 4042 5042)

LOG_DIR="reproduce_dynamic_logs"
mkdir -p $LOG_DIR

echo "========================================================"
echo "🚀 启动 Uplift 动态队列并发复现 (Seed: 1042~5042)"
echo "▶️ [GPU ${CUDA_A}] 队列容量: ${MAX_JOBS_A} 并发进程"
echo "▶️ [GPU ${CUDA_B}] 队列容量: ${MAX_JOBS_B} 并发进程"
echo "📄 日志输出目录: $LOG_DIR/"
echo "========================================================"

# ---------------------------------------------------------
# 📦 模型清单拆分
# ---------------------------------------------------------
MODELS_A=(
    # "MTMT:y_mtmt_mlp:mtmt_mlp_nas_20260409"
    # "MOTTO:y_motto:motto_kdd25_official_20260409"
    "ECUP:y_ecup:run_1_65536"
    "DragonNet:y_baseline_dragonnet:run_dragonnet"
    # "TARNET:y_v1_base:run_v1_base"
    # "TARNET:y_v3_moe:run_v3_moe"
    # "TARNET:y_v6_res_moe:run_v6_res_moe"
    # "TARNET:y_v4_loss_strata:run_v4_strata"
    # "TARNET:y_v7_soft_top5:run_v7_soft_top5"
)

MODELS_B=(
    # "TARNET:y_v7_soft_top10:run_v7_soft_top10"
    # "TARNET:y_v8_s4:run_v8_s4"
    # "TARNET:y_v8_s5:run_v8_s5"
    "TARNET:y_v8_s6:run_v8_s6"
    "TARNET:y_v10_conflict_both:run_v10_both"
    "TARNET:y_v8_s6_temp10.0:run_y_v8_s6_temp10.0"
    # "TARNET:y_v8_s6_temp20.0:run_y_v8_s6_temp20.0"
    # "S_Learner:y_baseline_s_learner:run_s_learner"
    # "T_Learner:y_baseline_t_learner:run_t_learner"
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
                --num_per_gpu $NUM_GPU > "$LOG_DIR/log_rep_${VERSION}_seed_${SEED}.txt" 2>&1 &
            
            echo "   -> 🚀 [GPU $GPU_ID] 成功投递任务 | 队列状态: $(jobs -pr | wc -l)/${MAX_CONCURRENCY} | 模型: ${MODEL} (Seed: ${SEED})"

            # 2. 🌟 核心控制阀门：如果当前子 shell 的后台进程数达到上限，就死循环等待！
            # jobs -pr 会列出当前运行的后台进程的 PID，wc -l 统计数量
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
# 🚀 启动双队列独立运行
# ---------------------------------------------------------
# 开启流水线 A (限制 MAX_JOBS_A 个并发，挂入后台)
(
    run_dynamic_queue "$CUDA_A" "$MAX_JOBS_A" "${MODELS_A[@]}"
) &

# 开启流水线 B (限制 MAX_JOBS_B 个并发，挂入后台)
(
    run_dynamic_queue "$CUDA_B" "$MAX_JOBS_B" "${MODELS_B[@]}"
) &

# 挂起终端，等待两大集群全部完工
wait

echo "========================================================"
echo "🎉 终极捷报：双卡动态滑动队列已全部清空！效率最大化达成！"
echo "========================================================"