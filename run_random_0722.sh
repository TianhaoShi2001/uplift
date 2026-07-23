#!/bin/bash
# ========================================================
# 🚀 EUEN + ECUP + MTMT 终极 Top 8 多 Seed 并发验证
# 策略：所有 8 组配置全部汇聚于 6 号卡，最高 5 并发滚动执行！
# ========================================================

ulimit -n 65536
MAX_JOBS=8
GPU_ID="6"

LOG_DIR="no_ray_all_top8_logs"
mkdir -p $LOG_DIR

SEEDS=(42 1042 2042 3042)

echo "========================================================"
echo "🚀 启动 EUEN + ECUP + MTMT 联合大盘 (4-Seed)"
echo "🔥 目标算力节点: GPU $GPU_ID"
echo "📄 日志输出目录: $LOG_DIR/"
echo "========================================================"

# 把之前的 EUEN 3个 和 ECUP/MTMT 5个 全部塞进 6号卡 任务队列
MODELS_ALL=(
    # "ecup_top1_wd0.01"
    # "ecup_top2_wd1e-05"
    # # --- EUEN_Academic 组 ---
    "euen_top1_h128_64_32_wd1e-05"
    # "mtmt_top1_aux1.0"

    "euen_top2_h128_64_32_wd1e-06"
    # "mtmt_top2_aux0.01"

    "euen_top3_h128_64_32_wd0.0001"
    # "mtmt_top3_aux0.1"

    # # --- ECUP_0717new 组 ---
    # "ecup_top1_wd0.01"
    # "ecup_top2_wd1e-05"
    
    # --- MTMT_0717new 组 --
)

for VERSION_NAME in "${MODELS_ALL[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        
        EXP_NAME="${VERSION_NAME}_seed_${SEED}"
        
        python main.py \
            --mode single_grid \
            --task train_y \
            --version "$VERSION_NAME" \
            --exp_name "$EXP_NAME" \
            --seed "$SEED" \
            --cuda "$GPU_ID" # > "$LOG_DIR/log_${EXP_NAME}.txt" 2>&1 &
        
        echo "   -> 🚀 [GPU $GPU_ID] 已投递 | 队列: $(jobs -pr | wc -l)/${MAX_JOBS} | ${EXP_NAME}"

        # 动态并发控制阀门
        while [ $(jobs -pr | wc -l) -ge "$MAX_JOBS" ]; do
            sleep 1
        done
        
    done
done

# 等待最后一批任务跑完
wait

echo "========================================================"
echo "🎉 终极捷报：所有 8 个冠军配置 (共计 32 个任务) 已在 6号卡 成功跑完！"
echo "========================================================"