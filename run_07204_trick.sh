#!/bin/bash
# ========================================================
# 🚀 Uplift v8_s6 消融 (RNC/C/OHEM) Top 12 多 Seed 并发验证
# 策略：12组配置均分 2/6/3 号卡，保持每卡 5 并发！
# ========================================================

ulimit -n 65536
MAX_JOBS=8

LOG_DIR="no_ray_v8_top12_logs"
mkdir -p $LOG_DIR
C_CKPT_PATH="/NAS/shith/uplift/ckpts/criteo/train_c/TARNET/c_v1_base/exp_c_explore/best_model.pth"
SEEDS=(4042)

echo "========================================================"
echo "🚀 启动 v8_s6 消融 Top 12 (4-Seed) 并发验证大盘"
echo "📄 日志输出目录: $LOG_DIR/"
echo "========================================================"

MODELS_GPU2=(
    "v8_rnc_a5_wd0.01_clip8.0"
    "v8_rnc_a10_wd0.01_clip8.0"
    "v8_rnc_a5_wd0.01_clip5.0"
    "v8_rnc_a10_wd0.001_clip3.0"
)

MODELS_GPU6=(
    "v8_c_a5_wd0.01_clip8.0"
    "v8_c_a5_wd0.01_clip5.0"
    "v8_c_a10_wd0.01_clip8.0"
    "v8_c_a10_wd0.001_clip3.0"
)

MODELS_GPU3=(
    "v8_ohem_a5_wd0.01_pct0.9"
    "v8_ohem_a5_wd0.01_pct0.8"
    "v8_ohem_a5_wd0.01_pct0.7"
    "v8_ohem_a5_wd0.01_pct0.6"
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
                --c_ckpt_path "$C_CKPT_PATH" --cuda "$GPU_ID" > "$LOG_DIR/log_${EXP_NAME}.txt" 2>&1 &
            
            echo "   -> 🚀 [GPU $GPU_ID] 已投递 | 队列: $(jobs -pr | wc -l)/${MAX_CONCURRENCY} | ${EXP_NAME}"

            while [ $(jobs -pr | wc -l) -ge "$MAX_CONCURRENCY" ]; do
                sleep 1
            done
            
        done
    done
    wait
    echo "✅ [GPU $GPU_ID] 当前流水线收网下班！"
}

# 并行拉起三张卡
( run_dynamic_queue "3" "$MAX_JOBS" "${MODELS_GPU2[@]}" ) &
( run_dynamic_queue "3" "$MAX_JOBS" "${MODELS_GPU6[@]}" ) &
( run_dynamic_queue "6" "$MAX_JOBS" "${MODELS_GPU3[@]}" ) &

wait

echo "========================================================"
echo "🎉 终极捷报：v8_s6 消融实验 Top 12 (4-Seed) 已全部投递并完成！"
echo "========================================================"