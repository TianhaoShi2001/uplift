#!/bin/bash
ulimit -n 65536

EXP_DATE=$(date +"%Y%m%d")
LOG_DIR="ecup_logs_${EXP_DATE}"
mkdir -p ${LOG_DIR}
CUDA="0"
echo "========================================================"
echo "🚀 启动 ECUP (Entire Chain Uplift) 论文复现验证"
echo "🔍 核心目标：验证全链路建模 (CTCVR=CTR*CVR) 是否解决 Chain-bias"
echo "========================================================"

python main.py \
    --mode eval \
    --task train_y \
    --model ECUP \
    --version y_ecup \
    --exp_name run_1_65536 \
    --cuda "${CUDA}" \
    --num_per_gpu 0.1
    # > ${LOG_DIR}/log_ecup.txt 2>&1 &
# "ecup_official_${EXP_DATE}" \
echo "✅ ECUP 训练流已挂入后台！"
echo "📊 请通过 'tail -f ${LOG_DIR}/log_ecup.txt' 实时观察进度。"