#!/bin/bash
ulimit -n 65536
# export CUDA_VISIBLE_DEVICES="1"

EXP_DATE=$(date +"%Y%m%d")
LOG_DIR="motto_logs_${EXP_DATE}"
mkdir -p ${LOG_DIR}
CUDA="0,3"
echo "========================================================"
echo "🚀 启动 MOTTO (KDD 2025) 混合专家多目标对齐模型"
echo "🔍 核心目标：验证 SDA (选择性分布对齐) 是否能解决强混淆偏差"
echo "========================================================"

python main.py \
    --mode eval \
    --task train_y \
    --model MOTTO \
    --version y_motto \
    --exp_name "motto_kdd25_official_${EXP_DATE}" \
    --num_per_gpu 0.2 \
    --cuda "${CUDA}" \
    # --project_root . \
    # > ${LOG_DIR}/log_motto.txt 2>&1 

echo "✅ MOTTO 训练流已挂入后台！"
echo "📊 请通过 'tail -f ${LOG_DIR}/log_motto.txt' 实时观察进度。"