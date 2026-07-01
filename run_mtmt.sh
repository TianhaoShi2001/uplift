#!/bin/bash

# ========================================================
# 🚀 MTMT 论文复现: MLP 搜索 vs. 真·ResNet18 架构对决
# ========================================================

# 提升文件句柄限制，防止 Ray 训练中途崩溃
ulimit -n 65536

# 每张卡分配的资源比例 (MTMT 模型较大，建议每张卡跑 1~2 个 Trial)
NUM_GPU="0.12" 

# 分配物理显卡 ID (请根据实际服务器空闲情况修改)
CUDA_MLP="3,6"
CUDA_RESNET="3"

# 实验名称前缀
EXP_DATE=$(date +"%Y%m%d")
LOG_DIR="mtmt_study_logs_${EXP_DATE}"
mkdir -p ${LOG_DIR}

echo "========================================================"
echo "🚀 启动 MTMT (STMT-MultiTask) 跨架构对比流水线"
echo "▶️ 流水线 A (MLP 架构搜索) 使用 CUDA: ${CUDA_MLP}"
echo "▶️ 流水线 B (ResNet18 官方架构) 使用 CUDA: ${CUDA_RESNET}"
echo "--------------------------------------------------------"
echo "🔍 核心目标：验证 MMoE 专家网络使用 1D-ResNet18 是否优于深度 MLP"
echo "========================================================"

# ---------------------------------------------------------
# 🔵 流水线 A: MTMT + MLP 灵活架构搜索 (y_mtmt_mlp)
# ---------------------------------------------------------
(
    echo "▶️ [流水线 A] 启动 MTMT-MLP 架构搜索..."
    # 搜索空间：[128,64,32], [64,32], [64]
    python main.py \
        --mode tune \
        --task train_y \
        --model MTMT \
        --version y_mtmt_mlp \
        --exp_name "mtmt_mlp_nas_${EXP_DATE}" \
        --cuda "${CUDA_MLP}" \
        --num_per_gpu "${NUM_GPU}"
        > ${LOG_DIR}/log_mtmt_mlp.txt 2>&1
    echo "✅ [流水线 A] MTMT-MLP 搜索任务圆满结束！"
) # &

# ---------------------------------------------------------
# 🟣 流水线 B: MTMT + ResNet18 官方复刻架构 (y_mtmt_resnet)
# ---------------------------------------------------------
# (
#     echo "▶️ [流水线 B] 启动 MTMT-ResNet18 (1D版) 验证..."
#     # 彻底遵循论文: ResNet18 backbone [cite: 188] + 4 Experts 
#     python main.py \
#         --mode tune \
#         --task train_y \
#         --model MTMT \
#         --version y_mtmt_resnet \
#         --exp_name "mtmt_resnet18_official_${EXP_DATE}" \
#         --cuda "${CUDA_RESNET}" \
#         --num_per_gpu "${NUM_GPU}"
#         # > ${LOG_DIR}/log_mtmt_resnet.txt 2>&1
#     echo "✅ [流水线 B] MTMT-ResNet18 官方架构复现跑完！"
# ) # &

echo "⏳ 2 条 MTMT 架构对比流水线已挂入后台并发执行..."
echo "📊 请通过 'tail -f ${LOG_DIR}/log_mtmt_mlp.txt' 实时观察 MLP 动态架构表现。"
echo "📊 请通过 'tail -f ${LOG_DIR}/log_mtmt_resnet.txt' 观察 ResNet18 是否产生论文所述的优势。"

wait

echo "🎉 [MTMT Study] 全链路对比实验全部结束！"
echo "📈 结果分析建议：请前往 results 目录对比 main_task (Y) 和 aux_task (C) 的 AUUC 指标。"