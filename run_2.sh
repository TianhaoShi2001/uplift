#!/bin/bash

# ========================================================
# 🚀 V8 演进架构 (Rank-Aware Truncated MoE) 4卡并发流水线
# ========================================================

ulimit -n 65536

# ---------------------------------------------------------
# 🛠️ 请在这里填入你的配置
# ---------------------------------------------------------
# 每张卡分配的资源比例 (Ray Tune 专用)
NUM_GPU="0.1"

# 分配给 4 条流水线的物理显卡 ID
CUDA_A="6"
CUDA_B="6"
CUDA_C="6"
CUDA_D="6"

# 最优 C 模型的绝对路径 (直接复用你的)
C_CKPT_PATH="/NAS/shith/uplift/ckpts/criteo/train_c/TARNET/c_v1_base/exp_c_explore/best_model.pth"
# ---------------------------------------------------------

# 创建日志文件夹
mkdir -p ablation_logs

echo "========================================================"
echo "🚀 启动极致物理隔离并行 (V8 Top5 核心矩阵列队)"
echo "▶️ 流水线 A (方案1) 使用 CUDA: ${CUDA_A}"
echo "▶️ 流水线 B (方案2) 使用 CUDA: ${CUDA_B}"
echo "▶️ 流水线 C (方案3) 使用 CUDA: ${CUDA_C}"
echo "▶️ 流水线 D (方案4/5) 使用 CUDA: ${CUDA_D}"
echo "📄 详细运行日志将输出至 ablation_logs/ 目录"
echo "========================================================"

# ---------------------------------------------------------
# 🟢 流水线 A: V8 方案 1 (动态阈值降分)
# ---------------------------------------------------------
(
    # echo "▶️ [流水线 A] 开始跑 V8 方案1 (Top 5%)..."
    # python main.py --mode tune --task train_y --model TARNET --version y_v8_s1_t5 --exp_name run_v8_s1_t5 --c_ckpt_path "$C_CKPT_PATH" --cuda "$CUDA_A" --num_per_gpu "$NUM_GPU" > ablation_logs/log_v8_s1_t5.txt 2>&1
    # echo "✅ [流水线 A] V8 方案1 (Top 5%) 跑完了！"

    echo "▶️ [流水线 A] 开始跑 V8 方案1 (Top 10%)..."
    python main.py --mode eval --task train_y --model TARNET --version y_v8_s1_t10 --exp_name run_v8_s1_t10 --c_ckpt_path "$C_CKPT_PATH" --cuda "$CUDA_A" --num_per_gpu "$NUM_GPU" # > ablation_logs/log_v8_s1_t10.txt 2>&1
    
    # echo "▶️ [流水线 A] 开始跑 V8 方案1 (Top 50%)..."
    # python main.py --mode tune --task train_y --model TARNET --version y_v8_s1_t50 --exp_name run_v8_s1_t50 --c_ckpt_path "$C_CKPT_PATH" --cuda "$CUDA_A" --num_per_gpu "$NUM_GPU" > ablation_logs/log_v8_s1_t50.txt 2>&1
) 

# # &

# # ---------------------------------------------------------
# # 🔵 流水线 B: V8 方案 2 (单门控特征驱动)
# # ---------------------------------------------------------
# (
#     echo "▶️ [流水线 B] 开始跑 V8 方案2 (Top 5%)..."
#     python main.py --mode tune --task train_y --model TARNET --version y_v8_s2_t5 --exp_name run_v8_s2_t5 --c_ckpt_path "$C_CKPT_PATH" --cuda "$CUDA_B" --num_per_gpu "$NUM_GPU" > ablation_logs/log_v8_s2_t5.txt 2>&1
#     echo "✅ [流水线 B] V8 方案2 (Top 5%) 跑完了！"

#     echo "▶️ [流水线 B] 开始跑 V8 方案2 (Top 10%)..."
#     python main.py --mode tune --task train_y --model TARNET --version y_v8_s2_t10 --exp_name run_v8_s2_t10 --c_ckpt_path "$C_CKPT_PATH" --cuda "$CUDA_B" --num_per_gpu "$NUM_GPU" > ablation_logs/log_v8_s2_t10.txt 2>&1
    
#     echo "▶️ [流水线 B] 开始跑 V8 方案2 (Top 50%)..."
#     python main.py --mode tune --task train_y --model TARNET --version y_v8_s2_t50 --exp_name run_v8_s2_t50 --c_ckpt_path "$C_CKPT_PATH" --cuda "$CUDA_B" --num_per_gpu "$NUM_GPU" > ablation_logs/log_v8_s2_t50.txt 2>&1
# ) &

# # ---------------------------------------------------------
# # 🟡 流水线 C: V8 方案 3 (独立多门控 - 🌟 最推荐)
# # ---------------------------------------------------------
# (
#     echo "▶️ [流水线 C] 开始跑 V8 方案3 (Top 5%)..."
#     python main.py --mode tune --task train_y --model TARNET --version y_v8_s3_t5 --exp_name run_v8_s3_t5 --c_ckpt_path "$C_CKPT_PATH" --cuda "$CUDA_C" --num_per_gpu "$NUM_GPU" > ablation_logs/log_v8_s3_t5.txt 2>&1
#     echo "✅ [流水线 C] V8 方案3 (Top 5%) 跑完了！"

#     echo "▶️ [流水线 C] 开始跑 V8 方案3 (Top 10%)..."
#     python main.py --mode tune --task train_y --model TARNET --version y_v8_s3_t10 --exp_name run_v8_s3_t10 --c_ckpt_path "$C_CKPT_PATH" --cuda "$CUDA_C" --num_per_gpu "$NUM_GPU" > ablation_logs/log_v8_s3_t10.txt 2>&1
    
#     echo "▶️ [流水线 C] 开始跑 V8 方案3 (Top 50%)..."
#     python main.py --mode tune --task train_y --model TARNET --version y_v8_s3_t50 --exp_name run_v8_s3_t50 --c_ckpt_path "$C_CKPT_PATH" --cuda "$CUDA_C" --num_per_gpu "$NUM_GPU" > ablation_logs/log_v8_s3_t50.txt 2>&1
# ) &

# # ---------------------------------------------------------
# # 🟣 流水线 D: V8 方案 4 & 5 (无锚点纯黑盒)
# # ---------------------------------------------------------
# (
#     echo "▶️ [流水线 D] 开始跑 V8 方案4 (纯特征 MLP)..."
#     python main.py --mode tune --task train_y --model TARNET --version y_v8_s4 --exp_name run_v8_s4 --c_ckpt_path "$C_CKPT_PATH" --cuda "$CUDA_D" --num_per_gpu "$NUM_GPU" > ablation_logs/log_v8_s4.txt 2>&1
#     echo "✅ [流水线 D] V8 方案4 跑完了！无缝衔接跑方案 5..."

#     echo "▶️ [流水线 D] 开始跑 V8 方案5 (纯 Softmax MMoE)..."
#     python main.py --mode tune --task train_y --model TARNET --version y_v8_s5 --exp_name run_v8_s5 --c_ckpt_path "$C_CKPT_PATH" --cuda "$CUDA_D" --num_per_gpu "$NUM_GPU" > ablation_logs/log_v8_s5.txt 2>&1
#     echo "✅ [流水线 D] V8 方案5 跑完了！"
# ) &

# # ---------------------------------------------------------
# # 总控台：等待所有后台任务结束
# # ---------------------------------------------------------
# echo "⏳ 4 条 GPU 流水线已全部扔进后台独立运行，互不干扰！"
# echo "💡 建议开 4 个终端窗口，分别执行："
# echo "   tail -f ablation_logs/log_v8_s1_t5.txt"
# echo "   tail -f ablation_logs/log_v8_s2_t5.txt"
# echo "   tail -f ablation_logs/log_v8_s3_t5.txt"
# echo "   tail -f ablation_logs/log_v8_s4.txt"

# wait

# echo "========================================================"
# echo "🎉 恭喜！V8 核心矩阵的 5 个 Top5/无锚点 实验已全部打完收工！"
# echo "========================================================"