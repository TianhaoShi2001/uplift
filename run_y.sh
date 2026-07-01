#!/bin/bash

# ========================================================
# 🚀 V7 排序感知截断 (Rank-Aware Truncated MoE) 并发流水线
# ========================================================

ulimit -n 65536

# ---------------------------------------------------------
# 🛠️ 请在这里填入你的配置
# ---------------------------------------------------------
# 每张卡分配的资源比例 (例如 "0.25")
NUM_GPU="0.5"

# 流水线 A 和 B 使用的显卡 (例如 "0,1" 或 "4")
CUDA_A="0,3"
CUDA_B="0,3"

# 最优 C 模型的绝对路径
C_CKPT_PATH="/NAS/shith/uplift/ckpts/criteo/train_c/TARNET/c_v1_base/exp_c_explore/best_model.pth"
# ---------------------------------------------------------

# 创建日志文件夹
mkdir -p ablation_logs

echo "========================================================"
echo "🚀 启动极致物理隔离并行 (V7 消融实验列队)"
echo "▶️ 流水线 A 使用 CUDA: ${CUDA_A}"
echo "▶️ 流水线 B 使用 CUDA: ${CUDA_B}"
echo "📄 详细运行日志将输出至 ablation_logs/ 目录"
echo "========================================================"


# ---------------------------------------------------------
# 🟡 对照组：TARNET Naive Multitask (Shared Bottom Baseline)
# ---------------------------------------------------------
(
    echo "▶️ [对照组] 开始跑 TARNET_Naive_Multitask Baseline..."
    # 注意这里 --version 传刚才写的 y_v4_naive_mt
    python main.py --mode tune --task train_y --version y_v1_naive_mt_small_c_weight --exp_name run_tarnet_naive_mt --cuda $CUDA_A --num_per_gpu $NUM_GPU > ablation_logs/log_tarnet_naive_small_mt.txt 2>&1
    echo "✅ [对照组] TARNET_Naive_MT 跑完了！"
) &


(
    echo "▶️ [对照组] 开始跑 TARNET_Naive_Multitask Baseline..."
    # 注意这里 --version 传刚才写的 y_v4_naive_mt
    python main.py --mode tune --task train_y --version y_v1_naive_mt_larger_c_weight --exp_name run_tarnet_naive_mt --cuda $CUDA_A --num_per_gpu $NUM_GPU > ablation_logs/log_tarnet_naive_large_mt.txt 2>&1
    echo "✅ [对照组] TARNET_Naive_MT 跑完了！"
) & 


# # ---------------------------------------------------------
# # 🟢 流水线 A: 冲锋 SOTA (Top 5% 系列 + Top 30% 对照组)
# # ---------------------------------------------------------
# (
#     echo "▶️ [流水线 A] 开始跑 V7_Hard_Top5 (理论最强)..."
#     python main.py --mode tune --task train_y --model TARNET --version y_v7_hard_top1 --exp_name run_v7_hard_top1 --c_ckpt_path $C_CKPT_PATH --cuda $CUDA_A --num_per_gpu $NUM_GPU > ablation_logs/log_v7_hard_top1.txt 2>&1
#     echo "✅ [流水线 A] V7_Hard_Top5 跑完了！无缝衔接跑 Soft Top5..."

#     # echo "▶️ [流水线 A] 开始跑 V7_Soft_Top5 (平滑过渡版)..."
#     # python main.py --mode tune --task train_y --model TARNET --version y_v7_soft_top5 --exp_name run_v7_soft_top5 --c_ckpt_path $C_CKPT_PATH --cuda $CUDA_A --num_per_gpu $NUM_GPU > ablation_logs/log_v7_soft_top5.txt 2>&1
#     # echo "✅ [流水线 A] V7_Soft_Top5 跑完了！无缝衔接跑 Top30 对照组..."

#     # echo "▶️ [流水线 A] 开始跑 V7_Hard_Top30 (反证对照组，预期变差)..."
#     # python main.py --mode tune --task train_y --model TARNET --version y_v7_hard_top30 --exp_name run_v7_hard_top30 --c_ckpt_path $C_CKPT_PATH --cuda $CUDA_A --num_per_gpu $NUM_GPU > ablation_logs/log_v7_hard_top30.txt 2>&1
#     # echo "✅ [流水线 A] 任务全部清空，流水线 A 下班！"
# ) &  # <--- 这里的 & 极其重要，让流水线 A 挂在后台

# # ---------------------------------------------------------
# # 🔵 流水线 B: 探底测试 (Top 10% 系列，看边界在哪里)
# # ---------------------------------------------------------
# (
#     echo "▶️ [流水线 B] 开始跑 V7_Hard_Top10 (边界试探)..."
#     python main.py --mode tune --task train_y --model TARNET --version y_v7_soft_top1 --exp_name run_v7_soft_top1 --c_ckpt_path $C_CKPT_PATH --cuda $CUDA_B --num_per_gpu $NUM_GPU > ablation_logs/log_v7_soft_top1.txt 2>&1
#     echo "✅ [流水线 B] V7_Hard_Top10 跑完了！无缝衔接跑 Soft Top10..."

#     # echo "▶️ [流水线 B] 开始跑 V7_Soft_Top10 (平滑探底版)..."
#     # python main.py --mode tune --task train_y --model TARNET --version y_v7_soft_top10 --exp_name run_v7_soft_top10 --c_ckpt_path $C_CKPT_PATH --cuda $CUDA_B --num_per_gpu $NUM_GPU > ablation_logs/log_v7_soft_top10.txt 2>&1
#     # echo "✅ [流水线 B] 任务全部清空，流水线 B 下班！"
# ) &  # <--- 这里的 & 让流水线 B 也挂在后台和 A 并排跑

# ---------------------------------------------------------
# 总控台：等这两条线都干完活
# ---------------------------------------------------------
echo "⏳ 两条 GPU 流水线已全部扔进后台独立运行，互不干扰！"
echo "💡 你可以使用命令 'tail -f ablation_logs/log_v7_hard_top5.txt' 实时查看第一视角的战况。"

wait

echo "========================================================"
echo "🎉 恭喜！双线流水线的 5 个 V7 黄金消融实验已全部打完收工！"