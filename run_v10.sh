#!/bin/bash

# ========================================================
# 🚀 终极杀器: V7(截断门控) + V10(先验冲突) 联合消融实验
# ========================================================

ulimit -n 65536

# 每张卡分配的资源比例
NUM_GPU="0.5"

# 分配给 3 条流水线的物理显卡 ID
CUDA_A="0,6"
CUDA_B="0,6"
CUDA_C="0,6"

# 最优 C 模型的绝对路径
C_CKPT_PATH="/NAS/shith/uplift/ckpts/criteo/train_c/TARNET/c_v1_base/exp_c_explore/best_model.pth"

mkdir -p v7_conflict_logs

echo "========================================================"
echo "🚀 启动 [V7 结构 + V10 Loss] 联合兵种流水线"
echo "▶️ 流水线 A (V7 + 仅罚羊毛党) 使用 CUDA: ${CUDA_A}"
echo "▶️ 流水线 B (V7 + 仅奖隐藏金子) 使用 CUDA: ${CUDA_B}"
echo "▶️ 流水线 C (V7 + 双向全面纠偏) 使用 CUDA: ${CUDA_C} [🌟 论文终极方案预选]"
echo "========================================================"

# ---------------------------------------------------------
# 🔴 流水线 A: V7 + 仅惩罚羊毛党 (wool_only)
# ---------------------------------------------------------
(
    echo "▶️ [流水线 A] 启动 V7 + wool_only 搜索..."
    python main.py --mode tune --task train_y --model TARNET --version y_v7_conflict_wool --exp_name run_v7_conflict_wool --c_ckpt_path "$C_CKPT_PATH" --cuda "$CUDA_A" --num_per_gpu "$NUM_GPU" > v7_conflict_logs/log_v7_wool.txt 2>&1
    echo "✅ [流水线 A] V7 + wool_only 跑完！"
) &

# ---------------------------------------------------------
# 🟡 流水线 B: V7 + 仅奖励隐藏金子 (gold_only)
# ---------------------------------------------------------
(
    echo "▶️ [流水线 B] 启动 V7 + gold_only 搜索..."
    python main.py --mode tune --task train_y --model TARNET --version y_v7_conflict_gold --exp_name run_v7_conflict_gold --c_ckpt_path "$C_CKPT_PATH" --cuda "$CUDA_B" --num_per_gpu "$NUM_GPU" > v7_conflict_logs/log_v7_gold.txt 2>&1
    echo "✅ [流水线 B] V7 + gold_only 跑完！"
) &

# ---------------------------------------------------------
# 🟢 流水线 C: V7 + 双向全面纠偏 (both)
# ---------------------------------------------------------
(
    echo "▶️ [流水线 C] 启动 V7 + both 双向搜索..."
    python main.py --mode tune --task train_y --model TARNET --version y_v7_conflict_both --exp_name run_v7_conflict_both --c_ckpt_path "$C_CKPT_PATH" --cuda "$CUDA_C" --num_per_gpu "$NUM_GPU" > v7_conflict_logs/log_v7_both.txt 2>&1
    echo "✅ [流水线 C] V7 + both 双向纠偏跑完！"
) &

echo "⏳ 3 条 [结构+优化] 流水线已挂入后台！"
wait
echo "🎉 V7 + V10 联合缝合怪消融实验全部结束！"



# #!/bin/bash

# # ========================================================
# # 🚀 V10 演进架构: Prior-Conflict 机制拆解消融实验
# # ========================================================

# ulimit -n 65536

# # 每张卡分配的资源比例
# NUM_GPU="0.5"

# # 分配给 3 条流水线的物理显卡 ID
# CUDA_A="5,6"
# CUDA_B="5,6"
# CUDA_C="5,6"

# # 最优 C 模型的绝对路径
# C_CKPT_PATH="/NAS/shith/uplift/ckpts/criteo/train_c/TARNET/c_v1_base/exp_c_explore/best_model.pth"

# mkdir -p conflict_ablation_logs

# echo "========================================================"
# echo "🚀 启动先验冲突敏感加权流水线 (共 9 组超参试验)"
# echo "▶️ 流水线 A (仅罚羊毛党) 使用 CUDA: ${CUDA_A}"
# echo "▶️ 流水线 B (仅奖隐藏金子) 使用 CUDA: ${CUDA_B}"
# echo "▶️ 流水线 C (双管齐下版) 使用 CUDA: ${CUDA_C} [🌟 理论最强]"
# echo "========================================================"

# # ---------------------------------------------------------
# # 🔴 流水线 A: 仅惩罚羊毛党 (wool_only)
# # ---------------------------------------------------------
# (
#     echo "▶️ [流水线 A] 启动 wool_only 搜索..."
#     python main.py --mode tune --task train_y --model TARNET --version y_v10_conflict_wool --exp_name run_v10_wool --c_ckpt_path "$C_CKPT_PATH" --cuda "$CUDA_A" --num_per_gpu "$NUM_GPU" > conflict_ablation_logs/log_v10_wool.txt 2>&1
#     echo "✅ [流水线 A] wool_only 跑完！"
# ) 

# # ---------------------------------------------------------
# # 🟡 流水线 B: 仅奖励隐藏金子 (gold_only)
# # ---------------------------------------------------------
# (
#     echo "▶️ [流水线 B] 启动 gold_only 搜索..."
#     python main.py --mode tune --task train_y --model TARNET --version y_v10_conflict_gold --exp_name run_v10_gold --c_ckpt_path "$C_CKPT_PATH" --cuda "$CUDA_B" --num_per_gpu "$NUM_GPU" > conflict_ablation_logs/log_v10_gold.txt 2>&1
#     echo "✅ [流水线 B] gold_only 跑完！"
# ) 

# # ---------------------------------------------------------
# # 🟢 流水线 C: 双向全面纠偏 (both)
# # ---------------------------------------------------------
# (
#     echo "▶️ [流水线 C] 启动 both 双向搜索..."
#     python main.py --mode tune --task train_y --model TARNET --version y_v10_conflict_both --exp_name run_v10_both --c_ckpt_path "$C_CKPT_PATH" --cuda "$CUDA_C" --num_per_gpu "$NUM_GPU" > conflict_ablation_logs/log_v10_both.txt 2>&1
#     echo "✅ [流水线 C] both 双向纠偏跑完！"
# ) 

# echo "⏳ 3 条流水线已挂入后台！"
# wait
# echo "🎉 V10 机制拆解消融实验全部结束！"