#!/bin/bash

# ========================================================
# 🚀 V8 vs V11: Causal MoE 融合机制 2x4 消融矩阵
# ========================================================

ulimit -n 65536
NUM_GPU="0.5"
CUDA_A="1" # V8 流水线显卡
CUDA_B="2" # V11 流水线显卡
C_CKPT_PATH="/NAS/shith/uplift/ckpts/criteo/train_c/TARNET/c_v1_base/exp_c_explore/best_model.pth"
LOG_DIR="ablation_logs"
mkdir -p $LOG_DIR

echo "========================================================"
echo "🚀 启动 8 组消融实验矩阵"
echo "========================================================"

# ---------------------------------------------------------
# 🟣 流水线 A: V8 系列 (原始概率空间)
# ---------------------------------------------------------
# (
#     echo "▶️ [A] 开始: V8_S4 (独立 MLP 乘法)..."
#     python main.py --mode tune --task train_y --model TARNET --version y_v8_s4 --exp_name run_v8_s4 --c_ckpt_path $C_CKPT_PATH --cuda $CUDA_A --num_per_gpu $NUM_GPU # > $LOG_DIR/log_v8_s4.txt 2>&1
    
#     echo "▶️ [A] 开始: V8_S6 (Logit 加法)..."
#     python main.py --mode tune --task train_y --model TARNET --version y_v8_s6 --exp_name run_v8_s6 --c_ckpt_path $C_CKPT_PATH --cuda $CUDA_A --num_per_gpu $NUM_GPU > $LOG_DIR/log_v8_s6.txt 2>&1
    
#     echo "▶️ [A] 开始: V8_S7 (动态信任插值)..."
#     python main.py --mode tune --task train_y --model TARNET --version y_v8_s7 --exp_name run_v8_s7 --c_ckpt_path $C_CKPT_PATH --cuda $CUDA_A --num_per_gpu $NUM_GPU > $LOG_DIR/log_v8_s7.txt 2>&1

#     echo "▶️ [A] 开始: V8_S8 (先验感知 + 乘法)..."
#     python main.py --mode tune --task train_y --model TARNET --version y_v8_s8 --exp_name run_v8_s8 --c_ckpt_path $C_CKPT_PATH --cuda $CUDA_A --num_per_gpu $NUM_GPU > $LOG_DIR/log_v8_s8.txt 2>&1
    
#     echo "✅ [A] V8 矩阵全部完成！"
# ) 

# ---------------------------------------------------------
# 🌟 流水线 B: V11 系列 (EMA Logit 动态空间)
# ---------------------------------------------------------
(
    echo "▶️ [B] 开始: V11_S4 (独立 MLP 乘法)..."
    python main.py --mode tune --task train_y --model TARNET --version y_v11_s4 --exp_name run_v11_s4 --c_ckpt_path $C_CKPT_PATH --cuda $CUDA_B --num_per_gpu $NUM_GPU > $LOG_DIR/log_v11_s4.txt 2>&1 &
    
    echo "▶️ [B] 开始: V11_S6 (Logit 加法 - 核心战力)..."
    python main.py --mode tune --task train_y --model TARNET --version y_v11_s6 --exp_name run_v11_s6 --c_ckpt_path $C_CKPT_PATH --cuda $CUDA_B --num_per_gpu $NUM_GPU > $LOG_DIR/log_v11_s6.txt 2>&1 &
    
    echo "▶️ [B] 开始: V11_S7 (动态信任插值)..."
    python main.py --mode tune --task train_y --model TARNET --version y_v11_s7 --exp_name run_v11_s7 --c_ckpt_path $C_CKPT_PATH --cuda $CUDA_B --num_per_gpu $NUM_GPU > $LOG_DIR/log_v11_s7.txt 2>&1 &
 
    echo "▶️ [B] 开始: V11_S8 (先验感知 + 乘法)..."
    python main.py --mode tune --task train_y --model TARNET --version y_v11_s8 --exp_name run_v11_s8 --c_ckpt_path $C_CKPT_PATH --cuda $CUDA_B --num_per_gpu $NUM_GPU > $LOG_DIR/log_v11_s8.txt 2>&1 &
    
    echo "✅ [B] V11 矩阵全部完成！"
) 

wait
echo "🎉 完美收工！所有 8 个变体执行完毕。"