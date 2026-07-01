#!/bin/bash
CUDA_ID="0"
DATA="criteo"
MODEL="TARNET"
EXP="exp_c_explore"
MODE="tune" # 记得先改成 debug 试水，没问题再改回 tune

echo "========== 启动 C 模型大比武 =========="

# 1. 基础 C 模型
# python main.py --num_per_gpu 0.33 --mode $MODE --task train_c --data_name $DATA --model $MODEL --version c_v1_base --exp_name $EXP --cuda $CUDA_ID # &

# # 2. 引入 Focal Loss
# python main.py --num_per_gpu 0.33 --mode $MODE --task train_c --data_name $DATA --model $MODEL --version c_v2_focal --exp_name $EXP --cuda $CUDA_ID &

# # # 3. 引入 Focal + MMD 对齐
python main.py --num_per_gpu 1 --mode $MODE --task train_c --data_name $DATA --model $MODEL --version c_v3_swd --exp_name $EXP --cuda $CUDA_ID # &

echo "🎉 C 模型探索完毕！请去 ./results 目录看哪个 version 的 AUC 和 PCOC 最高！"