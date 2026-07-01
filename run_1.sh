#!/bin/bash

# 创建独立日志目录
mkdir -p logs

echo "🚀 终极狂暴模式：CUDA 2 单卡全量并发！"
echo "📊 配置: 按照 0.2 颗粒度分配 GPU (理论上单卡可同时挂 5 个 Trial)"

# 统一定死使用 CUDA 2
CUDA_ID="6"
NUM_GPU="0.05"

# ==========================================
# ⚔️ 万箭齐发：并发与串行混合挂载
# ==========================================

# echo "⏳ 启动 1: C_v1_base (后台并发)..."
# python main.py --mode eval --task train_c --model TARNET --version c_v1_base --exp_name exp_c_explore --data_name criteo --cuda '0' --num_per_gpu $NUM_GPU > logs/c_v1_base.log 2>&1 &

echo "⏳ 启动 2: C_v2_focal (后台并发)..."
python main.py --mode tune --task train_c --model TARNET --version c_v2_focal --exp_name exp_c_explore --data_name criteo --cuda 6 --num_per_gpu $NUM_GPU > logs/c_v2_focal.log 2>&1 &

# echo "⏳ 启动 3: Y_v1_baseline (后台并发)..."
# python main.py --mode tune --task train_y --model TARNET --version v1_baseline --exp_name exp_y_explore --data_name criteo --cuda '0,1,2,5' --num_per_gpu $NUM_GPU > logs/y_v1_baseline.log 2>&1 &

# echo "⏳ 启动 4: 对齐双雄 SWD & Moment (这俩会在后台排队顺序执行)..."
# (
#     # 先跑 SWD
#     python main.py --mode tune --task train_c --model TARNET --version c_v3_swd --exp_name exp_c_explore --data_name criteo --cuda $CUDA_ID --num_per_gpu $NUM_GPU > logs/c_v3_swd.log 2>&1
    
echo "✅ 紧接着启动 C_v4_moment..."
    
#     # 跑完接着跑 Moment
# echo "⏳ 启动 2: C_v2_focal (后台并发)..."
python main.py --mode tune --task train_c --model TARNET --version c_v4_moment --exp_name exp_c_explore --data_name criteo --cuda 6--num_per_gpu $NUM_GPU > logs/c_v4_moment.log 2>&1
# ) &

echo "🎯 任务已全部投递进 CUDA $CUDA_ID 的后台！"
echo "👀 正在疯狂炼丹中..."
echo "👉 监控基础线: tail -f logs/c_v1_base.log"
echo "👉 监控对齐线: tail -f logs/c_v3_swd.log (跑完会切到 logs/c_v4_moment.log)"

# 阻塞主脚本，直到所有后台任务（包括那个子 Shell 里的两个）全部跑完
wait

echo "🎉🎉🎉 CUDA $CUDA_ID 上的所有任务彻底跑完！请去 ./results 收割！"