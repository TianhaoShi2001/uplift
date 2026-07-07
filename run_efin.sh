#!/bin/bash
# =========================================================================
# 🚀 Uplift Modeling EFIN 官方全宽度容量单流水线串行搜参脚本
# 物理逻辑：单卡单流水线排队串行，拒绝并发，确保每档实验吃满整卡的资源
# =========================================================================
ulimit -n 65536

# -------------------------------------------------------------------------
# 🛠️ 独占单卡资源配置
# -------------------------------------------------------------------------
NUM_GPU=1
CUDA_TARGET="3"  # 🎯 锁死在你的卡 3 上，一个接一个按顺序排队打
LOG_DIR="experiment_logs_efin_rank"
mkdir -p $LOG_DIR

echo "========================================================"
echo "🚀 启动 EFIN 官方拓扑单流水线串行大阅兵"
echo "▶️ 所有任务将独占使用 GPU: ${CUDA_TARGET}，按顺序排队推进"
echo "📄 详细输出日志将保存在 $LOG_DIR/ 目录下"
echo "========================================================"

# --- 实验 3/3: 极限高容量层级 ---
echo "🔥 [3/3] 开始训练: EFIN (efin_embed_dim = 128)..."
python main.py --mode tune --task train_y --model EFIN --version y_baseline_efin_128 --exp_name run_efin_rank128 --cuda $CUDA_TARGET --num_per_gpu $NUM_GPU # > $LOG_DIR/log_efin_rank128.txt 2>&1
echo "✅ [3/3] R=128 训练与最优蛊王全量验证已平稳落幕。"
echo "========================================================"


# --- 实验 1/3: 极简低容量层级 ---
echo "🔥 [1/3] 开始训练: EFIN (efin_embed_dim = 32)..."
python main.py --mode tune --task train_y --model EFIN --version y_baseline_efin_32 --exp_name run_efin_rank32 --cuda $CUDA_TARGET --num_per_gpu $NUM_GPU > $LOG_DIR/log_efin_rank32.txt 2>&1
echo "✅ [1/3] R=32 训练与最优蛊王全量验证已平稳落幕。"
echo "========================================================"

# --- 实验 2/3: 黄金基准容量层级 ---
echo "🔥 [2/3] 开始训练: EFIN (efin_embed_dim = 64)..."
python main.py --mode tune --task train_y --model EFIN --version y_baseline_efin_64 --exp_name run_efin_rank64 --cuda $CUDA_TARGET --num_per_gpu $NUM_GPU > $LOG_DIR/log_efin_rank64.txt 2>&1
echo "✅ [2/3] R=64 训练与最优蛊王全量验证已平稳落幕。"
echo "========================================================"


echo "🏁🏁🏁 终极捷报：EFIN 单流水线三档容量串行调参试验全部圆满收网！"