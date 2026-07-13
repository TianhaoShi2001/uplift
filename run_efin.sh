#!/bin/bash
# =========================================================================
# 🚀 Uplift Modeling EFIN 官方全宽度容量双流水线并发搜参脚本
# 物理逻辑：两路大军后台异步并发（&），分流到指定 GPU 独占或切分运行，极限压榨算力
# =========================================================================
ulimit -n 65536

# -------------------------------------------------------------------------
# 🛠️ 双核资源与显卡精细配置
# -------------------------------------------------------------------------
NUM_GPU=1     # 每个 Trial 分配的 GPU 份额
CUDA_A="3"       # 流水线 A 锁死在卡 3 (跑 R=32 和 R=128)
CUDA_B="0"       # 流水线 B 锁死在卡 1 (跑 R=64 黄金基准)
LOG_DIR="experiment_logs_efin_rank"
mkdir -p $LOG_DIR

echo "========================================================"
echo "🚀 启动 EFIN 官方拓扑双流水线并发大阅兵"
echo "▶️ [流水线 A] 将使用 GPU: ${CUDA_A} (负责 R=32 与 R=128 极端两级)"
echo "▶️ [流水线 B] 将使用 GPU: ${CUDA_B} (负责 R=64 黄金核心容量)"
echo "📄 所有任务的详细输出将保存在 $LOG_DIR/ 目录下"
echo "========================================================"

# -------------------------------------------------------------------------
# 🟢 [流水线 A] 战区 (独占 GPU: 3)
# -------------------------------------------------------------------------
(
    echo "🔵 [流水线 B] 已在 GPU ${CUDA_B} 启动！"
    

    
    echo "▶️ [流水线 A] 正在训练 [3/3]: EFIN (efin_embed_dim = 128)..."
    python main.py --mode tune --task train_y --model EFIN --version y_baseline_efin_128 --exp_name run_efin_rank128 --cuda $CUDA_A --num_per_gpu $NUM_GPU > $LOG_DIR/log_efin_rank128.txt 2>&1
    echo "✅ [流水线 A] R=128 训练与蛊王验证已平稳落幕。"

    echo "▶️ [流水线 B] 正在训练 [2/3]: EFIN (efin_embed_dim = 64)..."
    python main.py --mode tune --task train_y --model EFIN --version y_baseline_efin_64 --exp_name run_efin_rank64 --cuda $CUDA_A --num_per_gpu $NUM_GPU > $LOG_DIR/log_efin_rank64.txt 2>&1
    echo "✅ [流水线 B] R=64 训练与蛊王验证已平稳落幕。"

    echo "🏁 [流水线 A] 极端容量对比任务全部闭环完成！"
)  &

# -------------------------------------------------------------------------
# 🔵 [流水线 B] 战区 (独占 GPU: 1)
# -------------------------------------------------------------------------
(


    echo "🟢 [流水线 A] 已在 GPU ${CUDA_A} 启动！"
    
    echo "▶️ [流水线 A] 正在训练 [1/3]: EFIN (efin_embed_dim = 32)..."
    python main.py --mode tune --task train_y --model EFIN --version y_baseline_efin_32 --exp_name run_efin_rank32 --cuda $CUDA_B --num_per_gpu $NUM_GPU > $LOG_DIR/log_efin_rank32.txt 2>&1
    echo "✅ [流水线 A] R=32 训练与蛊王验证已平稳落幕。"

    echo "🏁 [流水线 B] 黄金核心容量任务闭环完成！"
)   &

# 🛑 核心物理阻断：死死按住主进程，等待两路大军全部合流落幕
wait

echo "========================================================"
echo "🎉 终极捷报：EFIN 双流水线所有容量组并发搜参试验全部平稳落幕！"
echo "========================================================"