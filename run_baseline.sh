#!/bin/bash

# ========================================================
# 🚀 Uplift Modeling 全宇宙终极自动化流水线 (Master Script)
# 包含：经典因果基线 + 显式增益模型 + 多任务/特征交互 SOTA
# ========================================================

# 提升系统最大文件打开数限制，防止 Ray 引擎因为并发过高报错
ulimit -n 65536

# ---------------------------------------------------------
# 🛠️ 全局资源配置
# ---------------------------------------------------------
# 每个 Trial 占用的显卡比例 (例如 "0.2" 表示 1 张卡同时跑 5 个参数组合)
NUM_GPU=0.5

# 显卡资源分配
CUDA_A="1" # 流水线 A 专用卡 (负责跑各种 Classic Baselines)
CUDA_B="1" # 流水线 B 专用卡 (负责跑 V8/V10 这种大结构)

# 独立 C 模型的最佳权重路径 (给 TARNET/V8 导航用的先验)
C_CKPT_PATH="/NAS/shith/uplift/ckpts/criteo/train_c/TARNET/c_v1_base/exp_c_explore/best_model.pth"

# 建立统一的实验日志文件夹
LOG_DIR="experiment_logs_0415"
mkdir -p $LOG_DIR

echo "========================================================"
echo "🚀 启动 Uplift 全链路双核并发大阅兵"
echo "▶️ [流水线 A] 将使用 GPU: ${CUDA_A} (主攻：经典基线与显式网络)"
echo "▶️ [流水线 B] 将使用 GPU: ${CUDA_B} (主攻：多任务与前沿 SOTA)"
echo "📄 所有任务的详细输出将保存在 $LOG_DIR/ 目录下"
echo "========================================================"

# ---------------------------------------------------------
# 🟢 流水线 A: 经典基线与显式网络 (S/T, CFR, DragonNet, EUEN, EFIN)
# ---------------------------------------------------------
(
echo "🟢 [流水线 A] 已启动！"

    # echo "▶️ [流水线 A] 正在训练: V1_Base"
    # python main.py --mode tune --task train_y --model TARNET --version y_v1_base --exp_name run_v1_base --cuda $CUDA_A --num_per_gpu $NUM_GPU > $LOG_DIR/log_v1_base.txt 2>&1

    # echo "▶️ [流水线 A] 正在训练: S_Learner"
    # python main.py --mode tune --task train_y --model S_Learner --version y_baseline_s_learner --exp_name run_s_learner --cuda $CUDA_A --num_per_gpu $NUM_GPU  > $LOG_DIR/log_s_learner.txt 2>&1

    # echo "▶️ [流水线 A] 正在训练: T_Learner"
    # python main.py --mode tune --task train_y --model T_Learner --version y_baseline_t_learner --exp_name run_t_learner --cuda $CUDA_A --num_per_gpu $NUM_GPU  > $LOG_DIR/log_t_learner.txt 2>&1

    # echo "▶️ [流水线 A] 正在训练: CFRNet"
    # python main.py --mode tune --task train_y --model CFRNet --version y_baseline_cfrnet --exp_name run_cfrnet --cuda $CUDA_A --num_per_gpu $NUM_GPU  > $LOG_DIR/log_cfrnet.txt 2>&1
    echo "▶️ [流水线 A] 正在训练: CFRNet"
    python main.py --mode tune --task train_y --model CFRNet --version y_baseline_cfrnet_2 --exp_name run_cfrnet_2 --cuda $CUDA_A --num_per_gpu $NUM_GPU  > $LOG_DIR/log_cfrnet_2.txt 2>&1

    # echo "▶️ [流水线 A] 正在训练: DragonNet"
    # python main.py --mode tune --task train_y --model DragonNet --version y_baseline_dragonnet --exp_name run_dragonnet --cuda $CUDA_A --num_per_gpu $NUM_GPU > $LOG_DIR/log_dragonnet.txt 2>&1

    # echo "▶️ [流水线 A] 正在训练: EUEN"
    # python main.py --mode tune --task train_y --model EUEN --version y_baseline_euen --exp_name run_euen --cuda $CUDA_A --num_per_gpu $NUM_GPU > $LOG_DIR/log_euen.txt 2>&1

    # echo "▶️ [流水线 A] 正在训练: EFIN"
    # python main.py --mode tune --task train_y --model EFIN --version y_baseline_efin --exp_name run_efin --cuda $CUDA_A --num_per_gpu $NUM_GPU > $LOG_DIR/log_efin.txt 2>&1

    # echo "▶️ [流水线 A] 正在训练: DESCN"
    # python main.py --mode tune --task train_y --model DESCN --version y_baseline_descn --exp_name run_descn --cuda $CUDA_A --num_per_gpu $NUM_GPU > $LOG_DIR/log_descn.txt 2>&1

    echo "✅ [流水线 A] 所有经典基线与显式网络跑完收工！"
)  # & # <-- [核心] 后台运行标志



# ---------------------------------------------------------
# 总控台：挂起等待
# ---------------------------------------------------------
echo "⏳ 调度完毕！两条并发流水线已投入后台激战。"
echo "💡 提示 1: 你可以随时使用命令  tail -f $LOG_DIR/log_dragonnet.txt  等查看实时进度。"
echo "💡 提示 2: 你可以使用  htop  或  nvidia-smi -l 1  监控系统资源利用率。"
echo "💡 提示 3: 如果不需要跑某个模型，直接在脚本里用 '#' 注释掉对应的 python 命令即可。"

# wait 命令会让脚本在这里等待，直到后台的 流水线A 和 流水线B 全部执行完毕才会退出
wait

echo "========================================================"
echo "🎉 终极捷报：所有 13 个 Uplift 模型（Baseline + SOTA）已全部训练完毕！"
echo "========================================================"