#!/bin/bash

# 提升系统最大文件打开数限制，防止 Ray 引擎因为并发过高报错
ulimit -n 65536

# ---------------------------------------------------------
# 🛠️ 全局资源配置
# ---------------------------------------------------------
# 每个 Trial 占用的显卡比例 (例如 "0.5" 表示 1 张卡同时跑 2 个参数组合)
NUM_GPU=0.25 # 0.33

# 显卡资源分配
CUDA_A="2,3,6"    # 流水线 A 专用卡 (负责跑各种 Classic Baselines / 显式网络)
CUDA_B="3" # 流水线 B 专用卡 (负责跑多任务 / 前沿 SOTA)

# 独立 C 模型的最佳权重路径 (给 TARNET/V8 导航用的先验)
C_CKPT_PATH="/NAS/shith/uplift/ckpts/criteo/train_c/TARNET/c_v1_base/exp_c_explore/best_model.pth"

# 建立统一的实验日志文件夹
LOG_DIR="experiment_logs_0719"
mkdir -p $LOG_DIR

echo "========================================================"
echo "🚀 启动 Uplift 全链路双核并发大阅兵 (0717new 对齐版)"
echo "▶️ [流水线 A] 将使用 GPU: ${CUDA_A} (主攻：经典基线与显式网络)"
echo "▶️ [流水线 B] 将使用 GPU: ${CUDA_B} (主攻：多任务与前沿 SOTA)"
echo "📄 所有任务的详细输出将保存在 $LOG_DIR/ 目录下"
echo "========================================================"

# ---------------------------------------------------------
# 🟢 流水线 A: 经典基线与显式网络 (EFIN_0717new 等)
# ---------------------------------------------------------
(
    echo "🟢 [流水线 A] 已启动！"

    # echo "▶️ [流水线 A] 正在训练: EFIN_0717new (官方 1:1 对齐版)"
    # python main.py --mode tune --task train_y --model EFIN_0717new --version y_efin_0717new --exp_name y_efin_0717new --cuda $CUDA_A --num_per_gpu $NUM_GPU # > $LOG_DIR/log_efin_0717new.txt 2>&1

    # 如有其他 A 组模型，可在此处继续追加...

    echo "✅ [流水线 A] 跑完收工！"
) # & # <-- [核心] 后台运行标志，推入后台，不阻塞脚本下行

# # ---------------------------------------------------------
# # 🔵 流水线 B: 多任务与前沿 SOTA (ECUP_0717new, MTMT_0717new 等)
# # ---------------------------------------------------------
# (
#     echo "🔵 [流水线 B] 已启动！"

    echo "▶️ [流水线 B] 正在训练: ECUP_0717new (修正共享权重/停梯度)"
    python main.py --mode tune --task train_y --model ECUP_0717new --version y_ecup_0717new --exp_name y_ecup_0717new --cuda $CUDA_A --num_per_gpu $NUM_GPU # > $LOG_DIR/log_ecup_0717new.txt 2>&1

    # echo "▶️ [流水线 B] 正在训练: MTMT_0717new (Softmax归一化/ResNet专家)"
    # python main.py --mode tune --task train_y --model MTMT_0717new --version y_mtmt_0717new --exp_name y_mtmt_0717new --cuda $CUDA_B --num_per_gpu $NUM_GPU # > $LOG_DIR/log_mtmt_0717new.txt 2>&1

#     echo "✅ [流水线 B] 跑完收工！"
# ) & # <-- [核心] 后台运行标志，推入后台，与流水线 A 并发并行

# ---------------------------------------------------------
# 总控台：挂起等待
# ---------------------------------------------------------
echo "⏳ 调度完毕！两条并发流水线已投入后台激战。"
echo "💡 提示 1: 你可以随时使用命令查看进度，例如: tail -f $LOG_DIR/log_efin_0717new.txt"
echo "💡 提示 2: 你可以使用 htop 或 nvidia-smi -l 1 监控系统资源利用率。"
echo "💡 提示 3: 如果不需要跑某个模型，直接用 '#' 注释掉对应的 python 命令行即可。"

# wait 命令会让脚本在这里挂起，直到后台的 流水线 A 和 流水线 B 全部执行完毕才会向下继续
wait

echo "========================================================"
echo "🎉 终极捷报：所有 0717new 系列 Uplift 对齐模型已全部并发训练完毕！"
echo "========================================================"