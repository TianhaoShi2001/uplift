#!/bin/bash
# =========================================================================
# 🚀 Uplift Modeling Ours S4/S6/V10 终极容量网格多流水线搜参脚本
# 物理逻辑：A/B 两路大军在后台各自独占指定卡，多任务 Ray Tune 并行推进，拒绝直积核爆
# =========================================================================
ulimit -n 65536

# -------------------------------------------------------------------------
# 🛠️ 显卡资源与大盘份额指定
# -------------------------------------------------------------------------
NUM_GPU=0.14     # 每个 Trial 占用的 GPU 份额 (控制单卡并发密度)
CUDA_A="3,2,6"      # 🎯 流水线 A 绑定的 GPU 卡号 (负责跑 S4 经典组与 Pure V10 爆发组)
CUDA_B="3,2,6"        # 🎯 流水线 B 绑定的 GPU 卡号 (负责跑 Ours S6 深度温度多维联动组)
LOG_DIR="experiment_logs_ours_s4"
mkdir -p $LOG_DIR

echo "========================================================"
echo "🚀 启动 Ours S4 + S6 + Pure V10 三轴网格 Ray Tune 搜参大阅兵"
echo "▶️ [流水线 A] 将独占 GPU: ${CUDA_A} (负责 S4 与 Pure V10 战区)"
echo "▶️ [流水线 B] 将独占 GPU: ${CUDA_B} (负责 Ours S6 温度超参联动战区)"
echo "📄 所有任务的详细输出将重定向保存在 $LOG_DIR/ 目录下"
echo "========================================================"


C_CKPT_PATH="/NAS/shith/uplift/ckpts/criteo/train_c/TARNET/c_v1_base/exp_c_explore/best_model.pth"

# -------------------------------------------------------------------------
# 🟢 [流水线 A] 战区 (独占 GPU_A, 顺次轰炸 S4 与 V10 大网格)
# -------------------------------------------------------------------------
    echo "🟢 [流水线 A] 已在 GPU ${CUDA_A} 顺畅起航！"
    
    # 1. 挂载 Ours S4 网格探索
    echo "▶️ [流水线 A] 1/2 正在全面起爆: Ours S4 多维网格搜参..."
    python main.py \
        --mode tune \
        --task train_y \
        --model TARNET \
        --version y_ours_s4_conflict_0717_h_None_search_alpha \
        --exp_name y_ours_s4_conflict_0717_h_None_search_alpha \
        --cuda "$CUDA_A" \
        --c_ckpt_path "$C_CKPT_PATH" --num_per_gpu $NUM_GPU > "$LOG_DIR/log_y_ours_s4_conflict_0717_h_None_search_alpha.txt"  2>&1 &
        
    2. 挂载 Pure V10 网格探索
    echo "▶️ [流水线 A] 2/2 正在紧接着起爆: Pure V10 对齐网格消融..."
    python main.py \
        --mode tune \
        --task train_y \
        --model TARNET_Baseline_PureV10 \
        --version y_ours_s6_conflict_0717_hNone_search_alpha \
        --exp_name y_ours_s6_conflict_0717_hNone_search_alpha \
        --cuda "$CUDA_A" \
        --c_ckpt_path "$C_CKPT_PATH" --num_per_gpu $NUM_GPU > "$LOG_DIR/log_y_ours_s6_conflict_0717_hNone_search_alpha.txt" 2>&1 &
        
#     echo "✅ [流水线 A] 负责的所有网格探索全部圆满收网。"

# # -------------------------------------------------------------------------
# # 🔵 [流水线 B] 战区 (独占 GPU_B, 挂起带有 Lambda 联动的 S6 大网格)
# # -------------------------------------------------------------------------
    # echo "🔵 [流水线 B] 已在 GPU ${CUDA_B} 顺畅起航！"
    
    # echo "▶️ [流水线 B] 正在全面起爆: Ours S6 温度 × 权值衰减 × 错位Alpha 联动大网格搜参..."
    python main.py \
        --mode tune \
        --task train_y \
        --model TARNET \
        --version y_pure_v10_0717_h_None_search_alpha \
        --exp_name y_pure_v10_0717_h_None_search_alpha \
        --cuda "$CUDA_A" \
        --c_ckpt_path "$C_CKPT_PATH" --num_per_gpu $NUM_GPU  > "$LOG_DIR/log_y_pure_v10_0717_h_None_search_alphah.txt" 2>&1 &
        
    echo "✅ [流水线 B] 负责的 S6 温度大网格消融全部收网。"

# 🛑 核心锁死：等待两条大流水线后台进程全部完工、合流回营
# wait

echo "========================================================"
echo "🎉 终极捷报：Ours S4/S6/V10 大盘多维网格 Ray Tune 搜参试验全部圆满落幕！"
echo "========================================================"