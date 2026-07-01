# #!/bin/bash

# # ========================================================
# # 🚀 终极消融实验矩阵 (The Ultimate Ablation Matrix)
# # 包含 V8 & V11 的 S4, S6, S7, S8 全排列及 5 档温度调参
# # ========================================================

# ulimit -n 65536
# NUM_GPU="0.2"
# CUDA_A="7" # 流水线 A (轻量级探测组)
# CUDA_B="0" # 流水线 B (核心战区组)

# C_CKPT_PATH="/NAS/shith/uplift/ckpts/criteo/train_c/TARNET/c_v1_base/exp_c_explore/best_model.pth"
# LOG_DIR="ablation_logs"
# mkdir -p $LOG_DIR

# echo "========================================================"
# echo "🚀 启动 32 组终极消融实验"
# echo "日志将实时落盘至: $LOG_DIR"
# echo "========================================================"

# # ---------------------------------------------------------
# # 🟣 流水线 A: V8 全部 (8个) + V11 基础探索 (9个) -> 共 17 任务
# # ---------------------------------------------------------
# (

#     echo "▶️ [流水线 A] 阶段 2: V8 S6 Logit加法 x 5档温度"
#     for temp in "0.2" "0.5" "1.0" "2.0" "5.0"; do
#         version="y_v8_s6_temp${temp}"
#         echo "   -> 正在运行 ${version}..."
#         python main.py --mode tune --task train_y --model TARNET --version ${version} --exp_name run_${version} --c_ckpt_path $C_CKPT_PATH --cuda $CUDA_A --num_per_gpu $NUM_GPU > $LOG_DIR/log_${version}.txt 2>&1
#     done

#     echo "✅ [流水线 A] 所有 17 个任务执行完毕下班！"
# ) & 

# # ---------------------------------------------------------
# # 🌟 流水线 B: 核心战区 V11_S6 (加法融合) 3大空间 x 5档温度 -> 共 15 任务
# # ---------------------------------------------------------
# (
#     echo "▶️ [流水线 B] 阶段 1: V11_S6 Lift对齐 x 5档温度"
#     for temp in "0.2" "0.5" "1.0" "2.0" "5.0"; do
#         version="y_v11_s6_lift_temp${temp}"
#         echo "   -> 正在运行 ${version}..."
#         python main.py --mode tune --task train_y --model TARNET --version ${version} --exp_name run_${version} --c_ckpt_path $C_CKPT_PATH --cuda $CUDA_A --num_per_gpu $NUM_GPU > $LOG_DIR/log_${version}.txt 2>&1
#     done

# ) &

# (
#     echo "▶️ [流水线 B] 阶段 2: V11_S6 Z-Score对齐 x 5档温度"
#     for temp in "0.2" "0.5" "1.0" "2.0" "5.0"; do
#         version="y_v11_s6_zscore_temp${temp}"
#         echo "   -> 正在运行 ${version}..."
#         python main.py --mode tune --task train_y --model TARNET --version ${version} --exp_name run_${version} --c_ckpt_path $C_CKPT_PATH --cuda $CUDA_B --num_per_gpu $NUM_GPU > $LOG_DIR/log_${version}.txt 2>&1
#     done
# ) &


# (
#     echo "▶️ [流水线 B] 阶段 3: V11_S6 Rank排序对齐 x 5档温度"
#     for temp in "0.2" "0.5" "1.0" "2.0" "5.0"; do
#         version="y_v11_s6_rank_temp${temp}"
#         echo "   -> 正在运行 ${version}..."
#         python main.py --mode tune --task train_y --model TARNET --version ${version} --exp_name run_${version} --c_ckpt_path $C_CKPT_PATH --cuda $CUDA_B --num_per_gpu $NUM_GPU > $LOG_DIR/log_${version}.txt 2>&1
#     done

#     echo "✅ [流水线 B] 核心战区 15 个温度消融任务全部竣工！"
# ) & 

# # 等待后台流水线 A 和 B 全部跑完
# wait

# echo "========================================================"
# echo "🎉 完美收工！所有 32 个极限消融变体执行完毕。"
# echo "💡 请使用以下命令一键提取最佳模型榜单："
# echo 'grep "\[Target_Y\] AUUC" ablation_logs/* | grep "(GLOBAL!)"'
# echo "========================================================"


#!/bin/bash

# ========================================================
# 🚀 超高温消融实验矩阵 (High Temperature Ablation)
# 针对 V8 S6 及 V11 S6 探索 10.0, 20.0, 50.0, 100.0
# ========================================================

ulimit -n 65536
NUM_GPU="0.2"
CUDA_A="7" # 流水线 A (8个任务)
CUDA_B="0" # 流水线 B (8个任务)

C_CKPT_PATH="/NAS/shith/uplift/ckpts/criteo/train_c/TARNET/c_v1_base/exp_c_explore/best_model.pth"
LOG_DIR="ablation_logs"
mkdir -p $LOG_DIR

echo "========================================================"
echo "🚀 启动 16 组超高温 (T=10~100) 消融实验"
echo "日志将实时落盘至: $LOG_DIR"
echo "========================================================"

# ---------------------------------------------------------
# 🔥 流水线 A: 负责跑 V8_S6 和 V11_S6_Lift
# ---------------------------------------------------------
(
    echo "▶️ [流水线 A] 阶段 1: V8 S6 Logit加法 x 超高温"
    for temp in "10.0" "20.0" "50.0" "100.0"; do
        version="y_v8_s6_temp${temp}"
        echo "   -> 正在运行 ${version}..."
        python main.py --mode tune --task train_y --model TARNET --version ${version} --exp_name run_${version} --c_ckpt_path $C_CKPT_PATH --cuda $CUDA_A --num_per_gpu $NUM_GPU > $LOG_DIR/log_${version}.txt 2>&1
    done
) &

(
    echo "▶️ [流水线 A] 阶段 2: V11_S6 Lift对齐 x 超高温"
    for temp in "10.0" "20.0" "50.0" "100.0"; do
        version="y_v11_s6_lift_temp${temp}"
        echo "   -> 正在运行 ${version}..."
        python main.py --mode tune --task train_y --model TARNET --version ${version} --exp_name run_${version} --c_ckpt_path $C_CKPT_PATH --cuda $CUDA_A --num_per_gpu $NUM_GPU > $LOG_DIR/log_${version}.txt 2>&1
    done

    echo "✅ [流水线 A] 8 个任务执行完毕下班！"
) & 

# ---------------------------------------------------------
# 🔥 流水线 B: 负责跑 V11_S6_Zscore 和 V11_S6_Rank
# ---------------------------------------------------------
(
    echo "▶️ [流水线 B] 阶段 1: V11_S6 Z-Score对齐 x 超高温"
    for temp in "10.0" "20.0" "50.0" "100.0"; do
        version="y_v11_s6_zscore_temp${temp}"
        echo "   -> 正在运行 ${version}..."
        python main.py --mode tune --task train_y --model TARNET --version ${version} --exp_name run_${version} --c_ckpt_path $C_CKPT_PATH --cuda $CUDA_B --num_per_gpu $NUM_GPU > $LOG_DIR/log_${version}.txt 2>&1
    done
) &

(
    echo "▶️ [流水线 B] 阶段 2: V11_S6 Rank排序对齐 x 超高温"
    for temp in "10.0" "20.0" "50.0" "100.0"; do
        version="y_v11_s6_rank_temp${temp}"
        echo "   -> 正在运行 ${version}..."
        python main.py --mode tune --task train_y --model TARNET --version ${version} --exp_name run_${version} --c_ckpt_path $C_CKPT_PATH --cuda $CUDA_B --num_per_gpu $NUM_GPU > $LOG_DIR/log_${version}.txt 2>&1
    done

    echo "✅ [流水线 B] 8 个任务执行完毕下班！"
) & 

# 等待后台流水线 A 和 B 全部跑完
wait

echo "========================================================"
echo "🎉 完美收工！所有 16 个超高温变体执行完毕。"
echo "💡 请使用以下命令一键提取最佳模型榜单："
echo 'grep "\[Target_Y\] AUUC" ablation_logs/* | grep "(GLOBAL!)"'
echo "========================================================"