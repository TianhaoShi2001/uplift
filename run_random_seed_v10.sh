#!/bin/bash
# =========================================================================
# 🚀 Uplift 极限榨取计划：PureV10 特种多层预测头多种子保底复现流水线
# 策略：空出一个补一个，时刻保持指定卡满载，告别显存闲置！
# 对齐目标：exp_y_pure_v10_all_same_alpha_head_0710_res_2_layer_head_search
# =========================================================================
ulimit -n 65536

# -------------------------------------------------------------------------
# 🛠️ 全局环境与【差异化并发容量】配置 (卡号与并发数自由指定)
# -------------------------------------------------------------------------
NUM_GPU=0.05
CUDA_A="7"       # 🎯 流水线 A 绑定的 GPU 卡号
MAX_JOBS_A=1     # 👈 动态队列容量：卡 1 最多同时跑 3 个种子任务
CUDA_B="6"       # 🎯 流水线 B 绑定的 GPU 卡号
MAX_JOBS_B=1     # 👈 动态队列容量：卡 6 最多同时跑 3 个种子任务

PROJECT_ROOT="/NAS/shith/uplift"
DATASET="criteo"

# 🌟 5 组黄金保底随机种子矩阵
SEEDS=(1042 2042 3042 4042 5042)
LOG_DIR="reproduce_pure_v10_logs"
mkdir -p $LOG_DIR

echo "========================================================"
echo "🚀 启动 PureV10 特异 head 实验动态队列多种子复现 (Seed: 1042~5042)"
echo "▶️ [GPU ${CUDA_A}] 队列容量: ${MAX_JOBS_A} 并发进程"
echo "▶️ [GPU ${CUDA_B}] 队列容量: ${MAX_JOBS_B} 并发进程"
echo "📄 日志输出目录: $LOG_DIR/"
echo "========================================================"

# -------------------------------------------------------------------------
# 📦 模型清单拆分 (物理路径解析: MODEL:VERSION:EXP_NAME)
# 🌟 核心对齐：三段字符串必须与你 NAS 盘上的真实物理文件夹名字像素级对齐！
# -------------------------------------------------------------------------
TARGET_ITEM="TARNET_Baseline_PureV10:y_pure_v10_all_same_alpha_head_0710:exp_y_pure_v10_all_same_alpha_head_0710_res_2_layer_head_search"

# -------------------------------------------------------------------------
# ⚙️ 核心：滑动窗口动态队列控制器
# -------------------------------------------------------------------------
run_dynamic_queue() {
    local GPU_ID=$1
    local MAX_CONCURRENCY=$2
    shift 2
    local SPECIFIC_SEEDS=("$@")
    
    IFS=":" read -r MODEL VERSION EXP_NAME <<< "${TARGET_ITEM}"
    CONFIG_PATH="${PROJECT_ROOT}/ray_results/tune/${DATASET}/train_y/${MODEL}/${VERSION}/${EXP_NAME}/best_config.json"
    
    # 防御检测：如果主搜参阶段根本没生成 best_config.json，直接熔断提示
    if [ ! -f "$CONFIG_PATH" ]; then
        echo "❌ [GPU $GPU_ID] 找不到指定的蛊王图纸，请核实路径: $CONFIG_PATH"
        return 1
    fi

    echo "🌊 [GPU $GPU_ID] 启动动态滑动队列 (分配种子: ${SPECIFIC_SEEDS[*]}) ..."
    
    for SEED in "${SPECIFIC_SEEDS[@]}"; do
        # 1. 把复现任务挂入后台异步运行 (指定 --mode reproduce)
        python main.py \
            --mode reproduce \
            --task train_y \
            --model "$MODEL" \
            --version "$VERSION" \
            --exp_name "${EXP_NAME}_seed_${SEED}" \
            --config_path "$CONFIG_PATH" \
            --seed "$SEED" \
            --cuda "$GPU_ID" \
            --num_per_gpu $NUM_GPU > "$LOG_DIR/log_rep_${VERSION}_seed_${SEED}.txt" 2>&1 
            
        echo "   -> 🚀 [GPU $GPU_ID] 投递成功 | 队列状态: $(jobs -pr | wc -l)/${MAX_CONCURRENCY} | Seed: ${SEED}"
        
        # 2. 🌟 滑动窗口控制阀门：跑满上限则死循环 sleep 阻塞，空出一个位子立马补发
        while [ $(jobs -pr | wc -l) -ge "$MAX_CONCURRENCY" ]; do
            sleep 1
        done
    done

    # 所有的任务都投递完了，等队列里最后几个收尾
    echo "⏳ [GPU $GPU_ID] 指定种子任务投递完毕，等待最后选手冲线..."
    wait
    echo "✅ [GPU $GPU_ID] 复现队列清空！"
}

# -------------------------------------------------------------------------
# 🚀 启动双队列分流排队 (Pipeline A 分流前 3 个种子，Pipeline B 分流后 2 个种子)
# -------------------------------------------------------------------------
SEEDS_A=("1042" "2042" "3042")
SEEDS_B=("4042" "5042")

# 开启流水线 A (独占 GPU A)
(
    run_dynamic_queue "$CUDA_A" "$MAX_JOBS_A" "${SEEDS_A[@]}"
) 

# 开启流水线 B (独占 GPU B)
(
    run_dynamic_queue "$CUDA_B" "$MAX_JOBS_B" "${SEEDS_B[@]}"
) 

# 强行挂起主终端，等待两大显卡集群合流
wait

echo "========================================================"
echo "🎉 终极捷报：PureV10 特异 head 实验 5 组种子保底复现全部落幕！"
echo "========================================================"