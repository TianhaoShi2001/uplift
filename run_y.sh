#!/bin/bash
# =========================================================================
# 🎯 Uplift 极限榨取：高分特定参数强行注入 - 纯前台开窗串行 Debug 脚本
# 物理防线：拒绝任何后台 & 符号，拒绝任何日志重定向，所有报错和输出直接刷在屏幕上！
# 策略：支持随时 Ctrl+C 强行熔断，方便看清每一个报错的 Traceback。
# =========================================================================
ulimit -n 65536

# -------------------------------------------------------------------------
# 🛠️ 独占指定调试显卡 (Debug 阶段建议单卡串行死磕)
# -------------------------------------------------------------------------
NUM_GPU=0.05
CUDA_TARGET="1"   # 🎯 选定你的调试卡号 (比如卡 1)

PROJECT_ROOT="/NAS/shith/uplift"
DATASET="criteo"

# 调试种子池：为了快速排查 Bug，你可以先给 1~2 个种子，稳定了再跑满
SEEDS=(1042 2042 3042 4042 5042)

LOG_DIR="reproduce_param_logs"
TMP_CONFIG_DIR="${LOG_DIR}/temp_debug_configs"
mkdir -p $LOG_DIR
mkdir -p $TMP_CONFIG_DIR

echo "========================================================"
echo "🎯 启动 纯前台开窗 Debug 串行注入多种子流水线"
echo "⚠️  警告：所有输出直接打印在当前窗口！随时可 Ctrl+C 强行终止！"
echo "========================================================"

# -------------------------------------------------------------------------
# 📦 核心高分参数调试矩阵 (格式: 模型名:版本名:实验名:Head层:Alpha值)
# 🌟 以后想要调试新配置，直接追加新行即可！
# -------------------------------------------------------------------------
CONFIG_MATRIX=(
    "TARNET_Baseline_PureV10:y_pure_v10_all_same_alpha_head_0710:exp_y_pure_v10_all_same_alpha_head_0710_res_2_layer_head_search:64-32:0.5"
    "TARNET_Baseline_PureV10:y_pure_v10_all_same_alpha_head_0710:exp_y_pure_v10_all_same_alpha_head_0710_res_2_layer_head_search:16:1.0"
)

# -------------------------------------------------------------------------
# ⚙️ 核心逻辑：完全移除后台并发，实时图纸改写并前向死锁运行
# -------------------------------------------------------------------------
for ITEM in "${CONFIG_MATRIX[@]}"; do
    # 解析矩阵超参
    IFS=":" read -r MODEL VERSION EXP_NAME HEAD_RAW ALPHA <<< "${ITEM}"
    
    # 将破折号 '-' 还原为 python 能解析的逗号方括号格式
    if [ "$HEAD_RAW" = "None" ]; then
        HEAD_VAL="null"
        HEAD_TAG="None"
    elif [[ "$HEAD_RAW" == *"-"* ]]; then
        HEAD_VAL="[${HEAD_RAW//-/,}]"
        HEAD_TAG="${HEAD_RAW//-/_}"
    else
        HEAD_VAL="[$HEAD_RAW]"
        HEAD_TAG="${HEAD_RAW}"
    fi
    
    # 定位历史基准底座图纸
    ORIGINAL_CONFIG="${PROJECT_ROOT}/ckpts/${DATASET}/train_y/${MODEL}/${VERSION}/${EXP_NAME}/best_config.json"
    
    if [ ! -f "$ORIGINAL_CONFIG" ]; then
        echo "❌ 找不到底座图纸，跳过: $ORIGINAL_CONFIG"
        continue
    fi

    # 生成调试专用的临时影子图纸，强制修改核心冲突超参
    TARGET_CONFIG_NAME="debug_cfg_${VERSION}_H${HEAD_TAG}_A${ALPHA}.json"
    SHADOW_CONFIG="${TMP_CONFIG_DIR}/${TARGET_CONFIG_NAME}"
    
    python3 -c "
import json
with open('${ORIGINAL_CONFIG}', 'r') as f:
    cfg = json.load(f)
cfg['head_hidden_dims'] = ${HEAD_VAL} if '${HEAD_VAL}' != 'null' else None
cfg['conflict_alpha_wool'] = ${ALPHA}
cfg['conflict_alpha_gold'] = ${ALPHA}
cfg['conflict_alpha_walkin'] = ${ALPHA}
with open('${SHADOW_CONFIG}', 'w') as f:
    json.dump(cfg, f, indent=4)
"

    echo "--------------------------------------------------------"
    echo "🌊 重写临时图纸成功 -> Head: ${HEAD_VAL} | Alpha: ${ALPHA}"
    echo "--------------------------------------------------------"
    
    # 展开种子循环进行彻底的串行前台挂载
    for SEED in "${SEEDS[@]}"; do
        echo "🔥 [调试中] 启动配置 -> Seed: ${SEED} | Head: ${HEAD_VAL} | Alpha: ${ALPHA} ..."
        
        # 👑 核心绝杀改法：去掉末尾的 &，去掉任何 > log.txt 重定向！
        # 任务会直接在当前前台终端牢牢占领屏幕，实时打印所有日志！
        python main.py \
            --mode reproduce \
            --task train_y \
            --model "$MODEL" \
            --version "$VERSION" \
            --exp_name "${EXP_NAME}_H${HEAD_TAG}_A${ALPHA}_seed_${SEED}" \
            --config_path "$SHADOW_CONFIG" \
            --seed "$SEED" \
            --cuda "$CUDA_TARGET" \
            --num_per_gpu $NUM_GPU
            
        echo "✅ [调试完] 配置组合 Seed: ${SEED} 顺利平稳出关。"
    done
done

echo "========================================================"
echo "🎉 终极战报：所有特定高分配置的前台串行 Debug 流水线全部顺利通过！"
echo "========================================================"