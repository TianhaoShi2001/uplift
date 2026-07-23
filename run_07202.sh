#!/bin/bash
# =========================================================================
# 🚀 Uplift y_v8_s6 核心消融实验 Tune 大盘全并行搜参脚本
# 物理逻辑：全并行轰炸！把 5 个消融组的 Ray Tune 主进程同时挂入后台，共享 2,6,3 卡池！
# =========================================================================
ulimit -n 65536

# -------------------------------------------------------------------------
# 🛠️ 显卡资源与切片份额指定
# -------------------------------------------------------------------------
NUM_GPU=0.5          # 👈 每个 Trial 占用的 GPU 份额 (根据显存大小决定并行密度)
CUDA_POOL="2,6,3"     # 🎯 共享战区 (5个实验的Ray Worker将全部在这3张卡上穿插抢占)
LOG_DIR="experiment_logs_v8_s6_ablations"
mkdir -p $LOG_DIR

# 如果你的模型需要依赖 C 侧的 Checkpoint，请取消注释并指定路径

C_CKPT_PATH="/NAS/shith/uplift/ckpts/criteo/train_c/TARNET/c_v1_base/exp_c_explore/best_model.pth"

echo "========================================================"
echo "🚀 启动 v8_s6 核心消融组合 (Tune 模式) 全并行大阅兵"
echo "▶️ [全局共享卡池] 将独占 GPU: ${CUDA_POOL}"
echo "📄 日志输出目录: $LOG_DIR/"
echo "========================================================"

echo "🟢 [战区] 显卡 ${CUDA_POOL} 已解锁，全线起飞！"

# 1. 挂载 cf_b0 对照尺
echo "▶️ [1/5] 正在起爆: cf_b0 (对照尺：conflict开 / renorm=mean / 无clip / 无ohem) ..."
python main.py \
    --mode tune \
    --task train_y \
    --model TARNET \
    --version y_v8_s6_cf_b0 \
    --exp_name y_v8_s6_cf_b0_tune \
    --cuda "$CUDA_POOL" \
    --num_per_gpu 1 --c_ckpt_path "$C_CKPT_PATH"  > "$LOG_DIR/log_y_v8_s6_cf_b0.txt" 2>&1 &
    # --c_ckpt_path "$C_CKPT_PATH" 

# 2. 挂载 rnc 实验
echo "▶️ [2/5] 正在起爆: rnc (关均值抹平 + 硬帽clip 3/5/8) ..."
python main.py \
    --mode tune \
    --task train_y \
    --model TARNET \
    --version y_v8_s6_rnc \
    --exp_name y_v8_s6_rnc_tune \
    --cuda "$CUDA_POOL" \
    --num_per_gpu $NUM_GPU --c_ckpt_path "$C_CKPT_PATH"  > "$LOG_DIR/log_y_v8_s6_rnc.txt" 2>&1 &

# 3. 挂载 c 实验
echo "▶️ [3/5] 正在起爆: c (开均值抹平 + 硬帽clip 3/5/8) ..."
python main.py \
    --mode tune \
    --task train_y \
    --model TARNET \
    --version y_v8_s6_c \
    --exp_name y_v8_s6_c_tune \
    --cuda "$CUDA_POOL" \
    --num_per_gpu $NUM_GPU --c_ckpt_path "$C_CKPT_PATH"  > "$LOG_DIR/log_y_v8_s6_c.txt" 2>&1 &

# 4. 挂载 ohem 实验
echo "▶️ [4/5] 正在起爆: ohem (renorm开 + 难样本挖掘ohem_pct) ..."
python main.py \
    --mode tune \
    --task train_y \
    --model TARNET \
    --version y_v8_s6_ohem \
    --exp_name y_v8_s6_ohem_tune \
    --cuda "$CUDA_POOL" \
    --num_per_gpu 0.25 --c_ckpt_path "$C_CKPT_PATH"  > "$LOG_DIR/log_y_v8_s6_ohem.txt" 2>&1 &

# 5. 挂载 bl 实验
echo "▶️ [5/5] 正在起爆: bl (下保底min_weight_thres 0.1/0.25/0.5) ..."
python main.py \
    --mode tune \
    --task train_y \
    --model TARNET \
    --version y_v8_s6_bl \
    --exp_name y_v8_s6_bl_tune \
    --cuda "$CUDA_POOL" \
    --num_per_gpu $NUM_GPU --c_ckpt_path "$C_CKPT_PATH"  > "$LOG_DIR/log_y_v8_s6_bl.txt" 2>&1 &

echo "⏳ [雷达监控] 5 大 Tune 进程已全部扔入后台，等待 Ray 引擎火力全开！"

# 🛑 核心锁死：挂起终端，等待所有 Tune 进程收网
wait

echo "========================================================"
echo "🎉 终极捷报：y_v8_s6 所有消融网格 Ray Tune 搜参试验全部圆满落幕！"
echo "========================================================"