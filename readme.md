🚀 全链路 Uplift 联合优化与多任务代码库文档
本代码库是一套面向真实业务场景的因果推断与 Uplift Modeling 深度学习框架。整体架构采用解耦设计，支持特征工程、多阶段联合训练（Stage 1 因果表征 & Stage 3 靶向优化）、大规模超参搜索以及细粒度的业务指标评估。

📂 一、 全局目录与核心模块功能
整个项目的文件分工明确，各个模块各司其职：

1. main.py (实验中枢与超参调度)
定位：整个项目的最高指挥官。

配置解析： 负责读取 JSON/YAML 配置文件，获取当前实验的任务类型（train_c 或 train_y）、模型超参、数据路径等。

分布式调度： 集成了 Ray Tune 引擎，负责拉起多个 Trial 并发执行网格搜索或贝叶斯调参，最大化利用 GPU 资源。

流程串联： 统筹调用 data_pipeline 加载数据，实例化 trainer 执行训练，并在实验结束后触发全局最优模型的最终落盘。

2. data_pipeline.py (数据加载与特征工程)
定位：模型的粮草大本营。

UpliftDataset: 核心 Dataset 类。负责从原始存储中读取数据，并严格区分连续特征 (x_cont)、类别特征 (x_cat)、干预标签 (t)、真实响应标签 (y) 以及潜在的 Complier 伪标签 (c)。

特征处理： 负责离散特征的 Tokenize（编码对齐）和连续特征的归一化/截断处理。

uplift_collate_fn: 数据拼装函数，保证 Dataloader 吐出的 Batch 结构稳定，特别是将类别特征整合成字典形式供 Embedding 层快速查询。

3. models.py (网络骨架与特征交互)
定位：模型的本体与结构图纸。

底层表示 (Shared Bottom)： 包含连续特征的 MLP 映射和类别特征的 Embedding 层，负责将稀疏高维特征转化为稠密表征。

特征融合与路由： 包含多种特征交互网络。从基础的 TARNET_Baseline 双塔结构，到引入门控机制的 TARNET_MoE，再到具有残差和截断机制的 TARNET_V7_Truncated_MoE，处理不同表征之间的复杂非线性组合。

双阶段定义： * C模型结构返回 (c0_pred, c1_pred, z_c)，用于特征干预不变性对齐。

Y模型结构返回 (y0_pred, y1_pred, pi_dict)，并且能无缝内嵌固化好的 C模型参数。

4. trainer.py (训练引擎)
定位：模型的“健身房”与“裁判”。

build_model: 动态模型装配车间。负责根据 config 拉起特定的网络，并处理 Y 模型对 C 模型权重的依赖加载。

train_trial: 核心训练循环。内置自动混合精度 (AMP) 加速，执行前向传播，调起对应的 Loss 进行反向传播。

打擂台机制： 全局维护一个最优指标记录（如 global_best_metric.txt），确保在 Ray 分布式并发下也能准确保存真正的 Global Best Model。

5. losses.py (优化目标与数学引擎)
定位：引导模型收敛的指南针。

表征对齐 (Stage 1): 提供 FocalLoss 解决样本不平衡，利用 mmd_loss 或 sliced_wasserstein_distance 消除选择偏差，强迫干预与控制组的表征空间对齐。

靶向联合 (Stage 3): 实现 Uplift 特有的魔法 Loss，包括 strata_weighted（靶向重加权，专注 Complier）、variance_reg（极值惩罚，压制自然转化客的 Uplift）和 pairwise（直接优化排序 AUUC）。

6. evaluator.py (离线评估器)
定位：对齐业务视角的质检员。

抛弃纯分类指标，专注增量收益。核心产出 AUUC（曲线下面积）、AUQC（Qini）和 Lift@10/30。

辅助提供 PCOC、MAE 监控模型的绝对概率预估是否存在偏差（Calibration），并落盘全量预测分布用于可视化。

-------------------------------------------------------------

🛠️ 二、 进阶指南：如何分 4 步接入多任务学习 (Multitask Learning)?
当你需要从单目标预测演进到多任务（例如同时预测点击率 CTR、转化率 CVR，或者主任务是转化、辅助任务是分享/收藏），你需要按照以下 4 步对当前框架进行动刀重构：

1. 第一步：注意 data_pipeline.py 
模型要学多个任务，数据必须供得上。

注意点：在 UpliftDataset 的 __getitem__ 以及 uplift_collate_fn 中，现在输出的是(x_cont, x_cat, t, y, c)，需要注意除了原本的y，c（中介任务，如点击转化的转化）作为第二个任务的目标，也需要用上。


2. 第二步：改造 models.py (拆分 Task Towers)
原本前向传播最后只有一个分类头，现在需要长出多个“脑袋”。

改动点：在你的 TARNET 类中，将原本最后的输出层拆分成独立的 MLP 塔（Tower_Main, Tower_Aux1）。

前向传播 (Forward)：

Python
# 原来: return y0_pred, y1_pred, pi_dict
# 重构后返回字典，方便扩展:
return {
    "main_task": (y0, y1),
    "aux_task": (c0, y1),
    "pi_dict": pi_dict
}
3. 第三步：改造 losses.py (增加多目标聚合器)
各个任务的量级和收敛速度不同，不能简单无脑相加。

改动点：在 compute_stage3_loss 中，分别计算主任务和辅助任务的 Loss。

聚合策略：

静态权重：total_loss = 1.0 * loss_main + 0.2 * loss_aux

动态权重 / 梯度对齐：引入不确定性权重 (Uncertainty Weighting) 或者 PCGrad 算法，缓解“跷跷板效应”（一个任务变好导致另一个任务变差）。

4. 第四步：改造 trainer.py (打通训练与日志血脉)
把前面的改动在训练循环中串接起来。

改动点：

解包数据时接收多个标签。

接收 model 吐出的字典格式预测结果。

将预测结果和对应的标签送入 losses.py 算总 Loss，执行 total_loss.backward()。

关键埋点：在 safe_report 和控制台打印时，必须把 loss_main 和 loss_aux 拆开来记录（不要只记一个 Total Loss），否则你根本不知道是哪个任务崩溃了。