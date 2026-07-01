# MTMT (Multi-Treatment Multi-Task) Uplift 架构拆解与数学原理

[cite_start]**来源:** *Multi-Treatment Multi-Task Uplift Modeling for Enhancing User Growth* (Wei et al., 2024) [cite: 3, 4]

---

## 1. 核心动机 (Motivation)
传统的 Uplift 模型在真实工业场景（如游戏、电商）中面临两大局限性：
1.  [cite_start]**单任务视角的局限:** 传统的 Uplift 模型大多只关注单一目标（Single-Task） [cite: 12][cite_start]。而在实际业务中，干预手段通常会同时影响多个指标（如短期活跃度、长期留存率、付费率） [cite: 64][cite_start]。忽略多任务之间的隐式关联会导致表征学习不充分，以及 ITE（个体处理效应）估计存在偏差 [cite: 66]。
2.  [cite_start]**粗暴的特征拼接:** 很多现有模型在处理干预特征（Treatment Features）时，仅仅是简单地将干预向量与用户特征进行拼接（Concatenation）或加权求和 [cite: 78][cite_start]。这种做法忽略了**不同干预对用户不同特征的敏感度是不同的**，无法在面向具体任务时，显式捕捉“干预”与“用户画像”之间的深层交互关系 [cite: 77, 79]。

[cite_start]**MTMT 的破局点:** 引入 MMoE 进行多任务底层表征解耦，并创新性地使用 **Self-Attention 机制**让干预信号主动去“Query”相关的用户特征 [cite: 84, 87, 98]。

---

## 2. 问题定义与因果框架
[cite_start]根据 Neyman-Rubin 潜在结果框架 [cite: 134]：
* [cite_start]观测数据集定义为 $\mathcal{D}=\{[x_{i}, t_{i}, y_{i}]\}_{i=1}^{n}$，其中 $x_{i} \in \mathbb{R}^{d}$ 为 $d$ 维用户特征，$t_{i} \in \{0,1\}$ 为干预指示变量，$y_{i} \in \mathbb{R}^{k}$ 代表 $k$ 个独立的业务目标任务 [cite: 133]。
* [cite_start]$y_{i}^{k}(0)$ 表示第 $i$ 个用户在未受干预时，在任务 $k$ 上的潜在自然响应 [cite: 135]。
* [cite_start]$\hat{\tau}^{k}(x_{i})$ 表示该用户在任务 $k$ 上的个体因果增量 (ITE/Uplift) [cite: 139]。
* 最终观测结果可表示为：
    $$y_{i}^{k}(1) = y_{i}^{k}(0) + \hat{\tau}^{k}(x_{i})$$

---

## 3. 核心组件拆解

### 3.1 Task-Oriented Feature Encoder (多任务底层解耦)
[cite_start]为了显式建模任务间的关系并提取任务专属特征，MTMT 采用 MMoE (Multi-gate Mixture-of-Experts) 作为用户特征编码器 [cite: 150]。
* **数学原理:** 假设底层有 $n$ 个专家网络 $E^{1}, E^{2}, \dots, E^{n}$，对于第 $k$ 个任务，有一个专属的门控网络 $G^{k}$ 用于评估各个专家的重要性。
    门控权重计算：
    $$G^{k} = softmax(W_{g}^{k}x_{i})$$
    [cite_start]其中 $W_{g}^{k}$ 为可学习的权重矩阵 [cite: 190, 191]。
    [cite_start]任务 $k$ 的专属特征表征 $\phi_{i}^{k}$ 为所有专家输出的加权求和 [cite: 184, 189]：
    $$\phi_{i}^{k} = G^{k}(x_{i}) \cdot \{E^{1}(x_{i}), E^{2}(x_{i}), \dots, E^{n}(x_{i})\}$$
* [cite_start]**物理意义:** $\phi_{i}^{k}$ 中**完全不包含干预信息**，它是纯粹的用户静态画像与历史行为特征在任务 $k$ 视角下的提纯 [cite: 192]。

### 3.2 Natural Response Estimation (独立自然响应预测)
在因果推断中，自然转化（Base/Natural Response）不应受到干预特征的污染。
* **数学原理:**
    利用任务专属表征 $\phi_{i}^{k}$，通过一个线性投影头直接预测如果**不发券**情况下的基准转化率 [cite: 192, 193]：
    $$\overline{y}_{i}^{k}(0) = W_{proj}^{k}\phi_{i}^{k}$$

### 3.3 User-Treatment Feature Interaction (干预-特征注意力交互)
[cite_start]这是该论文的核心创新点。抛弃传统的 Concat，采用自注意力机制显式建模干预与特征的化学反应 [cite: 98, 204]。
* **数学原理:**
    [cite_start]首先将离散的干预信号 $t_{i}$ 映射为稠密的 Embedding 向量 $\epsilon_{i}$ [cite: 195]。
    随后，将干预 Embedding $\epsilon_{i}$ 作为 **Query**，将用户的任务专属特征 $\phi_{i}^{k}$ 作为 **Key** 和 **Value** [cite: 205]：
    $$\psi_{i}^{k} = softmax\left(\frac{(W^{\tau}\epsilon_{i}) \times (W^{u}\phi_{i}^{k})^{T}}{\sqrt{d_{u}}}\right) W^{v}\phi_{i}^{k}$$
    *(注：公式适配单干预场景进行了降维简化)* [cite: 207, 209]。
* **物理意义:** 干预向量在询问高维特征空间：“如果要预测增量，我应该重点关注这个用户的哪些特征？”。输出的 $\psi_{i}^{k}$ 是一个**“被干预信号激活后的”处理效应感知表征 (Treatment-aware Representation)** [cite: 18, 206]。

### 3.4 Uplift Estimation & Final Output (增量增强与最终叠加)
* **数学原理:**
    将交互后的表征 $\psi_{i}^{k}$ 送入 MLP 增强器，直接预测出任务 $k$ 上的因果增量 (Uplift) [cite: 211, 213]：
    $$\overline{\tau}^{k}(x_{i}) = W^{\tau}_{mlp}MLP(\psi_{i}^{k})$$
    最终干预组的预测响应严格遵循加法结构 [cite: 224, 225]：
    $$\overline{y}_{i}^{k}(1) = \overline{y}_{i}^{k}(0) + \overline{\tau}^{k}(x_{i})$$
* **物理意义:** 这种结构在数学上保证了模型是对 Uplift 本身进行建模，而非仅仅用两个独立塔分别拟合响应。它强制模型去解释“增量是从哪里来的”。

---

## 4. 损失函数 (Multi-Task Masked Loss)
在训练阶段，由于无法同时观测到用户的两个潜在结果，损失函数根据实际干预分配进行 Mask。
[cite_start]针对所有任务 $k \in \{1, \dots, K\}$，联合优化 MSE / BCE 损失 [cite: 232, 233, 240]：
$$\mathcal{L}_{total} = \sum_{k=1}^{K} \left[ \sum_{i \in Control} Loss(y_{i}^{k}, \overline{y}_{i}^{k}(0)) + \sum_{i \in Treatment} Loss(y_{i}^{k}, \overline{y}_{i}^{k}(1)) \right]$$
通过联合反向传播，辅助任务的梯度可以帮助主任务的 MMoE 底层专家网络提取出更加泛化、鲁棒的用户共性特征。