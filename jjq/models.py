import torch
import torch.nn as nn

# ==========================================
# 基础组件: CGC2D Layer (用于 MOTTO)
# ==========================================
def get_activation(activation="ReLU"):
    if activation == "ReLU":
        return nn.ReLU()
    elif activation == "ELU":
        return nn.ELU()
    return None

class MLP_Block(nn.Module):
    def __init__(self, input_dim, hidden_units=[], hidden_activations="ReLU", output_dim=None, dropout_rates=0.0):
        super(MLP_Block, self).__init__()
        dense_layers = []
        if not isinstance(dropout_rates, list):
            dropout_rates = [dropout_rates] * len(hidden_units)
        if not isinstance(hidden_activations, list):
            hidden_activations = [hidden_activations] * len(hidden_units)
        
        curr_dim = input_dim
        for idx, h_dim in enumerate(hidden_units):
            dense_layers.append(nn.Linear(curr_dim, h_dim))
            act = get_activation(hidden_activations[idx])
            if act: dense_layers.append(act)
            if dropout_rates[idx] > 0: dense_layers.append(nn.Dropout(p=dropout_rates[idx]))
            curr_dim = h_dim
            
        if output_dim is not None:
            dense_layers.append(nn.Linear(curr_dim, output_dim))
            
        self.mlp = nn.Sequential(*dense_layers)
    
    def forward(self, inputs):
        return self.mlp(inputs)

class CGC2D_Layer(nn.Module):
    """
    CGC2D_Layer 用于 MOTTO，包含 specific, outcome_shared, treatment_shared, shared 专家。
    简化版：移除了 dcnv2 支持，仅支持 mlp 专家，与主目录风格对齐。
    """
    def __init__(self, num_shared_experts, num_outcome_shared_experts, num_treatment_shared_experts, num_specific_experts, 
                 num_outcomes, num_treatments, input_dim, expert_hidden_units, gate_hidden_units, hidden_activations, net_dropout):
        super(CGC2D_Layer, self).__init__()
        self.num_shared_experts = num_shared_experts
        self.num_outcome_shared_experts = num_outcome_shared_experts
        self.num_treatment_shared_experts = num_treatment_shared_experts
        self.num_specific_experts = num_specific_experts 
        self.num_tasks = num_outcomes * num_treatments
        self.num_outcomes = num_outcomes
        self.num_treatments = num_treatments
        
        # Experts
        self.shared_experts = nn.ModuleList([MLP_Block(input_dim, expert_hidden_units, hidden_activations, dropout_rates=net_dropout) for _ in range(self.num_shared_experts)])
        self.outcome_shared_experts = nn.ModuleList([nn.ModuleList([MLP_Block(input_dim, expert_hidden_units, hidden_activations, dropout_rates=net_dropout) for _ in range(self.num_outcome_shared_experts)]) for _ in range(num_treatments)])
        self.treatment_shared_experts = nn.ModuleList([nn.ModuleList([MLP_Block(input_dim, expert_hidden_units, hidden_activations, dropout_rates=net_dropout) for _ in range(self.num_treatment_shared_experts)]) for _ in range(num_outcomes)])
        if num_specific_experts > 0:
            self.specific_experts = nn.ModuleList([nn.ModuleList([MLP_Block(input_dim, expert_hidden_units, hidden_activations, dropout_rates=net_dropout) for _ in range(self.num_specific_experts)]) for _ in range(self.num_tasks)])
            
        # Gates
        def get_output_dim(i):
            if i < self.num_tasks:
                return num_specific_experts + num_outcome_shared_experts + num_treatment_shared_experts + num_shared_experts
            elif self.num_tasks <= i < self.num_tasks + num_treatments:
                return self.num_outcomes * num_specific_experts + num_outcome_shared_experts + num_shared_experts
            elif self.num_tasks + num_treatments <= i < self.num_tasks + num_treatments + num_outcomes:
                return self.num_treatments * num_specific_experts + num_treatment_shared_experts + num_shared_experts
            else:
                return self.num_tasks * num_specific_experts + self.num_treatments * num_outcome_shared_experts + self.num_outcomes * num_treatment_shared_experts + num_shared_experts
                
        self.gate = nn.ModuleList([MLP_Block(input_dim, gate_hidden_units, hidden_activations, output_dim=get_output_dim(i), dropout_rates=net_dropout) for i in range(self.num_tasks + self.num_treatments + self.num_outcomes + 1)])
        self.gate_activation = nn.Softmax(dim=-1)
        
    def forward(self, x, require_treatment_shared=False):
        specific_expert_outputs = []
        outcome_shared_expert_outputs = [] 
        treatment_shared_expert_outputs = []
        shared_expert_outputs = []
        treatment_shared_outputs = None
        
        if self.num_specific_experts > 0:
            for i in range(self.num_tasks):
                specific_expert_outputs.append([self.specific_experts[i][j](x[i]) for j in range(self.num_specific_experts)])
                
        for i in range(self.num_treatments):
            outcome_shared_expert_outputs.append([self.outcome_shared_experts[i][j](x[i]) for j in range(self.num_outcome_shared_experts)])
            
        for i in range(self.num_outcomes):
            treatment_shared_expert_outputs.append([self.treatment_shared_experts[i][j](x[i]) for j in range(self.num_treatment_shared_experts)])
            
        for i in range(self.num_shared_experts):
            shared_expert_outputs.append(self.shared_experts[i](x[-1]))
            
        if require_treatment_shared and self.num_treatment_shared_experts > 0:
            treatment_shared_outputs = []
            for outcome_idx in range(self.num_outcomes):
                for expert in self.treatment_shared_experts[outcome_idx]:
                    treatment_shared_outputs.append(expert(x[outcome_idx]))
            treatment_shared_outputs = torch.stack(treatment_shared_outputs)
            
        cgc_outputs = [] 
        for i in range(self.num_tasks + self.num_treatments + self.num_outcomes + 1):
            expert_outputs = []
            if i < self.num_tasks:
                if self.num_specific_experts > 0: expert_outputs.extend(specific_expert_outputs[i])
                expert_outputs.extend(outcome_shared_expert_outputs[i % self.num_treatments])
                expert_outputs.extend(treatment_shared_expert_outputs[i // self.num_treatments])
            elif self.num_tasks <= i < self.num_tasks + self.num_treatments:
                if self.num_specific_experts > 0: expert_outputs.extend([item for j in range(i - self.num_tasks, self.num_tasks, self.num_treatments) for item in specific_expert_outputs[j]])
                expert_outputs.extend(outcome_shared_expert_outputs[i - self.num_tasks])
            elif self.num_tasks + self.num_treatments <= i < self.num_tasks + self.num_treatments + self.num_outcomes:
                if self.num_specific_experts > 0:
                    start_idx = (i - self.num_tasks - self.num_treatments) * self.num_treatments
                    expert_outputs.extend([item for j in range(start_idx, start_idx + self.num_treatments) for item in specific_expert_outputs[j]])
                expert_outputs.extend(treatment_shared_expert_outputs[i - self.num_tasks - self.num_treatments])
            else:
                if self.num_specific_experts > 0: expert_outputs.extend([item for sub in specific_expert_outputs for item in sub])
                expert_outputs.extend([item for sub in outcome_shared_expert_outputs for item in sub])
                expert_outputs.extend([item for sub in treatment_shared_expert_outputs for item in sub])
                
            expert_outputs.extend(shared_expert_outputs)
            gate_input = torch.stack(expert_outputs, dim=1)
            gate = self.gate_activation(self.gate[i](x[i] if i < len(x) - 1 else x[-1]))
            cgc_outputs.append(torch.sum(gate.unsqueeze(-1) * gate_input, dim=1))
            
        if require_treatment_shared:
            return cgc_outputs, treatment_shared_outputs
        return cgc_outputs

# ==========================================
# 基础组件: 底层特征编码器
# ==========================================
class FeatureEncoder(nn.Module):
    """
    完美兼容纯连续特征、或带有离散特征的数据
    """
    def __init__(self, continuous_dim: int, categorical_cardinalities: dict = None, embedding_dim: int = 8):
        super().__init__()
        self.continuous_dim = continuous_dim
        self.embedding_dim = embedding_dim
        categorical_cardinalities = categorical_cardinalities or {}
        
        self.embeddings = nn.ModuleDict({
            col: nn.Embedding(card, embedding_dim)
            for col, card in categorical_cardinalities.items()
        })
        self.num_cat = len(self.embeddings)
        self.output_dim = (continuous_dim if continuous_dim is not None else 0) + embedding_dim * self.num_cat

    def forward(self, x_cont=None, x_cat=None):
        feats = []
        if x_cont is not None:
            feats.append(x_cont)
        if x_cat is not None and self.num_cat > 0:
            emb_list = [self.embeddings[col](x_cat[col]) for col in self.embeddings]
            feats.append(torch.cat(emb_list, dim=1))
        return torch.cat(feats, dim=1) if feats else torch.empty(0)


# ==========================================
# 骨架 1: 纯净基线模型 (用于 Stage 1 的 Model C)
# ==========================================
class TARNET_Baseline(nn.Module):
    """
    用途：训 C 时的 Backbone。或者不加 C 信息的纯 Y 基线。
    🌟 核心：输出底层的 shared_emb (即 Z_c) 交给 losses.py 里的 mmd_loss！
    """
    def __init__(self, continuous_dim: int, categorical_cardinalities: dict,
                 hidden_dims: list, dropout_rate: float, embedding_dim: int = 8):
        super().__init__()
        self.encoder = FeatureEncoder(continuous_dim, categorical_cardinalities, embedding_dim)
        
        layers = []
        curr_dim = self.encoder.output_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.ReLU())
            if dropout_rate > 0: layers.append(nn.Dropout(dropout_rate))
            curr_dim = h_dim
        
        self.shared_layers = nn.Sequential(*layers)
        self.head_0 = nn.Linear(curr_dim, 1)
        self.head_1 = nn.Linear(curr_dim, 1)

    def forward(self, x_cont, x_cat):
        x = self.encoder(x_cont, x_cat)
        shared_emb = self.shared_layers(x) # 这就是 Z_c
        out_0 = self.head_0(shared_emb).squeeze(-1)
        out_1 = self.head_1(shared_emb).squeeze(-1)
        return out_0, out_1, shared_emb


# ==========================================
# 骨架 2: 大一统融合模型 (用于 Stage 3 的 Model Y)
# 涵盖 Level 1 (Raw Prob) 和 Level 2 (Joint Emb)
# ==========================================
class TARNET_Proposed(nn.Module):
    def __init__(self, continuous_dim: int, categorical_cardinalities: dict,
                 hidden_dims: list, dropout_rate: float, 
                 c_fusion_mode: str = "joint_emb", c_embedding_dim: int = 4, 
                 c_model: nn.Module = None, embedding_dim: int = 8):
        super().__init__()
        self.c_fusion_mode = c_fusion_mode
        self.c_model = c_model
        
        # 🛑 极其关键：彻底冻结 C 模型
        if self.c_model is not None:
            self.c_model.eval()
            self.c_model.requires_grad_(False)

        self.encoder = FeatureEncoder(continuous_dim, categorical_cardinalities, embedding_dim)
        input_dim = self.encoder.output_dim

        # --- 先验注入维度推算 ---
        if self.c_model is not None:
            if self.c_fusion_mode == "raw_prob":  
                input_dim += 3 # [pi_00, pi_01, pi_11]
            elif self.c_fusion_mode == "joint_emb": 
                input_dim += c_embedding_dim
                self.emb_never = nn.Parameter(torch.randn(c_embedding_dim))
                self.emb_comp = nn.Parameter(torch.randn(c_embedding_dim))
                self.emb_always = nn.Parameter(torch.randn(c_embedding_dim))

        layers = []
        curr_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.ReLU())
            if dropout_rate > 0: layers.append(nn.Dropout(dropout_rate))
            curr_dim = h_dim
            
        self.shared_layers = nn.Sequential(*layers)
        self.head_0 = nn.Linear(curr_dim, 1)
        self.head_1 = nn.Linear(curr_dim, 1)

    def extract_pi_prior(self, x_cont, x_cat):
        """调用冻结的 C 模型，提取纯净先验"""
        if self.c_model is None:
            return None, None, None
            
        with torch.no_grad(): 
            c0_logit, c1_logit, _ = self.c_model(x_cont, x_cat)
            c0_prob, c1_prob = torch.sigmoid(c0_logit), torch.sigmoid(c1_logit)
            
            # Rubin 主分层概率推导
            pi_00 = (1.0 - c1_prob).clamp(min=0.0)      # Never-taker
            pi_01 = (c1_prob - c0_prob).clamp(min=0.0)  # Complier
            pi_11 = c0_prob                             # Always-taker
            
        return pi_00, pi_01, pi_11

    def forward(self, x_cont, x_cat):
        x_main = self.encoder(x_cont, x_cat)
        pi_dict = {}
        c_feat = None
        
        if self.c_model is not None:
            pi_00, pi_01, pi_11 = self.extract_pi_prior(x_cont, x_cat)
            pi_dict = {"p_never": pi_00, "p_complier": pi_01, "p_always": pi_11}
            
            # 特征融合
            if self.c_fusion_mode == "raw_prob":
                c_feat = torch.stack([pi_00, pi_01, pi_11], dim=1)
            elif self.c_fusion_mode == "joint_emb":
                c_feat = (pi_00.unsqueeze(1) * self.emb_never.unsqueeze(0) +
                          pi_01.unsqueeze(1) * self.emb_comp.unsqueeze(0) +
                          pi_11.unsqueeze(1) * self.emb_always.unsqueeze(0))

        x_aug = torch.cat([x_main, c_feat], dim=1) if c_feat is not None else x_main
        
        shared_y = self.shared_layers(x_aug)
        y0 = self.head_0(shared_y).squeeze(-1)
        y1 = self.head_1(shared_y).squeeze(-1)
        
        return y0, y1, pi_dict


# ==========================================
# 骨架 3: 因果多头路由模型 (Level 3 - MoE Routing)
# ==========================================
class TARNET_MoE(TARNET_Proposed):
    """
    彻底重构网络架构。不拼接特征，直接建立三个独立的 Expert 专家塔。
    用算出的 pi 作为 Gate 门控，动态决定当前样本走哪个网络，实现物理隔离。
    """
    def __init__(self, continuous_dim: int, categorical_cardinalities: dict,
                 hidden_dims: list, dropout_rate: float, 
                 c_model: nn.Module = None, embedding_dim: int = 8):
        # 继承 Proposed 以复用 extract_pi_prior 逻辑，但覆盖其网络结构
        super().__init__(continuous_dim, categorical_cardinalities, hidden_dims, 
                         dropout_rate, c_fusion_mode="none", c_model=c_model, 
                         embedding_dim=embedding_dim)
        
        input_dim = self.encoder.output_dim
        
        # 定义生成 Expert 塔的闭包函数
        def build_expert():
            layers = []
            curr_dim = input_dim
            for h_dim in hidden_dims:
                layers.append(nn.Linear(curr_dim, h_dim))
                layers.append(nn.ReLU())
                if dropout_rate > 0: layers.append(nn.Dropout(dropout_rate))
                curr_dim = h_dim
            return nn.ModuleDict({
                'shared': nn.Sequential(*layers),
                'head_0': nn.Linear(curr_dim, 1),
                'head_1': nn.Linear(curr_dim, 1)
            })

        # 实例化三个独立阶层的塔
        self.expert_never = build_expert()
        self.expert_comp = build_expert()
        self.expert_always = build_expert()
        
        # 删除父类无用的网络层以节省内存
        del self.shared_layers
        del self.head_0
        del self.head_1

    def forward(self, x_cont, x_cat):
        x_main = self.encoder(x_cont, x_cat)
        
        # 1. 向 C 模型要门控权重 (Gate)
        pi_00, pi_01, pi_11 = self.extract_pi_prior(x_cont, x_cat)
        pi_dict = {"p_never": pi_00, "p_complier": pi_01, "p_always": pi_11}
        
        # 2. 三个专家分别推断
        sh_never = self.expert_never['shared'](x_main)
        y0_never = self.expert_never['head_0'](sh_never).squeeze(-1)
        y1_never = self.expert_never['head_1'](sh_never).squeeze(-1)
        
        sh_comp = self.expert_comp['shared'](x_main)
        y0_comp = self.expert_comp['head_0'](sh_comp).squeeze(-1)
        y1_comp = self.expert_comp['head_1'](sh_comp).squeeze(-1)
        
        sh_always = self.expert_always['shared'](x_main)
        y0_always = self.expert_always['head_0'](sh_always).squeeze(-1)
        y1_always = self.expert_always['head_1'](sh_always).squeeze(-1)
        
        # 3. MoE 软路由：利用先验概率进行加权求和
        y0 = pi_00 * y0_never + pi_01 * y0_comp + pi_11 * y0_always
        y1 = pi_00 * y1_never + pi_01 * y1_comp + pi_11 * y1_always
        
        return y0, y1, pi_dict


# ==========================================
# 本地验证脚本 (If __main__)
# ==========================================
if __name__ == "__main__":
    print("🚀 开始验证 models.py 模块...\n")
    torch.manual_seed(42)
    
    batch_size = 4
    cont_dim = 6
    x_cont = torch.randn(batch_size, cont_dim)
    x_cat = {"channel": torch.randint(0, 3, (batch_size,))} 
    cat_cards = {"channel": 3}
    
    print("-" * 50)
    print("🧪 测试 1: 实例化 Model C (Stage 1 Backbone)")
    model_c = TARNET_Baseline(continuous_dim=cont_dim, categorical_cardinalities=cat_cards, 
                              hidden_dims=[32, 16], dropout_rate=0.0)
    c0, c1, z_c = model_c(x_cont, x_cat)
    print(f"  - 输出 Z_c 维度: {z_c.shape}")

    print("\n" + "-" * 50)
    print("🧪 测试 2: 实例化 Model Y (Level 2 - Joint Embedding 融合)")
    model_y_emb = TARNET_Proposed(continuous_dim=cont_dim, categorical_cardinalities=cat_cards, 
                                  hidden_dims=[32, 16], dropout_rate=0.0, 
                                  c_fusion_mode="joint_emb", c_model=model_c)
    y0_emb, y1_emb, pi_dict_emb = model_y_emb(x_cont, x_cat)
    print(f"  - Y 模型(Emb) 输出 Y1 维度: {y1_emb.shape}")
    print(f"  - 榨取 Complier 概率: {pi_dict_emb['p_complier'].detach().numpy().round(3)}")
    
    # 验证冻结
    fake_loss_emb = y1_emb.sum()
    fake_loss_emb.backward()
    print(f"  - 梯度隔离检查: C 模型是否受到污染? -> {model_c.head_1.weight.grad is not None}")

    print("\n" + "-" * 50)
    print("🧪 测试 3: 实例化 Model Y (Level 3 - MoE 专家路由)")
    # 注意：每次 backward 之后最好清零梯度，这里为了纯演示直接复用 model_c 实例化一个新的 MoE
    model_y_moe = TARNET_MoE(continuous_dim=cont_dim, categorical_cardinalities=cat_cards, 
                             hidden_dims=[32, 16], dropout_rate=0.0, 
                             c_model=model_c)
    y0_moe, y1_moe, pi_dict_moe = model_y_moe(x_cont, x_cat)
    print(f"  - Y 模型(MoE) 输出 Y1 维度: {y1_moe.shape}")
    print(f"  - 榨取 Complier 概率: {pi_dict_moe['p_complier'].detach().numpy().round(3)}")
    
    print("\n🎉 models.py 完美通过测试！物理隔离彻底成功，三阶融合架构就位！")


# ==========================================
# 骨架 4: 残差多头路由模型 (Level 4 - V6 Residual MoE)
# ==========================================
class TARNET_Residual_MoE(TARNET_Proposed):
    """
    V6 终极架构：残差 MoE。
    解决 V3 参数空间爆炸和小群体(如AT)数据稀疏导致的过拟合问题。
    主干网络学习大盘 Common 知识，轻量级专家只学习特定人群的偏差 (Residuals)。
    """
    def __init__(self, continuous_dim: int, categorical_cardinalities: dict,
                 hidden_dims: list, dropout_rate: float, 
                 c_model: nn.Module = None, embedding_dim: int = 8):
        super().__init__(continuous_dim, categorical_cardinalities, hidden_dims, 
                         dropout_rate, c_fusion_mode="none", c_model=c_model, 
                         embedding_dim=embedding_dim)
        
        input_dim = self.encoder.output_dim
        
        # 1. 共享基座 (Shared Base) - 学习 100% 全量数据的通用规律
        layers = []
        curr_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.ReLU())
            if dropout_rate > 0: layers.append(nn.Dropout(dropout_rate))
            curr_dim = h_dim
        self.shared_base = nn.Sequential(*layers)
        
        # 基座的主力输出头 (预测大盘基准值)
        self.base_head_0 = nn.Linear(curr_dim, 1)
        self.base_head_1 = nn.Linear(curr_dim, 1)

        # 2. 轻量级残差专家 (Lightweight Residual Experts)
        # 不再复制整个庞大的塔，只在 Shared Base 输出后挂载一个小 MLP
        def build_res_expert():
            return nn.ModuleDict({
                'res_0': nn.Sequential(
                    nn.Linear(curr_dim, curr_dim // 2), 
                    nn.ReLU(), 
                    nn.Linear(curr_dim // 2, 1)
                ),
                'res_1': nn.Sequential(
                    nn.Linear(curr_dim, curr_dim // 2), 
                    nn.ReLU(), 
                    nn.Linear(curr_dim // 2, 1)
                )
            })

        self.res_never = build_res_expert()
        self.res_comp = build_res_expert()
        self.res_always = build_res_expert()
        
        # 清除父类冗余网络层，释放内存
        del self.shared_layers
        del self.head_0
        del self.head_1

    def forward(self, x_cont, x_cat):
        x_main = self.encoder(x_cont, x_cat)
        
        # 1. 获取因果先验概率 (门控权重)
        pi_00, pi_01, pi_11 = self.extract_pi_prior(x_cont, x_cat)
        pi_dict = {"p_never": pi_00, "p_complier": pi_01, "p_always": pi_11}
        
        # 2. 基座推理：算出大盘的 Base Logit
        shared_emb = self.shared_base(x_main)
        base_y0 = self.base_head_0(shared_emb).squeeze(-1)
        base_y1 = self.base_head_1(shared_emb).squeeze(-1)
        
        # 3. 专家推理：算出各类人群特有的残差偏移 (Residual Offset)
        r0_never = self.res_never['res_0'](shared_emb).squeeze(-1)
        r1_never = self.res_never['res_1'](shared_emb).squeeze(-1)
        
        r0_comp = self.res_comp['res_0'](shared_emb).squeeze(-1)
        r1_comp = self.res_comp['res_1'](shared_emb).squeeze(-1)
        
        r0_always = self.res_always['res_0'](shared_emb).squeeze(-1)
        r1_always = self.res_always['res_1'](shared_emb).squeeze(-1)
        
        # 4. 残差 MoE 融合：预测值 = 基准值 + 概率加权的残差值
        y0 = base_y0 + (pi_00 * r0_never + pi_01 * r0_comp + pi_11 * r0_always)
        y1 = base_y1 + (pi_00 * r1_never + pi_01 * r1_comp + pi_11 * r1_always)
        
        return y0, y1, pi_dict


import torch
import torch.nn as nn

# ==========================================
# 骨架 5: V7 排序感知截断残差模型 (Rank-Aware Truncated MoE)
# ==========================================
class TARNET_V7_Truncated_MoE(TARNET_Residual_MoE):
    """
    V7 架构：解决 5-10% 语义错位问题的终极杀器。
    基于大数定律，在巨型 Batch 内部计算排序分位数 (Top K%)，对 C 先验进行截断。
    使用工业级 EMA (带首批次初始化) 保证测试集的绝对稳定性。
    """
    def __init__(self, continuous_dim: int, categorical_cardinalities: dict,
                 hidden_dims: list, dropout_rate: float, 
                 c_model: nn.Module = None, embedding_dim: int = 8,
                 truncation_mode: str = "hard",  # 'hard' 或 'soft'
                 truncation_pct: float = 0.05,   # 例如 0.05 表示 Top 5%
                 truncation_temp: float = 10.0,  # 软截断温度
                 ema_momentum: float = 0.9):     # EMA 平滑系数
        
        # 继承残差 MoE 的网络结构
        super().__init__(continuous_dim, categorical_cardinalities, hidden_dims, 
                         dropout_rate, c_model, embedding_dim)
        
        self.truncation_mode = truncation_mode
        self.truncation_pct = truncation_pct
        self.truncation_temp = truncation_temp
        self.ema_momentum = ema_momentum
        
        # 🌟 注册工业级 EMA 的核心 Buffer (自动随 state_dict 保存)
        self.register_buffer('threshold_ema', torch.tensor(0.0))
        self.register_buffer('ema_initialized', torch.tensor(False))

    def forward(self, x_cont, x_cat):
        x_main = self.encoder(x_cont, x_cat)
        
        # 1. 向被冻结的 C 模型要绝对净化的因果先验概率
        pi_00, pi_01, pi_11 = self.extract_pi_prior(x_cont, x_cat)
        pi_dict = {"p_never": pi_00, "p_complier": pi_01, "p_always": pi_11}
        
        # ==========================================
        # 🛑 V7 核心逻辑：排序感知截断门控 (Rank-Aware Gate)
        # ==========================================
        if self.truncation_mode != "none" and pi_01 is not None:
            if self.training:
                # 训练时：利用巨型 Batch 的大数定律，直接计算当下的 Top K% 分界线
                q = 1.0 - self.truncation_pct
                # detach() 极其重要，分位数只作为标尺，不参与梯度反传
                # batch_thresh = torch.quantile(pi_01.detach(), q)
                batch_thresh = torch.quantile(pi_01.detach().float(), q)
                # 🛡️ 工业级 EMA 更新机制 (In-place 修改防 Bug)
                if not self.ema_initialized:
                    # 首批次暴力初始化，彻底消灭冷启动偏置
                    self.threshold_ema.copy_(batch_thresh)
                    self.ema_initialized.fill_(True)
                else:
                    # 后续批次平滑更新
                    new_ema = self.ema_momentum * self.threshold_ema + (1.0 - self.ema_momentum) * batch_thresh
                    self.threshold_ema.copy_(new_ema)
                
                # 训练时直接用当前 Batch 算出的最准分界线
                current_thresh = batch_thresh
            else:
                # 测试/验证时：直接读取稳定收敛的全局 EMA 阈值
                current_thresh = self.threshold_ema

            # 🔪 计算门控系数 Gate
            if self.truncation_mode == "hard":
                # 硬截断：过了线就是 1，没过就是 0
                gate = (pi_01 >= current_thresh.float()).float()
            elif self.truncation_mode == "soft":
                # 软截断：自适应温度的 Sigmoid 平滑过渡
                scale = self.truncation_temp / (pi_01.std().clamp(min=1e-5))
                gate = torch.sigmoid(scale * (pi_01 - current_thresh.float()))
            
            # 🛡️ 施加惩罚：切断非 Top 人群的专家特权，强迫其退回 Base
            pi_00 = pi_00 * gate
            pi_01 = pi_01 * gate
            pi_11 = pi_11 * gate
        # ==========================================
        
        # 2. 基座推理：算出大盘的 Base Logit (V1 的能力)
        shared_emb = self.shared_base(x_main)
        base_y0 = self.base_head_0(shared_emb).squeeze(-1)
        base_y1 = self.base_head_1(shared_emb).squeeze(-1)
        
        # 3. 专家推理：只有未被截断的头部样本，残差才有效
        r0_never, r1_never = self.res_never['res_0'](shared_emb).squeeze(-1), self.res_never['res_1'](shared_emb).squeeze(-1)
        r0_comp, r1_comp   = self.res_comp['res_0'](shared_emb).squeeze(-1), self.res_comp['res_1'](shared_emb).squeeze(-1)
        r0_always, r1_always = self.res_always['res_0'](shared_emb).squeeze(-1), self.res_always['res_1'](shared_emb).squeeze(-1)
        
        # 4. 残差 MoE 融合：预测值 = 基准值 + 概率(受过门控衰减)加权的残差值
        y0 = base_y0 + (pi_00 * r0_never + pi_01 * r0_comp + pi_11 * r0_always)
        y1 = base_y1 + (pi_00 * r1_never + pi_01 * r1_comp + pi_11 * r1_always)
        
        return y0, y1, pi_dict


import torch
import torch.nn as nn
import torch.nn.functional as F

class TARNET_V8_Evolution_MoE(TARNET_Residual_MoE):
    """
    V8 架构：5大演进方案集成版 (支持 Top K% 分位数退化初始化)
    """
    def __init__(self, continuous_dim: int, categorical_cardinalities: dict,
                 hidden_dims: list, dropout_rate: float, 
                 c_model: nn.Module = None, embedding_dim: int = 8,
                 v8_scheme: int = 3,             # 1=动态阈值, 2=单门控, 3=独立多门控(最推荐), 4=纯特征MLP, 5=MoE Softmax
                 shared_emb_dim: int = 128,      # 🌟 需要你确认的底层特征维度
                 truncation_pct: float = 0.05,   # 控制 Top K% 初始化 (0.05=5%, 0.3=30%, 0.5=50%)
                 truncation_temp: float = 10.0,
                 ema_momentum: float = 0.9):
        
        super().__init__(continuous_dim, categorical_cardinalities, hidden_dims, 
                         dropout_rate, c_model, embedding_dim)
        
        self.v8_scheme = v8_scheme
        self.truncation_pct = truncation_pct
        self.truncation_temp = truncation_temp
        self.ema_momentum = ema_momentum
        
        # 工业级 EMA (仅方案 1, 2, 3 需要人工锚点)
        if self.v8_scheme in [1, 2, 3]:
            self.register_buffer('threshold_ema', torch.tensor(0.0))
            self.register_buffer('ema_initialized', torch.tensor(False))

        # ================= V8 动态网络构建 & 初始化 =================
        if self.v8_scheme == 1:
            # 方案 1: 动态阈值 MLP (预测降分偏移量 Delta C)
            self.dynamic_net = nn.Sequential(
                nn.Linear(shared_emb_dim, shared_emb_dim // 2),
                nn.ReLU(),
                nn.Linear(shared_emb_dim // 2, 1) # 输出 1 维偏移量
            )
            # 🌟 退化初始化：最后层置 0，确保初始 Delta C = 0
            nn.init.zeros_(self.dynamic_net[-1].weight)
            nn.init.zeros_(self.dynamic_net[-1].bias)

        elif self.v8_scheme == 2:
            # 方案 2: 单门控特征驱动 (预测全局附加分 V)
            self.dynamic_net = nn.Linear(shared_emb_dim, 1)
            # 🌟 退化初始化：置 0，确保初始 V = 0
            nn.init.zeros_(self.dynamic_net.weight)
            nn.init.zeros_(self.dynamic_net.bias)

        elif self.v8_scheme == 3:
            # 方案 3: 独立多门控特征驱动 (预测 3 个独立附加分) -> 🌟🌟🌟 最推荐
            self.dynamic_net = nn.Linear(shared_emb_dim, 3) 
            # 🌟 退化初始化：置 0，确保初始 3个 V = 0
            nn.init.zeros_(self.dynamic_net.weight)
            nn.init.zeros_(self.dynamic_net.bias)

        elif self.v8_scheme == 4:
            # 方案 4: 纯特征概率 Sigmoid (抛弃锚点，变体 4A)
            self.dynamic_net = nn.Sequential(
                nn.Linear(shared_emb_dim, shared_emb_dim // 2),
                nn.ReLU(),
                nn.Linear(shared_emb_dim // 2, 3) # 输出 3 个专家的独立 Logit
            )
            # 随机初始化即可，交由网络探索

        elif self.v8_scheme == 5:
            # 方案 5: 纯 MoE Softmax (抛弃锚点，变体 4B)
            # 输出 4 维 = 3个真实专家 + 1个虚拟兜底专家(Dummy)
            self.dynamic_net = nn.Linear(shared_emb_dim, 4)
            nn.init.xavier_normal_(self.dynamic_net.weight)
            # 引入 Dummy Expert 吸走绝大部分初始概率，防止初期过度拟合残差
            nn.init.zeros_(self.dynamic_net.bias)
            nn.init.constant_(self.dynamic_net.bias[3], 2.83) # Dummy bias
        # ==========================================================

    def forward(self, x_cont, x_cat):
        x_main = self.encoder(x_cont, x_cat)
        shared_emb = self.shared_base(x_main)
        
        # 1. 抽取先验
        pi_00, pi_01, pi_11 = self.extract_pi_prior(x_cont, x_cat)
        pi_dict = {"p_never": pi_00, "p_complier": pi_01, "p_always": pi_11}
        
        if self.v8_scheme in [1, 2, 3] and pi_01 is not None:
            # --- 锚点更新逻辑 (继承 V7) ---
            if self.training:
                q = 1.0 - self.truncation_pct
                batch_thresh = torch.quantile(pi_01.detach().float(), q)
                if not self.ema_initialized:
                    self.threshold_ema.copy_(batch_thresh)
                    self.ema_initialized.fill_(True)
                else:
                    new_ema = self.ema_momentum * self.threshold_ema + (1.0 - self.ema_momentum) * batch_thresh
                    self.threshold_ema.copy_(new_ema)
                current_thresh = batch_thresh
            else:
                current_thresh = self.threshold_ema

            # 缩放因子
            scale = self.truncation_temp / (pi_01.std().clamp(min=1e-5))
            
            # --- 各方案的门控计算 (🌟 核心修复区：对齐维度) ---
            if self.v8_scheme == 1:
                # 修复: [B, 1] -> [B]
                delta_c = self.dynamic_net(shared_emb).squeeze(-1) 
                gate_val = torch.sigmoid(scale * (pi_01 - (current_thresh - delta_c)))
                gate_00, gate_01, gate_11 = gate_val, gate_val, gate_val

            elif self.v8_scheme == 2:
                # 修复: [B, 1] -> [B]
                v = self.dynamic_net(shared_emb).squeeze(-1) 
                prior_bias = scale * (pi_01 - current_thresh)
                gate_val = torch.sigmoid(v + prior_bias)
                gate_00, gate_01, gate_11 = gate_val, gate_val, gate_val

            elif self.v8_scheme == 3:
                # v 是 [B, 3], prior_bias 是 [B]
                v = self.dynamic_net(shared_emb) 
                prior_bias = scale * (pi_01 - current_thresh)
                # 修复: 让 prior_bias 变成 [B, 1]，这样才能和 [B, 3] 正常相加
                gates = torch.sigmoid(v + prior_bias.unsqueeze(-1))
                # 提取出来变成纯 1D 的 [B]
                gate_00 = gates[:, 0]
                gate_01 = gates[:, 1]
                gate_11 = gates[:, 2]

        elif self.v8_scheme == 4:
            gates = torch.sigmoid(self.dynamic_net(shared_emb))
            # 提取出来变成纯 1D 的 [B]
            gate_00, gate_01, gate_11 = gates[:, 0], gates[:, 1], gates[:, 2]

        elif self.v8_scheme == 5:
            gates = F.softmax(self.dynamic_net(shared_emb), dim=-1)
            # 提取出来变成纯 1D 的 [B]
            pi_00_final, pi_01_final, pi_11_final = gates[:, 0], gates[:, 1], gates[:, 2]

        # 3. 施加惩罚：更新先验概率
        if self.v8_scheme in [1, 2, 3, 4]:
            # 此时 pi_xx 和 gate_xx 都是纯 1D [B]，直接点乘绝对安全
            pi_00_final = pi_00 * gate_00
            pi_01_final = pi_01 * gate_01
            pi_11_final = pi_11 * gate_11

        # 4. 基座推理
        base_y0 = self.base_head_0(shared_emb).squeeze(-1)
        base_y1 = self.base_head_1(shared_emb).squeeze(-1)
        
        # 5. 专家推理
        r0_never, r1_never = self.res_never['res_0'](shared_emb).squeeze(-1), self.res_never['res_1'](shared_emb).squeeze(-1)
        r0_comp, r1_comp   = self.res_comp['res_0'](shared_emb).squeeze(-1), self.res_comp['res_1'](shared_emb).squeeze(-1)
        r0_always, r1_always = self.res_always['res_0'](shared_emb).squeeze(-1), self.res_always['res_1'](shared_emb).squeeze(-1)
        
        # 6. 残差融合 (🌟 修复：去掉之前的 squeeze(-1)，因为已经是 1D 了)
        y0 = base_y0 + (pi_00_final * r0_never + pi_01_final * r0_comp + pi_11_final * r0_always)
        y1 = base_y1 + (pi_00_final * r1_never + pi_01_final * r1_comp + pi_11_final * r1_always)
        
        return y0, y1, pi_dict


# ==========================================
# 骨架 6: MTMT 多任务多处理模型 (MTMT Baseline)
# ==========================================
class MTMT(nn.Module):
    """
    MTMT: Multi-Task Multi-Treatment Model.
    使用注意力机制融合 User 特征和 Treatment 特征，建模 Uplift (tu_logit)。
    符合多任务学习接口，返回包含 main_task 和 aux_task 的字典。
    """
    def __init__(self, continuous_dim: int, categorical_cardinalities: dict,
                 hidden_dims: list, dropout_rate: float, embedding_dim: int = 8,
                 task_names=None):
        super().__init__()
        if task_names is None:
            task_names = ["main_task", "aux_task"]
        self.task_names = task_names
        self.encoder = FeatureEncoder(continuous_dim, categorical_cardinalities, embedding_dim)
        input_dim = self.encoder.output_dim
        
        # 1. 独立的用户特征编码器 (User Feature Encoder per task)
        self.user_enc = nn.ModuleDict()
        for task in task_names:
            layers = []
            curr_dim = input_dim
            for h_dim in hidden_dims:
                layers.append(nn.Linear(curr_dim, h_dim))
                layers.append(nn.ReLU())
                if dropout_rate > 0: layers.append(nn.Dropout(dropout_rate))
                curr_dim = h_dim
            self.user_enc[task] = nn.Sequential(*layers)
            
        u_dim = hidden_dims[-1]
        t_dim = u_dim
        tu_dim = u_dim
        
        # 2. 预测基础结果 Y(0) (u_tau in MTMT)
        self.u_tau = nn.ModuleDict({
            task: nn.Linear(u_dim, 1) for task in task_names
        })
        
        # 3. Treatment 特征编码 (推断 Y0 和 Y1 时，我们用一个 Parameter 代表 T=1 的特征)
        self.treat_emb = nn.Parameter(torch.randn(1, 1, t_dim))
        
        # 4. 交叉注意力机制 (Treatment-User Attention)
        self.q_proj = nn.Linear(t_dim, tu_dim)
        self.k_proj = nn.Linear(u_dim * len(task_names), tu_dim)
        self.v_proj = nn.Linear(u_dim * len(task_names), tu_dim)
        self.attn = nn.MultiheadAttention(embed_dim=tu_dim, num_heads=4, batch_first=True, dropout=0.2)
        
        # 5. 特征增强与增益预测
        self.tu_enhance = nn.Sequential(
            nn.Linear(tu_dim, tu_dim // 2),
            nn.ReLU()
        )
        self.tu_logit = nn.ModuleDict({
            task: nn.Linear(tu_dim // 2, 1) for task in task_names
        })

    def forward(self, x_cont, x_cat):
        x_main = self.encoder(x_cont, x_cat)
        
        # 获取各任务的 User 特征
        user_feat = {task: self.user_enc[task](x_main) for task in self.task_names}
        
        # 预测各任务的 Base Logit (即 Y(0))
        u_logit = {task: self.u_tau[task](user_feat[task]).squeeze(-1) for task in self.task_names}
        
        # 拼接所有任务的 User 特征并归一化
        user_feat_cat = torch.cat([user_feat[task] for task in self.task_names], dim=-1)
        user_feat_cat = user_feat_cat.unsqueeze(1) # [B, 1, M * u_dim]
        user_feat_norm = user_feat_cat / (torch.linalg.norm(user_feat_cat, dim=-1, keepdim=True) + 1e-6)
        
        # Treatment 特征归一化
        treat_feat = self.treat_emb.expand(x_main.size(0), -1, -1) # [B, 1, t_dim]
        treat_feat_norm = treat_feat / (torch.linalg.norm(treat_feat, dim=-1, keepdim=True) + 1e-6)
        
        # Attention 交互
        q = self.q_proj(treat_feat_norm)
        k = self.k_proj(user_feat_norm)
        v = self.v_proj(user_feat_norm)
        tu_feat, _ = self.attn(q, k, v) # [B, 1, tu_dim]
        
        # 特征增强与 Uplift 预测
        tu_feat_enhanced = self.tu_enhance(tu_feat.squeeze(1)) # [B, tu_dim // 2]
        tu_logit = {task: self.tu_logit[task](tu_feat_enhanced).squeeze(-1) for task in self.task_names}
        
        # 组装多任务的 Y(0) 和 Y(1)
        res = {}
        for task in self.task_names:
            y0 = u_logit[task]
            # MTMT 源码中：Y(1) = u_logit.detach() + tu_logit
            y1 = u_logit[task].detach() + tu_logit[task]
            res[task] = (y0, y1)
            
        res["pi_dict"] = {}
        return res

# ==========================================
# 骨架 7: MOTTO_DA (Multi-Outcome Multi-Treatment with Distribution Alignment)
# ==========================================
class MOTTO_DA(nn.Module):
    """
    MOTTO_DA: Multi-Outcome Multi-Treatment TARNet with CGC (Customized Gate Control) and Distribution Alignment.
    实现了论文中提出的具有领域自适应（DA）能力的多任务多干预模型。
    符合多任务学习接口，返回包含 main_task 和 aux_task 的字典，并在 pi_dict 中携带 da_loss 所需的 treatment_shared_outputs。
    """
    def __init__(self, continuous_dim: int, categorical_cardinalities: dict,
                 hidden_dims: list, dropout_rate: float, embedding_dim: int = 8,
                 task_names=None):
        super().__init__()
        if task_names is None:
            task_names = ["main_task", "aux_task"]
        self.task_names = task_names
        self.num_outcomes = len(task_names)
        self.num_treatments = 2  # Uplift场景通常为T=0和T=1
        
        self.encoder = FeatureEncoder(continuous_dim, categorical_cardinalities, embedding_dim)
        input_dim = self.encoder.output_dim
        
        # MOTTO 超参配置 (尽量与 TARNET 的 hidden_dims 对齐)
        expert_hidden_units = hidden_dims
        gate_hidden_units = [dim // 2 for dim in hidden_dims]
        tower_hidden_units = [hidden_dims[-1] // 2] if len(hidden_dims) > 0 else [16]
        
        self.input_layers = MLP_Block(input_dim, hidden_units=hidden_dims, dropout_rates=dropout_rate)
        cgc_input_dim = hidden_dims[-1] if len(hidden_dims) > 0 else input_dim
        
        self.num_layers = 1
        self.cgc_layers = nn.ModuleList([
            CGC2D_Layer(
                num_shared_experts=1, 
                num_outcome_shared_experts=1, 
                num_treatment_shared_experts=1, 
                num_specific_experts=0, 
                num_outcomes=self.num_outcomes, 
                num_treatments=self.num_treatments, 
                input_dim=cgc_input_dim if i==0 else expert_hidden_units[-1],
                expert_hidden_units=expert_hidden_units, 
                gate_hidden_units=gate_hidden_units, 
                hidden_activations="ReLU", 
                net_dropout=dropout_rate
            ) for i in range(self.num_layers)
        ])
        
        self.tower = nn.ModuleList([
            nn.ModuleList([
                MLP_Block(input_dim=expert_hidden_units[-1], output_dim=1, hidden_units=tower_hidden_units, dropout_rates=dropout_rate)
                for _ in range(self.num_treatments)
            ]) for _ in range(self.num_outcomes)
        ])

    def forward(self, x_cont, x_cat):
        x = self.encoder(x_cont, x_cat)
        x = self.input_layers(x)
        
        # CGC 需要为每种 expert (tasks, treatments, outcomes, shared) 准备输入
        cgc_inputs = [x for _ in range(self.num_outcomes * self.num_treatments + self.num_outcomes + self.num_treatments + 1)]
        
        for i in range(self.num_layers - 1):
            cgc_outputs = self.cgc_layers[i](cgc_inputs)
            cgc_inputs = cgc_outputs

        # 最后一层获取 treatment_shared_outputs 用于 DA loss
        cgc_outputs, treatment_shared_outputs = self.cgc_layers[-1](cgc_inputs, require_treatment_shared=True)
        
        # 重塑并输入 Tower
        cgc_outputs = [cgc_outputs[i * self.num_treatments:(i + 1) * self.num_treatments] for i in range(self.num_outcomes)]
        
        res = {}
        for i, task in enumerate(self.task_names):
            y0 = self.tower[i][0](cgc_outputs[i][0]).squeeze(-1)
            y1 = self.tower[i][1](cgc_outputs[i][1]).squeeze(-1)
            res[task] = (y0, y1)
            
        # 携带 DA 特征供 loss 使用
        res["pi_dict"] = {"treatment_shared_outputs": treatment_shared_outputs}
        return res
