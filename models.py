import torch
import torch.nn as nn

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
# 论文复刻组件 1: 1D ResNet18 (专为 Tabular 特征定制)
# ==========================================
class BasicBlock1D(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock1D, self).__init__()
        # 1D 卷积
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet18_1D(nn.Module):
    def __init__(self, num_classes=64):
        super(ResNet18_1D, self).__init__()
        self.in_planes = 64
        
        # 为了防止表格数据特征过短导致维度崩溃，这里改用 kernel=3, stride=1 的安全起手式
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        
        # 4 个 Stage (对于表格数据，强制 stride=1 防止长度坍缩，靠通道维度提取特征)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=1)
        self.layer3 = self._make_layer(256, 2, stride=1)
        self.layer4 = self._make_layer(512, 2, stride=1)
        
        # 全局池化 & 最终映射
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * BasicBlock1D.expansion, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(BasicBlock1D(self.in_planes, planes, s))
            self.in_planes = planes * BasicBlock1D.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # x: [Batch, Features] -> 增加通道维度 -> [Batch, 1, Features]
        x = x.unsqueeze(1) 
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.adaptive_pool(out)    # [Batch, 512, 1]
        out = out.view(out.size(0), -1)  # [Batch, 512]
        out = self.fc(out)               # [Batch, num_classes]
        return out


# ==========================================
# 论文复刻组件 2: 动态专家生成工厂
# ==========================================
def build_expert(expert_type, input_dim, hidden_dims, dropout_rate=0.1):
    layers = []
    curr_dim = input_dim
    
    if expert_type == "mlp":
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            curr_dim = h_dim
        return nn.Sequential(*layers), curr_dim
        
    elif expert_type == "resnet18":
        # hidden_dims[0] 指定最后输出的专家特征维度
        out_dim = hidden_dims[0] if isinstance(hidden_dims, list) else hidden_dims
        return ResNet18_1D(num_classes=out_dim), out_dim
        
    else:
        raise ValueError(f"❌ 不支持的 expert_type: {expert_type}")


# ==========================================
# 论文复刻组件 3: 多任务 MMoE 层
# ==========================================
class MMoE_Layer(nn.Module):
    def __init__(self, input_dim, num_experts, num_tasks, expert_type="mlp", expert_hidden_dims=[64], dropout_rate=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        
        self.experts = nn.ModuleList()
        for _ in range(num_experts):
            expert_net, self.expert_out_dim = build_expert(expert_type, input_dim, expert_hidden_dims, dropout_rate)
            self.experts.append(expert_net)
        
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, num_experts),
                nn.Softmax(dim=-1)
            ) for _ in range(num_tasks)
        ])

    def forward(self, x):
        # expert_outputs: [B, num_experts, expert_out_dim]
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1) 
        
        task_reps = []
        for i in range(self.num_tasks):
            gate_weights = self.gates[i](x).unsqueeze(-1) # [B, num_experts, 1]
            task_rep = torch.sum(expert_outputs * gate_weights, dim=1) # [B, expert_out_dim]
            task_reps.append(task_rep)
            
        return task_reps


# ==========================================
# 论文复刻组件 4: 干预-特征 Attention 交互模块 (Eq 6)
# ==========================================
class UserTreatmentInteraction(nn.Module):
    def __init__(self, t_dim, u_dim, out_dim):
        super().__init__()
        self.W_t = nn.Linear(t_dim, out_dim)
        self.W_u = nn.Linear(u_dim, out_dim)
        self.W_v = nn.Linear(u_dim, out_dim)
        self.scale = out_dim ** 0.5

    def forward(self, epsilon, phi):
        Q = self.W_t(epsilon) # [B, out_dim]
        K = self.W_u(phi)     # [B, out_dim]
        V = self.W_v(phi)     # [B, out_dim]
        
        # Attention Score
        attn_score = torch.sigmoid((Q * K) / self.scale) 
        psi = attn_score * V # [B, out_dim]
        return psi


# ==========================================
# 🌟 终极组装: MTMT (Multi-Treatment Multi-Task) 论文模型
# ==========================================
class MTMT_STMT(nn.Module):
    """
    MTMT 核心架构 (适配 Single-Treatment 场景)：
    1. 自然响应 y(0) 由 Task 特征 phi 直接预测
    2. 增量响应 tau 由 Treatment 特征与 phi Attention 交互后预测
    3. 最终响应 y(1) = y(0) + tau
    """
    def __init__(self, continuous_dim: int, categorical_cardinalities: dict,
                 num_experts: int = 4, expert_type: str = "mlp", expert_hidden_dims: list = [64], 
                 dropout_rate: float = 0.1, t_emb_dim: int = 16):
        super().__init__()
        
        # 1. 基础特征编码
        self.encoder = FeatureEncoder(continuous_dim, categorical_cardinalities)
        
        # 2. MMoE 拆离任务特征 (论文设定)
        self.mmoe = MMoE_Layer(
            input_dim=self.encoder.output_dim, 
            num_experts=num_experts, 
            num_tasks=2, # 预设为双任务 (主任务 Y, 辅任务 C)
            expert_type=expert_type, 
            expert_hidden_dims=expert_hidden_dims, 
            dropout_rate=dropout_rate
        )
        expert_out_dim = self.mmoe.expert_out_dim
        
        # 3. Base Treatment 显式 Embedding (论文设定)
        self.t_emb = nn.Parameter(torch.randn(1, t_emb_dim)) 
        
        # 4. 主任务 Head (Y)
        self.y0_head_main = nn.Linear(expert_out_dim, 1)
        self.interaction_main = UserTreatmentInteraction(t_emb_dim, expert_out_dim, expert_out_dim)
        self.enhancer_main = nn.Sequential(
            nn.Linear(expert_out_dim, 32), 
            nn.ReLU(), 
            nn.Linear(32, 1)
        )
        
        # 5. 辅助任务 Head (C)
        self.y0_head_aux = nn.Linear(expert_out_dim, 1)
        self.interaction_aux = UserTreatmentInteraction(t_emb_dim, expert_out_dim, expert_out_dim)
        self.enhancer_aux = nn.Sequential(
            nn.Linear(expert_out_dim, 32), 
            nn.ReLU(), 
            nn.Linear(32, 1)
        )

    def forward(self, x_cont, x_cat):
        # -> 特征编码
        x = self.encoder(x_cont, x_cat)
        
        # -> MMoE 分配专属表征
        phi_main, phi_aux = self.mmoe(x)
        
        # -> 预测自然转化 y(0)
        y0_main = self.y0_head_main(phi_main).squeeze(-1)
        y0_aux = self.y0_head_aux(phi_aux).squeeze(-1)
        
        # -> 广播干预 Embedding [1, t_emb_dim] -> [B, t_emb_dim]
        eps = self.t_emb.expand(x.size(0), -1) 
        
        # -> 干预与任务特征交互 (Attention)
        psi_main = self.interaction_main(eps, phi_main)
        psi_aux = self.interaction_aux(eps, phi_aux)
        
        # -> 计算增量 Uplift (tau)
        tau_main = self.enhancer_main(psi_main).squeeze(-1)
        tau_aux = self.enhancer_aux(psi_aux).squeeze(-1)
        
        # -> 叠加得到干预后转化 y(1)
        y1_main = y0_main + tau_main
        y1_aux = y0_aux + tau_aux
        
        # 🌟 返回标准格式，无缝对接 Trainer 字典结构
        return {
            "main_task": (y0_main, y1_main), 
            "aux_task": (y0_aux, y1_aux),    
            "pi_dict": {} # MTMT 没有显式的 P_complier 先验估计，所以留空
        }


# ==========================================
# 🌟 WWW 2024 顶会复刻: ECUP (Entire Chain Uplift) 
# 严格对齐 Section 5.1.2 与 Table 2
# ==========================================
class TENet(nn.Module):
    """Treatment-Enhanced Network (TENet) - 论文 Fig 5"""
    def __init__(self, f_num, d_dim, num_heads=2):
        super().__init__()
        self.f_num = f_num
        self.d_dim = d_dim
        
        # Self-Attention 捕捉交叉特征
        self.self_attn = nn.MultiheadAttention(embed_dim=d_dim, num_heads=num_heads, batch_first=True)
        
        # 🌟 对齐 1: TIE 层数为 1 (the number of layers of treatment information extractor is 1)
        self.tie_mlp_g = nn.Linear(d_dim, d_dim)
        self.tie_mlp_w = nn.Linear(d_dim, d_dim)
        
    def forward(self, E_x, E_tr):
        # E_x: [B, f_num, d_dim], E_tr: [B, 1, d_dim]
        E_att, _ = self.self_attn(E_x, E_x, E_x) 
        
        E_bit_g = self.tie_mlp_g(E_tr) 
        E_bit_w = self.tie_mlp_w(E_tr) 
        
        E_TAU_g = E_att * E_bit_g 
        W_b = E_att * E_bit_w     
        
        gate = torch.sigmoid(W_b)
        E_r = E_x * gate + E_TAU_g * (1 - gate) 
        
        E_r_final = torch.cat([E_r, E_tr], dim=1) # [B, f_num + 1, d_dim]
        return E_r_final

class TAENet(nn.Module):
    """Task-Enhanced Network (TAENet) - 注入任务先验"""
    def __init__(self, d_dim, tae_h, tower_h, num_tasks=2, num_heads=2, gamma=1.0):
        super().__init__()
        self.gamma = gamma
        self.num_tasks = num_tasks
        self.E_ta = nn.Parameter(torch.randn(num_tasks, d_dim)) 
        
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_dim, num_heads=num_heads, batch_first=True)
        
        # 🌟 对齐 2: TAEGate 层数为 2，隐层为 tae_h (h_gate)，输出必须为 tower_h 以便做后续的特征缩放
        self.tae_mlp = nn.Sequential(
            nn.Linear(d_dim, tae_h),
            nn.ReLU(),
            nn.Linear(tae_h, tower_h) # 输出维度严格对齐 Tower 的宽度
        )
        
    def forward(self, E_r_final):
        B = E_r_final.size(0)
        query = self.E_ta.unsqueeze(0).expand(B, -1, -1) # [B, 2, d_dim]
        
        E_pri, _ = self.cross_attn(query, E_r_final, E_r_final) 
        
        # [B, 2, tower_h]
        delta = self.gamma * torch.sigmoid(self.tae_mlp(E_pri))
        return delta

class ECUP_Model(nn.Module):
    """ECUP 终极全链路框架"""
    def __init__(self, continuous_dim: int, categorical_cardinalities: dict, 
                 tower_h: int = 128, tae_h: int = 64, d_dim: int = 16, 
                 num_heads: int = 2, gamma: float = 1.0):
        super().__init__()
        
        # 1. 记录字段信息
        self.d_dim = d_dim
        self.cont_dim = continuous_dim
        self.cat_cards = categorical_cardinalities or {}
        self.f_num = self.cont_dim + len(self.cat_cards)

        # 2. 字段独立投影 (保证公平性与论文 Eq 3 对齐)
        self.cont_projections = nn.ModuleList([
            nn.Linear(1, d_dim) for _ in range(self.cont_dim)
        ])
        self.cat_embeddings = nn.ModuleDict({
            col: nn.Embedding(card, d_dim)
            for col, card in self.cat_cards.items()
        })

        self.t_emb_0 = nn.Parameter(torch.randn(1, 1, d_dim))
        self.t_emb_1 = nn.Parameter(torch.randn(1, 1, d_dim))
        
        self.tenet = TENet(f_num=self.f_num, d_dim=d_dim, num_heads=num_heads)
        
        # 传入 tower_h 给 TAENet，保证门控输出维度能和塔的神经元一对一对齐
        self.taenet = TAENet(d_dim=d_dim, tae_h=tae_h, tower_h=tower_h, 
                             num_tasks=2, num_heads=num_heads, gamma=gamma)
        
        # 🌟 对齐 3: 严格构建 L=3 的 Tower，且前两层宽度固定为 tower_h
        # Layer 1: Shared Layer
        self.shared_layer = nn.Sequential(nn.Linear((self.f_num + 1) * d_dim, tower_h), nn.ReLU())
        
        # Layer 2: Task-specific Layer
        self.ctr_layer2 = nn.Sequential(nn.Linear(tower_h, tower_h), nn.ReLU())
        self.cvr_layer2 = nn.Sequential(nn.Linear(tower_h, tower_h), nn.ReLU())
        
        # Layer 3: Task-specific Output Layer (The Last Layer)
        self.ctr_out = nn.Linear(tower_h, 1)
        self.cvr_out = nn.Linear(tower_h, 1)

    def _get_initial_embeddings(self, x_cont, x_cat):
        B = x_cont.shape[0] if x_cont is not None else next(iter(x_cat.values())).shape[0]
        all_field_embeds = []

        if x_cont is not None:
            for i in range(self.cont_dim):
                feat = x_cont[:, i:i+1]
                all_field_embeds.append(self.cont_projections[i](feat).unsqueeze(1)) 

        if x_cat is not None:
            for col, val in x_cat.items():
                all_field_embeds.append(self.cat_embeddings[col](val).unsqueeze(1)) 

        return torch.cat(all_field_embeds, dim=1)

    def _forward_once(self, x_cont, x_cat, T):
        E_x = self._get_initial_embeddings(x_cont, x_cat) # [B, f_num, d_dim]
        B = E_x.size(0)
        
        E_tr = self.t_emb_1.expand(B, -1, -1) if T == 1 else self.t_emb_0.expand(B, -1, -1)
        
        E_r_final = self.tenet(E_x, E_tr) 
        delta = self.taenet(E_r_final) # [B, 2, tower_h]
        
        # 🌟 对齐 3 (续): 落实 "scale with each layer ... except the last layer"
        # --- Layer 1 ---
        l1_out = self.shared_layer(E_r_final.view(B, -1)) # [B, tower_h]
        
        # --- Layer 2 & Scale ---
        # CTR 分支
        ctr_h1 = l1_out * delta[:, 0, :]               # 缩放第一层输出
        ctr_h2 = self.ctr_layer2(ctr_h1)               # 过第二层
        ctr_h2 = ctr_h2 * delta[:, 0, :]               # 缩放第二层输出
        ctr_logit = self.ctr_out(ctr_h2).squeeze(-1)   # 过最后一层 (不缩放)
        
        # CVR 分支
        cvr_h1 = l1_out * delta[:, 1, :]               # 缩放第一层输出
        cvr_h2 = self.cvr_layer2(cvr_h1)               # 过第二层
        cvr_h2 = cvr_h2 * delta[:, 1, :]               # 缩放第二层输出
        cvr_logit = self.cvr_out(cvr_h2).squeeze(-1)   # 过最后一层 (不缩放)
        
        return ctr_logit, cvr_logit

    def forward(self, x_cont, x_cat):
        c0_logit, cvr0_logit = self._forward_once(x_cont, x_cat, T=0)
        c1_logit, cvr1_logit = self._forward_once(x_cont, x_cat, T=1)
        return {
            "c_logits": (c0_logit, c1_logit),
            "cvr_logits": (cvr0_logit, cvr1_logit),
            "pi_dict": {}
        }




# ==========================================
# 🌟 KDD 2025: MOTTO 
# (已包含：特征公平投影 + 动态维度专家 + SDA对齐)
# ==========================================

class ExpertNetwork(nn.Module):
    """支持多层灵活深度的基础专家网络"""
    def __init__(self, input_dim, hidden_dims):
        super().__init__()
        layers = []
        curr_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.ReLU())
            curr_dim = h_dim
        self.net = nn.Sequential(*layers)
        self.output_dim = curr_dim # 记录最终输出维度，供 Tower 使用

    def forward(self, x):
        return self.net(x)

class MOTTO_Model(nn.Module):
    def __init__(self, continuous_dim: int, categorical_cardinalities: dict, 
                 d_dim: int = 16, bottom_dim: int = 128, 
                 expert_hidden_dims: list = [64, 64], tower_dim: int = 64,
                 use_specific_experts: bool = True):
        super().__init__()
        
        self.use_specific_experts = use_specific_experts
        
        # ==========================================
        # 🌟 缝合点 1：绝对公平的初始特征投影 (借鉴 ECUP 思想)
        # ==========================================
        self.cont_dim = continuous_dim
        self.cat_cards = categorical_cardinalities or {}
        # 对于连续特征：独立的 Linear(1, d_dim)
        self.cont_projections = nn.ModuleList([
            nn.Linear(1, d_dim) for _ in range(self.cont_dim)
        ])
        # 对于离散特征：查表 Embedding
        self.cat_embeddings = nn.ModuleDict({
            col: nn.Embedding(card, d_dim)
            for col, card in self.cat_cards.items()
        })
        
        # 展平后的总维度 = (连续字段数 + 离散字段数) * d_dim
        self.total_feature_dim = (self.cont_dim + len(self.cat_cards)) * d_dim
        
        # ==========================================
        # 共享底层 (Shared Bottom)
        # ==========================================
        self.shared_bottom = nn.Sequential(
            nn.Linear(self.total_feature_dim, bottom_dim),
            nn.ReLU()
        )
        
        # ==========================================
        # 🌟 缝合点 2：动态维度的隔离专家组 (所有专家结构绝对统一)
        # ==========================================
        # 1. 全局共享专家
        self.exp_global = ExpertNetwork(bottom_dim, expert_hidden_dims)
        expert_out_dim = self.exp_global.output_dim # 获取专家的最终输出维度
        
        # 2. 结果共享专家 (T=0, T=1)
        self.exp_out_sh_t0 = ExpertNetwork(bottom_dim, expert_hidden_dims)
        self.exp_out_sh_t1 = ExpertNetwork(bottom_dim, expert_hidden_dims)
        
        # 3. 干预共享专家 (Main, Aux) - 用于 SDA 对齐
        self.exp_trt_sh_main = ExpertNetwork(bottom_dim, expert_hidden_dims)
        self.exp_trt_sh_aux = ExpertNetwork(bottom_dim, expert_hidden_dims)
        
        # 4. 特定专家 (T0-Main, T1-Main, T0-Aux, T1-Aux)
        self.num_experts_per_gate = 3
        if use_specific_experts:
            self.exp_spec_t0_main = ExpertNetwork(bottom_dim, expert_hidden_dims)
            self.exp_spec_t1_main = ExpertNetwork(bottom_dim, expert_hidden_dims)
            self.exp_spec_t0_aux = ExpertNetwork(bottom_dim, expert_hidden_dims)
            self.exp_spec_t1_aux = ExpertNetwork(bottom_dim, expert_hidden_dims)
            self.num_experts_per_gate = 4
            
        # ==========================================
        # 门控网络与预测塔
        # ==========================================
        # Gates: 输出权重 [B, num_experts]
        self.gate_t0_main = nn.Sequential(nn.Linear(bottom_dim, self.num_experts_per_gate), nn.Softmax(dim=-1))
        self.gate_t1_main = nn.Sequential(nn.Linear(bottom_dim, self.num_experts_per_gate), nn.Softmax(dim=-1))
        self.gate_t0_aux = nn.Sequential(nn.Linear(bottom_dim, self.num_experts_per_gate), nn.Softmax(dim=-1))
        self.gate_t1_aux = nn.Sequential(nn.Linear(bottom_dim, self.num_experts_per_gate), nn.Softmax(dim=-1))
        
        # Towers: 接收 expert_out_dim 维度的聚合特征
        def build_tower():
            return nn.Sequential(nn.Linear(expert_out_dim, tower_dim), nn.ReLU(), nn.Linear(tower_dim, 1))
            
        self.tower_t0_main = build_tower()
        self.tower_t1_main = build_tower()
        self.tower_t0_aux = build_tower()
        self.tower_t1_aux = build_tower()

    def _get_fair_embeddings(self, x_cont, x_cat):
        """保证连续与离散特征公平性的初始投影"""
        all_field_embeds = []
        if x_cont is not None:
            for i in range(self.cont_dim):
                feat = x_cont[:, i:i+1]
                all_field_embeds.append(self.cont_projections[i](feat)) # [B, d_dim]
        if x_cat is not None:
            for col, val in x_cat.items():
                all_field_embeds.append(self.cat_embeddings[col](val)) # [B, d_dim]
        # 展平成一维向量，供 Shared Bottom 使用
        return torch.cat(all_field_embeds, dim=1) # [B, f_num * d_dim]

    def forward(self, x_cont, x_cat):
        # 1. 提取底层特征 (经过公平投影与 Shared Bottom)
        x_flat = self._get_fair_embeddings(x_cont, x_cat)
        h_bottom = self.shared_bottom(x_flat)
        
        # 2. 计算所有专家的输出
        e_glob = self.exp_global(h_bottom)
        e_out_t0 = self.exp_out_sh_t0(h_bottom)
        e_out_t1 = self.exp_out_sh_t1(h_bottom)
        e_trt_main = self.exp_trt_sh_main(h_bottom)
        e_trt_aux = self.exp_trt_sh_aux(h_bottom)
        
        if self.use_specific_experts:
            e_sp_t0_main = self.exp_spec_t0_main(h_bottom)
            e_sp_t1_main = self.exp_spec_t1_main(h_bottom)
            e_sp_t0_aux = self.exp_spec_t0_aux(h_bottom)
            e_sp_t1_aux = self.exp_spec_t1_aux(h_bottom)
            
        # 3. 路由与聚合 (Routing & Aggregation)
        # --- (T=0, Main) ---
        exps_t0_main = [e_glob, e_out_t0, e_trt_main] + ([e_sp_t0_main] if self.use_specific_experts else [])
        agg_t0_main = torch.sum(torch.stack(exps_t0_main, dim=1) * self.gate_t0_main(h_bottom).unsqueeze(-1), dim=1)
        y0_main = self.tower_t0_main(agg_t0_main).squeeze(-1)
        
        # --- (T=1, Main) ---
        exps_t1_main = [e_glob, e_out_t1, e_trt_main] + ([e_sp_t1_main] if self.use_specific_experts else [])
        agg_t1_main = torch.sum(torch.stack(exps_t1_main, dim=1) * self.gate_t1_main(h_bottom).unsqueeze(-1), dim=1)
        y1_main = self.tower_t1_main(agg_t1_main).squeeze(-1)
        
        # --- (T=0, Aux) ---
        exps_t0_aux = [e_glob, e_out_t0, e_trt_aux] + ([e_sp_t0_aux] if self.use_specific_experts else [])
        agg_t0_aux = torch.sum(torch.stack(exps_t0_aux, dim=1) * self.gate_t0_aux(h_bottom).unsqueeze(-1), dim=1)
        y0_aux = self.tower_t0_aux(agg_t0_aux).squeeze(-1)
        
        # --- (T=1, Aux) ---
        exps_t1_aux = [e_glob, e_out_t1, e_trt_aux] + ([e_sp_t1_aux] if self.use_specific_experts else [])
        agg_t1_aux = torch.sum(torch.stack(exps_t1_aux, dim=1) * self.gate_t1_aux(h_bottom).unsqueeze(-1), dim=1)
        y1_aux = self.tower_t1_aux(agg_t1_aux).squeeze(-1)

        # 4. 返回预测与 SDA 表征
        return {
            "main_task": (y0_main, y1_main),
            "aux_task": (y0_aux, y1_aux),
            "sda_reps": {"main": e_trt_main, "aux": e_trt_aux},
            "pi_dict": {}
        }



# ==========================================
# 🌟 新增：TARNET Naive Multitask Baseline
# 共享底层 (Shared Bottom) + 4 个独立预测头 (Y0, Y1, C0, C1)
# ==========================================
class TARNET_Naive_MT(nn.Module):
    def __init__(self, continuous_dim, categorical_cardinalities, hidden_dims=[128, 64, 32], dropout_rate=0.1):
        super().__init__()
        self.encoder = FeatureEncoder(continuous_dim, categorical_cardinalities)
        input_dim = self.encoder.output_dim
        
        # 共享表征层 (Shared Bottom)
        layers = []
        curr_dim = input_dim
        for dim in hidden_dims:
            layers.extend([nn.Linear(curr_dim, dim), nn.ReLU(), nn.Dropout(dropout_rate)])
            curr_dim = dim
        self.shared_bottom = nn.Sequential(*layers)
        
        # 4 个独立的靶向预测头
        self.head_y0 = nn.Linear(curr_dim, 1)
        self.head_y1 = nn.Linear(curr_dim, 1)
        self.head_c0 = nn.Linear(curr_dim, 1)
        self.head_c1 = nn.Linear(curr_dim, 1)

    def forward(self, x_cont, x_cat):
        # 1. 提取底层特征
        feat = self.encoder(x_cont, x_cat)
        
        # 2. 共享网络前向传播
        shared_rep = self.shared_bottom(feat)
        
        # 3. 四头分发 (挤压掉最后一个维度，保持与原有 BCE Logits 的兼容)
        y0_logit = self.head_y0(shared_rep).squeeze(-1)
        y1_logit = self.head_y1(shared_rep).squeeze(-1)
        c0_logit = self.head_c0(shared_rep).squeeze(-1)
        c1_logit = self.head_c1(shared_rep).squeeze(-1)
        
        # 4. 构造 pi_dict (符合业务物理直觉)
        p_c0 = torch.sigmoid(c0_logit)
        p_c1 = torch.sigmoid(c1_logit)
        
        pi_dict = {
            "p_complier": torch.clamp(p_c1 - p_c0, min=0.0), # 敏感客的理论概率
            "p_never": torch.clamp(1.0 - p_c1, min=0.0),     # 绝对不响应客
            "p_always": torch.clamp(p_c0, min=0.0)           # 自然转化客
        }
        
        # 5. 打包输出，完美兼容 evaluator 和 loss 引擎
        return {
            "main_task": (y0_logit, y1_logit),
            "aux_task": (c0_logit, c1_logit),
            "pi_dict": pi_dict
        }


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
        # 🌟 白盒埋点：V1/V2 的底层表征
        if not self.training:
            # 构造一个伪 pi_dict 传递白盒数据
            pi_dict = {"wb_shared_emb": shared_emb}
            return out_0, out_1, pi_dict

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
        if not self.training:
            pi_dict["wb_shared_emb"] = shared_y
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
        
        if not self.training:
            # 1. 独立表征
            pi_dict["wb_sh_never"] = sh_never
            pi_dict["wb_sh_comp"] = sh_comp
            pi_dict["wb_sh_always"] = sh_always
            
            # 2. Never-Taker 专家绝对打分
            pi_dict["wb_exp_y0_never"] = y0_never
            pi_dict["wb_exp_y1_never"] = y1_never
            pi_dict["wb_exp_u_never"] = y1_never - y0_never
            
            # 3. Complier 专家绝对打分
            pi_dict["wb_exp_y0_comp"] = y0_comp
            pi_dict["wb_exp_y1_comp"] = y1_comp
            pi_dict["wb_exp_u_comp"] = y1_comp - y0_comp
            
            # 4. Always-Taker 专家绝对打分
            pi_dict["wb_exp_y0_always"] = y0_always
            pi_dict["wb_exp_y1_always"] = y1_always
            pi_dict["wb_exp_u_always"] = y1_always - y0_always
        return y0, y1, pi_dict





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
        
        if not self.training:
            pi_dict["wb_shared_emb"] = shared_emb
            
            # 1. Base 塔打分
            pi_dict["wb_base_y0"] = base_y0
            pi_dict["wb_base_y1"] = base_y1
            pi_dict["wb_base_u"] = base_y1 - base_y0
            
            # 2. Never-Taker 专家残差
            pi_dict["wb_res_y0_never"] = r0_never
            pi_dict["wb_res_y1_never"] = r1_never
            pi_dict["wb_res_u_never"] = r1_never - r0_never
            
            # 3. Complier 专家残差
            pi_dict["wb_res_y0_comp"] = r0_comp
            pi_dict["wb_res_y1_comp"] = r1_comp
            pi_dict["wb_res_u_comp"] = r1_comp - r0_comp
            
            # 4. Always-Taker 专家残差
            pi_dict["wb_res_y0_always"] = r0_always
            pi_dict["wb_res_y1_always"] = r1_always
            pi_dict["wb_res_u_always"] = r1_always - r0_always

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
        if not self.training:
            pi_dict["wb_shared_emb"] = shared_emb
            if 'gate_for_wb' in locals() and gate is not None:
                pi_dict["wb_v7_gate"] = gate
                
            # 🌟 新增对齐：V7 被门控衰减后的最终概率
            pi_dict["wb_final_pi_never"] = pi_00
            pi_dict["wb_final_pi_comp"] = pi_01
            pi_dict["wb_final_pi_always"] = pi_11
                
            pi_dict["wb_base_y0"] = base_y0
            pi_dict["wb_base_y1"] = base_y1
            pi_dict["wb_base_u"] = base_y1 - base_y0
            
            pi_dict["wb_res_y0_never"] = r0_never
            pi_dict["wb_res_y1_never"] = r1_never
            pi_dict["wb_res_u_never"] = r1_never - r0_never
            
            pi_dict["wb_res_y0_comp"] = r0_comp
            pi_dict["wb_res_y1_comp"] = r1_comp
            pi_dict["wb_res_u_comp"] = r1_comp - r0_comp
            
            pi_dict["wb_res_y0_always"] = r0_always
            pi_dict["wb_res_y1_always"] = r1_always
            pi_dict["wb_res_u_always"] = r1_always - r0_always
        return y0, y1, pi_dict


# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class TARNET_V8_Evolution_MoE(TARNET_Residual_MoE):
#     """
#     V8 架构：5大演进方案集成版 (支持 Top K% 分位数退化初始化)
#     """
#     def __init__(self, continuous_dim: int, categorical_cardinalities: dict,
#                  hidden_dims: list, dropout_rate: float, 
#                  c_model: nn.Module = None, embedding_dim: int = 8,
#                  v8_scheme: int = 3,             # 1=动态阈值, 2=单门控, 3=独立多门控(最推荐), 4=纯特征MLP, 5=MoE Softmax
#                  shared_emb_dim: int = 128,      # 🌟 需要你确认的底层特征维度
#                  truncation_pct: float = 0.05,   # 控制 Top K% 初始化 (0.05=5%, 0.3=30%, 0.5=50%)
#                  truncation_temp: float = 10.0,
#                  ema_momentum: float = 0.9):
        
#         super().__init__(continuous_dim, categorical_cardinalities, hidden_dims, 
#                          dropout_rate, c_model, embedding_dim)
        
#         self.v8_scheme = v8_scheme
#         self.truncation_pct = truncation_pct
#         self.truncation_temp = truncation_temp
#         self.ema_momentum = ema_momentum
        
#         # 工业级 EMA (仅方案 1, 2, 3 需要人工锚点)
#         if self.v8_scheme in [1, 2, 3]:
#             self.register_buffer('threshold_ema', torch.tensor(0.0))
#             self.register_buffer('ema_initialized', torch.tensor(False))

#         # ================= V8 动态网络构建 & 初始化 =================
#         if self.v8_scheme == 1:
#             # 方案 1: 动态阈值 MLP (预测降分偏移量 Delta C)
#             self.dynamic_net = nn.Sequential(
#                 nn.Linear(shared_emb_dim, shared_emb_dim // 2),
#                 nn.ReLU(),
#                 nn.Linear(shared_emb_dim // 2, 1) # 输出 1 维偏移量
#             )
#             # 🌟 退化初始化：最后层置 0，确保初始 Delta C = 0
#             nn.init.zeros_(self.dynamic_net[-1].weight)
#             nn.init.zeros_(self.dynamic_net[-1].bias)

#         elif self.v8_scheme == 2:
#             # 方案 2: 单门控特征驱动 (预测全局附加分 V)
#             self.dynamic_net = nn.Linear(shared_emb_dim, 1)
#             # 🌟 退化初始化：置 0，确保初始 V = 0
#             nn.init.zeros_(self.dynamic_net.weight)
#             nn.init.zeros_(self.dynamic_net.bias)

#         elif self.v8_scheme == 3:
#             # 方案 3: 独立多门控特征驱动 (预测 3 个独立附加分) -> 🌟🌟🌟 最推荐
#             self.dynamic_net = nn.Linear(shared_emb_dim, 3) 
#             # 🌟 退化初始化：置 0，确保初始 3个 V = 0
#             nn.init.zeros_(self.dynamic_net.weight)
#             nn.init.zeros_(self.dynamic_net.bias)

#         elif self.v8_scheme == 4:
#             # 方案 4: 纯特征概率 Sigmoid (抛弃锚点，变体 4A)
#             self.dynamic_net = nn.Sequential(
#                 nn.Linear(shared_emb_dim, shared_emb_dim // 2),
#                 nn.ReLU(),
#                 nn.Linear(shared_emb_dim // 2, 3) # 输出 3 个专家的独立 Logit
#             )
#             # 随机初始化即可，交由网络探索

#         elif self.v8_scheme == 5:
#             # 方案 5: 纯 MoE Softmax (抛弃锚点，变体 4B)
#             # 输出 4 维 = 3个真实专家 + 1个虚拟兜底专家(Dummy)
#             self.dynamic_net = nn.Linear(shared_emb_dim, 4)
#             nn.init.xavier_normal_(self.dynamic_net.weight)
#             # 引入 Dummy Expert 吸走绝大部分初始概率，防止初期过度拟合残差
#             nn.init.zeros_(self.dynamic_net.bias)
#             nn.init.constant_(self.dynamic_net.bias[3], 2.83) # Dummy bias

#         # ==========================================================

#     def forward(self, x_cont, x_cat):
#         x_main = self.encoder(x_cont, x_cat)
#         shared_emb = self.shared_base(x_main)
        
#         # 1. 抽取先验
#         pi_00, pi_01, pi_11 = self.extract_pi_prior(x_cont, x_cat)
#         pi_dict = {"p_never": pi_00, "p_complier": pi_01, "p_always": pi_11}
        
#         if self.v8_scheme in [1, 2, 3] and pi_01 is not None:
#             # --- 锚点更新逻辑 (继承 V7) ---
#             if self.training:
#                 q = 1.0 - self.truncation_pct
#                 batch_thresh = torch.quantile(pi_01.detach().float(), q)
#                 if not self.ema_initialized:
#                     self.threshold_ema.copy_(batch_thresh)
#                     self.ema_initialized.fill_(True)
#                 else:
#                     new_ema = self.ema_momentum * self.threshold_ema + (1.0 - self.ema_momentum) * batch_thresh
#                     self.threshold_ema.copy_(new_ema)
#                 current_thresh = batch_thresh
#             else:
#                 current_thresh = self.threshold_ema

#             # 缩放因子
#             scale = self.truncation_temp / (pi_01.std().clamp(min=1e-5))
            
#             # --- 各方案的门控计算 (🌟 核心修复区：对齐维度) ---
#             if self.v8_scheme == 1:
#                 # 修复: [B, 1] -> [B]
#                 delta_c = self.dynamic_net(shared_emb).squeeze(-1) 
#                 gate_val = torch.sigmoid(scale * (pi_01 - (current_thresh - delta_c)))
#                 gate_00, gate_01, gate_11 = gate_val, gate_val, gate_val

#             elif self.v8_scheme == 2:
#                 # 修复: [B, 1] -> [B]
#                 v = self.dynamic_net(shared_emb).squeeze(-1) 
#                 prior_bias = scale * (pi_01 - current_thresh)
#                 gate_val = torch.sigmoid(v + prior_bias)
#                 gate_00, gate_01, gate_11 = gate_val, gate_val, gate_val

#             elif self.v8_scheme == 3:
#                 # v 是 [B, 3], prior_bias 是 [B]
#                 v = self.dynamic_net(shared_emb) 
#                 prior_bias = scale * (pi_01 - current_thresh)
#                 # 修复: 让 prior_bias 变成 [B, 1]，这样才能和 [B, 3] 正常相加
#                 gates = torch.sigmoid(v + prior_bias.unsqueeze(-1))
#                 # 提取出来变成纯 1D 的 [B]
#                 gate_00 = gates[:, 0]
#                 gate_01 = gates[:, 1]
#                 gate_11 = gates[:, 2]

#         elif self.v8_scheme == 4:
#             gates = torch.sigmoid(self.dynamic_net(shared_emb))
#             # 提取出来变成纯 1D 的 [B]
#             gate_00, gate_01, gate_11 = gates[:, 0], gates[:, 1], gates[:, 2]

#         elif self.v8_scheme == 5:
#             gates = F.softmax(self.dynamic_net(shared_emb), dim=-1)
#             # 提取出来变成纯 1D 的 [B]
#             pi_00_final, pi_01_final, pi_11_final = gates[:, 0], gates[:, 1], gates[:, 2]

        


#         # 3. 施加惩罚：更新先验概率
#         if self.v8_scheme in [1, 2, 3, 4]:
#             # 此时 pi_xx 和 gate_xx 都是纯 1D [B]，直接点乘绝对安全
#             pi_00_final = pi_00 * gate_00
#             pi_01_final = pi_01 * gate_01
#             pi_11_final = pi_11 * gate_11

#         # 4. 基座推理
#         base_y0 = self.base_head_0(shared_emb).squeeze(-1)
#         base_y1 = self.base_head_1(shared_emb).squeeze(-1)
        
#         # 5. 专家推理
#         r0_never, r1_never = self.res_never['res_0'](shared_emb).squeeze(-1), self.res_never['res_1'](shared_emb).squeeze(-1)
#         r0_comp, r1_comp   = self.res_comp['res_0'](shared_emb).squeeze(-1), self.res_comp['res_1'](shared_emb).squeeze(-1)
#         r0_always, r1_always = self.res_always['res_0'](shared_emb).squeeze(-1), self.res_always['res_1'](shared_emb).squeeze(-1)
        
#         # 6. 残差融合 (🌟 修复：去掉之前的 squeeze(-1)，因为已经是 1D 了)
#         y0 = base_y0 + (pi_00_final * r0_never + pi_01_final * r0_comp + pi_11_final * r0_always)
#         y1 = base_y1 + (pi_00_final * r1_never + pi_01_final * r1_comp + pi_11_final * r1_always)

#         if not self.training:
#             pi_dict["wb_shared_emb"] = shared_emb
            
#             # V8 特有：重塑后的先验与门控
#             pi_dict["wb_final_pi_never"] = pi_00_final
#             pi_dict["wb_final_pi_comp"] = pi_01_final
#             pi_dict["wb_final_pi_always"] = pi_11_final
            
#             if 'gate_00' in locals():
#                 pi_dict["wb_gate_never"] = gate_00
#                 pi_dict["wb_gate_comp"] = gate_01
#                 pi_dict["wb_gate_always"] = gate_11
                
#             # Base 与 残差三件套 (与 V6/V7 保持一致，方便纵向对比)
#             pi_dict["wb_base_y0"] = base_y0
#             pi_dict["wb_base_y1"] = base_y1
#             pi_dict["wb_base_u"] = base_y1 - base_y0
            
#             pi_dict["wb_res_y0_never"] = r0_never
#             pi_dict["wb_res_y1_never"] = r1_never
#             pi_dict["wb_res_u_never"] = r1_never - r0_never
            
#             pi_dict["wb_res_y0_comp"] = r0_comp
#             pi_dict["wb_res_y1_comp"] = r1_comp
#             pi_dict["wb_res_u_comp"] = r1_comp - r0_comp
            
#             pi_dict["wb_res_y0_always"] = r0_always
#             pi_dict["wb_res_y1_always"] = r1_always
#             pi_dict["wb_res_u_always"] = r1_always - r0_always
#         return y0, y1, pi_dict




# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class DynamicPriorAlignmentLayer(nn.Module):
#     def __init__(self, align_type='lift', momentum=0.05, eps=1e-7):
#         """
#         全自动大盘先验对齐层
#         momentum: EMA 滑动平均的更新步长 (0.05 表示新 Batch 占 5%，历史占 95%)
#         """
#         super().__init__()
#         self.align_type = align_type
#         self.momentum = momentum
#         self.eps = eps
        
#         # 注册为 buffer，会保存在 state_dict 里，但不需要梯度
#         self.register_buffer('running_mean', torch.zeros(3, dtype=torch.float32))
#         self.register_buffer('running_var', torch.ones(3, dtype=torch.float32))
#         self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

#     def forward(self, priors):
#         """
#         priors: Model C 吐出的原始概率, Shape (B, 3) -> [pi_00, pi_01, pi_11]
#         """
#         # 🌟 1. 全自动大盘状态更新 (仅在训练时)
#         if self.training:
#             # 算出当前 Batch 的均值和方差
#             batch_mean = priors.mean(dim=0)
#             batch_var = priors.var(dim=0, unbiased=False)
            
#             # 阻断梯度，纯统计更新
#             with torch.no_grad():
#                 if self.num_batches_tracked == 0:
#                     # 第一个 Batch，直接覆盖初始化
#                     self.running_mean.copy_(batch_mean)
#                     self.running_var.copy_(batch_var)
#                 else:
#                     # 平滑更新 EMA
#                     self.running_mean.copy_((1 - self.momentum) * self.running_mean + self.momentum * batch_mean)
#                     self.running_var.copy_((1 - self.momentum) * self.running_var + self.momentum * batch_var)
                
#                 self.num_batches_tracked += 1

#         # 🌟 2. 为了保证锚点的绝对稳定性，前向对齐始终使用全局 running_mean
#         # (不能用 batch_mean，否则一个小 Batch 里全是羊毛党会导致 Lift 计算错乱)
#         global_mean = self.running_mean
#         global_std = torch.sqrt(self.running_var + self.eps)

#         # 🌟 3. 执行空间映射
#         if self.align_type == 'lift':
#             # 方案 B: Lift Log-Odds = ln( P_i / 全局期望 P )
#             aligned_logits = torch.log((priors + self.eps) / (global_mean + self.eps))
            
#         elif self.align_type == 'z_score':
#             # 方案 A: Z-Score = (P_i - 全局期望) / 全局标准差
#             aligned_logits = (priors - global_mean) / (global_std + self.eps)
            
#         elif self.align_type == 'rank':
#             # 方案 C: Batch 内百分位排序 (依赖大 Batch Size，直接用当前 Batch 算)
#             B = priors.shape[0]
#             if B > 1:
#                 ranks = priors.argsort(dim=0).argsort(dim=0).float() / (B - 1)
#             else:
#                 ranks = torch.ones_like(priors) * 0.5
#             aligned_logits = ranks
#         else:
#             aligned_logits = priors
            
#         return aligned_logits



import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 🟣 V8 架构：5大演进方案 + 新增 S6(Logit加法), S7(插值), S8(先验感知)
# ==========================================
class TARNET_V8_Evolution_MoE(TARNET_Residual_MoE):
    """
    V8 架构：8大演进方案集成版 (支持 Top K% 分位数退化初始化)
    """
    def __init__(self, continuous_dim: int, categorical_cardinalities: dict,
                 hidden_dims: list, dropout_rate: float, 
                 c_model: nn.Module = None, embedding_dim: int = 8,
                 v8_scheme: int = 3,             # 1~5保持原样; 新增 6=Logit加法, 7=动态信任插值, 8=先验感知+乘法
                 shared_emb_dim: int = 128,      
                 truncation_pct: float = 0.05,   
                 truncation_temp: float = 10.0,
                 ema_momentum: float = 0.9,
                 align_temp: float = 1.0):
        
        super().__init__(continuous_dim, categorical_cardinalities, hidden_dims, 
                         dropout_rate, c_model, embedding_dim)
        
        self.v8_scheme = v8_scheme
        self.truncation_pct = truncation_pct
        self.truncation_temp = truncation_temp
        self.ema_momentum = ema_momentum
        self.temp = align_temp
        
        # 工业级 EMA (仅方案 1, 2, 3 需要人工锚点)
        if self.v8_scheme in [1, 2, 3]:
            self.register_buffer('threshold_ema', torch.tensor(0.0))
            self.register_buffer('ema_initialized', torch.tensor(False))

        # ================= V8 动态网络构建 & 初始化 =================
        if self.v8_scheme == 1:
            self.dynamic_net = nn.Sequential(
                nn.Linear(shared_emb_dim, shared_emb_dim // 2),
                nn.ReLU(),
                nn.Linear(shared_emb_dim // 2, 1) 
            )
            nn.init.zeros_(self.dynamic_net[-1].weight)
            nn.init.zeros_(self.dynamic_net[-1].bias)

        elif self.v8_scheme == 2:
            self.dynamic_net = nn.Linear(shared_emb_dim, 1)
            nn.init.zeros_(self.dynamic_net.weight)
            nn.init.zeros_(self.dynamic_net.bias)

        elif self.v8_scheme == 3:
            self.dynamic_net = nn.Linear(shared_emb_dim, 3) 
            nn.init.zeros_(self.dynamic_net.weight)
            nn.init.zeros_(self.dynamic_net.bias)

        elif self.v8_scheme == 4:
            self.dynamic_net = nn.Sequential(
                nn.Linear(shared_emb_dim, shared_emb_dim // 2),
                nn.ReLU(),
                nn.Linear(shared_emb_dim // 2, 3) 
            )

        elif self.v8_scheme == 5:
            self.dynamic_net = nn.Linear(shared_emb_dim, 4)
            nn.init.xavier_normal_(self.dynamic_net.weight)
            nn.init.zeros_(self.dynamic_net.bias)
            nn.init.constant_(self.dynamic_net.bias[3], 2.83) 

        # 🌟🌟🌟 新增：S6, S7, S8 🌟🌟🌟
        elif self.v8_scheme == 6:
            # 方案 6: Logit 空间加法融合 (先验转Logit + 特征Offset -> Softmax)
            self.dynamic_net = nn.Sequential(
                nn.Linear(shared_emb_dim, shared_emb_dim // 2),
                nn.ReLU(),
                nn.Linear(shared_emb_dim // 2, 3) 
            )
            nn.init.zeros_(self.dynamic_net[-1].weight)
            nn.init.zeros_(self.dynamic_net[-1].bias)

        elif self.v8_scheme == 7:
            # 方案 7: 动态信任插值 (3维特征预测概率 + 1维信任度 alpha)
            self.dynamic_net = nn.Sequential(
                nn.Linear(shared_emb_dim, shared_emb_dim // 2),
                nn.ReLU(),
                nn.Linear(shared_emb_dim // 2, 4) 
            )
            nn.init.zeros_(self.dynamic_net[-1].weight)
            nn.init.zeros_(self.dynamic_net[-1].bias)
            nn.init.constant_(self.dynamic_net[-1].bias[3], 2.0) # 初始偏向 1 (信任先验)

        elif self.v8_scheme == 8:
            # 方案 8: 先验感知注入 + 特征乘法 (输入拼接 3维先验)
            self.dynamic_net = nn.Sequential(
                nn.Linear(shared_emb_dim + 3, shared_emb_dim // 2),
                nn.ReLU(),
                nn.Linear(shared_emb_dim // 2, 3) 
            )
        # ==========================================================

    def forward(self, x_cont, x_cat):
        x_main = self.encoder(x_cont, x_cat)
        shared_emb = self.shared_base(x_main)
        
        pi_00, pi_01, pi_11 = self.extract_pi_prior(x_cont, x_cat)
        pi_dict = {"p_never": pi_00, "p_complier": pi_01, "p_always": pi_11}
        
        if self.v8_scheme in [1, 2, 3] and pi_01 is not None:
            # --- 锚点更新逻辑 (完全保留原版) ---
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

            scale = self.truncation_temp / (pi_01.std().clamp(min=1e-5))
            
            if self.v8_scheme == 1:
                delta_c = self.dynamic_net(shared_emb).squeeze(-1) 
                gate_val = torch.sigmoid(scale * (pi_01 - (current_thresh - delta_c)))
                gate_00, gate_01, gate_11 = gate_val, gate_val, gate_val

            elif self.v8_scheme == 2:
                v = self.dynamic_net(shared_emb).squeeze(-1) 
                prior_bias = scale * (pi_01 - current_thresh)
                gate_val = torch.sigmoid(v + prior_bias)
                gate_00, gate_01, gate_11 = gate_val, gate_val, gate_val

            elif self.v8_scheme == 3:
                v = self.dynamic_net(shared_emb) 
                prior_bias = scale * (pi_01 - current_thresh)
                gates = torch.sigmoid(v + prior_bias.unsqueeze(-1))
                gate_00, gate_01, gate_11 = gates[:, 0], gates[:, 1], gates[:, 2]

        elif self.v8_scheme == 4:
            gates = torch.sigmoid(self.dynamic_net(shared_emb))
            gate_00, gate_01, gate_11 = gates[:, 0], gates[:, 1], gates[:, 2]

        elif self.v8_scheme == 5:
            gates = F.softmax(self.dynamic_net(shared_emb), dim=-1)
            pi_00_final, pi_01_final, pi_11_final = gates[:, 0], gates[:, 1], gates[:, 2]

        # 🌟🌟🌟 新增：S6, S7, S8 前向逻辑 🌟🌟🌟
        elif self.v8_scheme == 6:
            feature_offsets = self.dynamic_net(shared_emb)
            # 先验转 Logit 加法融合
            pi_prior_3d = torch.stack([pi_00, pi_01, pi_11], dim=1)
            logit_prior = torch.log(pi_prior_3d + 1e-7)
            pi_final_3d = F.softmax((logit_prior + feature_offsets) / self.temp, dim=-1)
            pi_00_final, pi_01_final, pi_11_final = pi_final_3d[:, 0], pi_final_3d[:, 1], pi_final_3d[:, 2]

        elif self.v8_scheme == 7:
            out = self.dynamic_net(shared_emb) 
            pi_feature_3d = F.softmax(out[:, :3], dim=-1)
            alpha = torch.sigmoid(out[:, 3]) # [B]
            pi_prior_3d = torch.stack([pi_00, pi_01, pi_11], dim=1)
            # 动态插值
            pi_final_3d = alpha.unsqueeze(-1) * pi_prior_3d + (1.0 - alpha.unsqueeze(-1)) * pi_feature_3d
            pi_00_final, pi_01_final, pi_11_final = pi_final_3d[:, 0], pi_final_3d[:, 1], pi_final_3d[:, 2]

        elif self.v8_scheme == 8:
            pi_prior_3d = torch.stack([pi_00, pi_01, pi_11], dim=1)
            # 先验注入
            enhanced_emb = torch.cat([shared_emb, pi_prior_3d], dim=-1)
            gates = torch.sigmoid(self.dynamic_net(enhanced_emb))
            gate_00, gate_01, gate_11 = gates[:, 0], gates[:, 1], gates[:, 2]

        # 3. 施加惩罚：更新先验概率 (🌟 加入 8，因为它也是产生 gate 的乘法)
        if self.v8_scheme in [1, 2, 3, 4, 8]:
            pi_00_final = pi_00 * gate_00
            pi_01_final = pi_01 * gate_01
            pi_11_final = pi_11 * gate_11

        # 4 & 5 & 6. 基座与残差推理 (完全不变)
        base_y0 = self.base_head_0(shared_emb).squeeze(-1)
        base_y1 = self.base_head_1(shared_emb).squeeze(-1)
        r0_never, r1_never = self.res_never['res_0'](shared_emb).squeeze(-1), self.res_never['res_1'](shared_emb).squeeze(-1)
        r0_comp, r1_comp   = self.res_comp['res_0'](shared_emb).squeeze(-1), self.res_comp['res_1'](shared_emb).squeeze(-1)
        r0_always, r1_always = self.res_always['res_0'](shared_emb).squeeze(-1), self.res_always['res_1'](shared_emb).squeeze(-1)
        
        y0 = base_y0 + (pi_00_final * r0_never + pi_01_final * r0_comp + pi_11_final * r0_always)
        y1 = base_y1 + (pi_00_final * r1_never + pi_01_final * r1_comp + pi_11_final * r1_always)

        if not self.training:
            pi_dict["wb_shared_emb"] = shared_emb
            pi_dict["wb_final_pi_never"] = pi_00_final
            pi_dict["wb_final_pi_comp"] = pi_01_final
            pi_dict["wb_final_pi_always"] = pi_11_final
            
            if 'gate_00' in locals():
                pi_dict["wb_gate_never"] = gate_00
                pi_dict["wb_gate_comp"] = gate_01
                pi_dict["wb_gate_always"] = gate_11
                
            pi_dict["wb_base_y0"] = base_y0
            pi_dict["wb_base_y1"] = base_y1
            pi_dict["wb_base_u"] = base_y1 - base_y0
            pi_dict["wb_res_y0_never"] = r0_never
            pi_dict["wb_res_y1_never"] = r1_never
            pi_dict["wb_res_u_never"] = r1_never - r0_never
            pi_dict["wb_res_y0_comp"] = r0_comp
            pi_dict["wb_res_y1_comp"] = r1_comp
            pi_dict["wb_res_u_comp"] = r1_comp - r0_comp
            pi_dict["wb_res_y0_always"] = r0_always
            pi_dict["wb_res_y1_always"] = r1_always
            pi_dict["wb_res_u_always"] = r1_always - r0_always
        return y0, y1, pi_dict


# ==========================================
# 🌟 V11 必备：EMA 动态大盘先验对齐层
# ==========================================
class DynamicPriorAlignmentLayer(nn.Module):
    def __init__(self, align_type='lift', momentum=0.05, eps=1e-7):
        super().__init__()
        self.align_type = align_type
        self.momentum = momentum
        self.eps = eps
        self.register_buffer('running_mean', torch.zeros(3, dtype=torch.float32))
        self.register_buffer('running_var', torch.ones(3, dtype=torch.float32))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

    def forward(self, priors):
        if self.training:
            batch_mean = priors.mean(dim=0)
            batch_var = priors.var(dim=0, unbiased=False)
            with torch.no_grad():
                if self.num_batches_tracked == 0:
                    self.running_mean.copy_(batch_mean)
                    self.running_var.copy_(batch_var)
                else:
                    self.running_mean.copy_((1 - self.momentum) * self.running_mean + self.momentum * batch_mean)
                    self.running_var.copy_((1 - self.momentum) * self.running_var + self.momentum * batch_var)
                self.num_batches_tracked += 1
                
        global_mean = self.running_mean
        global_std = torch.sqrt(self.running_var + self.eps)

        if self.align_type == 'lift':
            aligned_logits = torch.log((priors + self.eps) / (global_mean + self.eps))
        elif self.align_type == 'z_score':
            aligned_logits = (priors - global_mean) / (global_std + self.eps)
        else:
            aligned_logits = priors
            
        return aligned_logits


# ==========================================
# 🌟 V11 架构：Logit 空间动态对齐流 (S4, S6, S7, S8)
# ==========================================
class TARNET_V11_Aligned_MoE(TARNET_Residual_MoE): 
    def __init__(self, continuous_dim: int, categorical_cardinalities: dict,
                 hidden_dims: list, dropout_rate: float, 
                 c_model: nn.Module = None, embedding_dim: int = 8,
                 v11_scheme: int = 4,            
                 align_type: str = 'lift',       
                 align_temp: float = 1.0):       
        
        super().__init__(continuous_dim, categorical_cardinalities, hidden_dims, 
                         dropout_rate, c_model, embedding_dim)
        self.v11_scheme = v11_scheme
        self.temp = align_temp
        shared_emb_dim = hidden_dims[-1]
        self.aligner = DynamicPriorAlignmentLayer(align_type=align_type)

        if self.v11_scheme == 4:
            # S4: 独立特征 MLP -> 乘法门控 (虽然是加法流，但依然支持只做特征乘法)
            self.dynamic_net = nn.Sequential(
                nn.Linear(shared_emb_dim, shared_emb_dim // 2),
                nn.ReLU(),
                nn.Linear(shared_emb_dim // 2, 3) 
            )
        elif self.v11_scheme == 6:
            # S6: Logit 加法融合
            self.dynamic_net = nn.Sequential(
                nn.Linear(shared_emb_dim, shared_emb_dim // 2),
                nn.ReLU(),
                nn.Linear(shared_emb_dim // 2, 3) 
            )
            nn.init.zeros_(self.dynamic_net[-1].weight)
            nn.init.zeros_(self.dynamic_net[-1].bias)
        elif self.v11_scheme == 7:
            # S7: 动态信任插值
            self.dynamic_net = nn.Sequential(
                nn.Linear(shared_emb_dim, shared_emb_dim // 2),
                nn.ReLU(),
                nn.Linear(shared_emb_dim // 2, 4) 
            )
            nn.init.zeros_(self.dynamic_net[-1].weight)
            nn.init.zeros_(self.dynamic_net[-1].bias)
            nn.init.constant_(self.dynamic_net[-1].bias[3], 2.0)
        elif self.v11_scheme == 8:
            # S8: 先验感知注入 + 乘法
            self.dynamic_net = nn.Sequential(
                nn.Linear(shared_emb_dim + 3, shared_emb_dim // 2),
                nn.ReLU(),
                nn.Linear(shared_emb_dim // 2, 3) 
            )

    def forward(self, x_cont, x_cat):
        x_main = self.encoder(x_cont, x_cat)
        shared_emb = self.shared_base(x_main)
        
        pi_00, pi_01, pi_11 = self.extract_pi_prior(x_cont, x_cat)
        pi_dict = {"p_never": pi_00, "p_complier": pi_01, "p_always": pi_11}
        raw_priors = torch.stack([pi_00, pi_01, pi_11], dim=1) 
        
        # 获取大盘对齐后的 Logit (Lift Log-Odds) 和 基准概率
        aligned_logits = self.aligner(raw_priors) 
        aligned_probs = F.softmax(aligned_logits / self.temp, dim=-1)

        # ✨ 融合策略分发
        if self.v11_scheme == 4:
            feature_gates = torch.sigmoid(self.dynamic_net(shared_emb))
            pi_final_3d = aligned_probs * feature_gates

        elif self.v11_scheme == 6:
            feature_offsets = self.dynamic_net(shared_emb)
            # Logit 加法：对齐后的 Logit + 特征 Offset -> Softmax
            pi_final_3d = F.softmax((aligned_logits + feature_offsets) / self.temp, dim=-1)

        elif self.v11_scheme == 7:
            out = self.dynamic_net(shared_emb)
            feature_probs = F.softmax(out[:, :3] / self.temp, dim=-1)
            alpha = torch.sigmoid(out[:, 3:4])
            pi_final_3d = alpha * aligned_probs + (1.0 - alpha) * feature_probs

        elif self.v11_scheme == 8:
            # 注入感知：把对齐后的 Logit 拼接到底层特征中
            enhanced_emb = torch.cat([shared_emb, aligned_logits], dim=-1)
            feature_gates = torch.sigmoid(self.dynamic_net(enhanced_emb))
            pi_final_3d = aligned_probs * feature_gates

        pi_00_final, pi_01_final, pi_11_final = pi_final_3d[:, 0], pi_final_3d[:, 1], pi_final_3d[:, 2]

        # 基座与残差专家融合
        base_y0, base_y1 = self.base_head_0(shared_emb).squeeze(-1), self.base_head_1(shared_emb).squeeze(-1)
        r0_never, r1_never = self.res_never['res_0'](shared_emb).squeeze(-1), self.res_never['res_1'](shared_emb).squeeze(-1)
        r0_comp, r1_comp   = self.res_comp['res_0'](shared_emb).squeeze(-1), self.res_comp['res_1'](shared_emb).squeeze(-1)
        r0_always, r1_always = self.res_always['res_0'](shared_emb).squeeze(-1), self.res_always['res_1'](shared_emb).squeeze(-1)
        
        y0 = base_y0 + (pi_00_final * r0_never + pi_01_final * r0_comp + pi_11_final * r0_always)
        y1 = base_y1 + (pi_00_final * r1_never + pi_01_final * r1_comp + pi_11_final * r1_always)

        if not self.training:
            pi_dict["wb_aligned_logit_comp"] = aligned_logits[:, 1]
            pi_dict["wb_final_pi_comp"] = pi_01_final

        return y0, y1, pi_dict



# ------------------------single task baseline
# ==========================================
# 🐉 经典基线: DragonNet (倾向分靶向正则化)
# ==========================================
class DragonNet(nn.Module):
    def __init__(self, continuous_dim, categorical_cardinalities, hidden_dims=[128, 64, 32]):
        super().__init__()
        self.encoder = FeatureEncoder(continuous_dim, categorical_cardinalities)
        
        # 1. 共享表征层 Z (论文推荐用 ELU 激活函数)
        layers = []
        curr_dim = self.encoder.output_dim
        for dim in hidden_dims:
            layers.extend([nn.Linear(curr_dim, dim), nn.ELU()])
            curr_dim = dim
        self.shared_rep = nn.Sequential(*layers)
        
        # 2. 三个独立预测头: Y0, Y1, T(倾向分)
        self.head_y0 = nn.Sequential(nn.Linear(curr_dim, 1))
        self.head_y1 = nn.Sequential(nn.Linear(curr_dim, 1))
        # 将 models.py 中的 head_t 改为单层线性映射
        self.head_t = nn.Sequential(nn.Linear(curr_dim, 1))
        
        # 3. 靶向正则化参数 (Targeted Regularization epsilon)
        self.epsilon = nn.Parameter(torch.tensor([0.0], requires_grad=True))

    def forward(self, x_cont, x_cat):
        z = self.shared_rep(self.encoder(x_cont, x_cat))
        y0_logit = self.head_y0(z).squeeze(-1)
        y1_logit = self.head_y1(z).squeeze(-1)
        t_logit = self.head_t(z).squeeze(-1)
        
        # t_logit 和 epsilon 塞进 pi_dict 传给 loss_fn
        return y0_logit, y1_logit, {"t_logit": t_logit, "epsilon": self.epsilon}

# ==========================================
# 🎯 显式增益: EUEN (Explicit Uplift Estimation Network)
# ==========================================
class EUEN(nn.Module):
    def __init__(self, continuous_dim, categorical_cardinalities, hidden_dims=[128, 64]):
        super().__init__()
        self.encoder = FeatureEncoder(continuous_dim, categorical_cardinalities)
        
        # 共享底座 (提取通用特征，与 TARNET 保持绝对一致)
        layers = []
        curr_dim = self.encoder.output_dim
        for dim in hidden_dims:
            layers.extend([nn.Linear(curr_dim, dim), nn.ReLU()])
            curr_dim = dim
        self.shared_bottom = nn.Sequential(*layers)
        
        # 🌟 改成极简单层 Linear 头！保证参数量公平
        # base_tower 对应论文里的 C (control net)
        self.base_tower = nn.Linear(curr_dim, 1)
        # uplift_tower 对应论文里的 U (uplift net)
        self.uplift_tower = nn.Linear(curr_dim, 1)

    def forward(self, x_cont, x_cat):
        z = self.shared_bottom(self.encoder(x_cont, x_cat))
        
        # 直接过单层，然后 squeeze 掉最后一个维度
        base_logit = self.base_tower(z).squeeze(-1)     # \mu_c(X)
        uplift_logit = self.uplift_tower(z).squeeze(-1) # \tau(X)
        
        # Logit 域直接相加
        y0_logit = base_logit
        y1_logit = base_logit + uplift_logit
        
        return y0_logit, y1_logit, {}

class EFIN(nn.Module):
    def __init__(self, continuous_dim, categorical_cardinalities, embed_dim=16, hidden_dims=[128, 64]):
        super().__init__()
        self.cont_dim = continuous_dim if continuous_dim is not None else 0
        self.cat_cards = categorical_cardinalities or {}
        self.embed_dim = embed_dim
        
        # -------------------------------------------
        # Module 1: The Feature Encoder (Paper Eq 3)
        # 必须把每个特征独立编码成 embed_dim，不能提前 concat！
        # -------------------------------------------
        # 连续特征：每个特征分配一个独立的线性映射 W*x + b
        if self.cont_dim > 0:
            self.cont_W = nn.Parameter(torch.empty(self.cont_dim, embed_dim))
            self.cont_b = nn.Parameter(torch.empty(self.cont_dim, embed_dim))
            nn.init.xavier_uniform_(self.cont_W)
            nn.init.zeros_(self.cont_b)
            
        # 离散特征：标准的 Lookup Table
        self.cat_embs = nn.ModuleDict({
            col: nn.Embedding(card, embed_dim) for col, card in self.cat_cards.items()
        })
        
        # 干预特征：(Paper 设定 treatment 也有特征，这里我们为二进制 T=1 分配一个专属 Embedding)
        self.t_emb = nn.Embedding(2, embed_dim) 
        
        self.num_features = self.cont_dim + len(self.cat_cards)
        
        # -------------------------------------------
        # Module 2: The Self-interaction (Paper Eq 4, 5, 6)
        # 负责控制组的自然响应 (Natural Response)
        # -------------------------------------------
        # 论文推荐使用 Self-attention
        self.self_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=2, batch_first=True)
        
        # 动态构建 natural_mlp，兼容任意层数
        natural_layers = []
        curr_dim = self.num_features * embed_dim
        for dim in hidden_dims:
            natural_layers.extend([nn.Linear(curr_dim, dim), nn.ReLU()])
            curr_dim = dim
        natural_layers.append(nn.Linear(curr_dim, 1))
        self.natural_mlp = nn.Sequential(*natural_layers)
        
        # -------------------------------------------
        # Module 3: Treatment-aware Interaction (Paper Eq 8, 9, 10)
        # 负责建模特定干预下，特征的敏感度 (Uplift)
        # -------------------------------------------
        self.W_t1 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_t2 = nn.Linear(embed_dim, embed_dim, bias=True)
        self.W_t0 = nn.Linear(embed_dim, 1, bias=False)
        
        # 动态构建 uplift_mlp，兼容任意层数
        uplift_layers = []
        curr_dim_uplift = embed_dim
        for dim in hidden_dims:
            uplift_layers.extend([nn.Linear(curr_dim_uplift, dim), nn.ReLU()])
            curr_dim_uplift = dim
        uplift_layers.append(nn.Linear(curr_dim_uplift, 1))
        self.uplift_mlp = nn.Sequential(*uplift_layers)
        
        # -------------------------------------------
        # Module 4: Intervention Constraint (Paper Eq 13)
        # 反向标签预测，防止分布差异
        # -------------------------------------------
        self.constraint_mlp = nn.Linear(embed_dim, 1)

    def forward(self, x_cont, x_cat):
        device = x_cont.device if x_cont is not None else x_cat[list(x_cat.keys())[0]].device
        B = x_cont.shape[0] if x_cont is not None else x_cat[list(x_cat.keys())[0]].shape[0]
        
        # ==================================
        # 1. 序列化特征编码 (B, F, D)
        # ==================================
        feat_embs = []
        if self.cont_dim > 0:
            # (B, F_cont, 1) * (1, F_cont, D) -> (B, F_cont, D)
            cont_e = x_cont.unsqueeze(-1) * self.cont_W.unsqueeze(0) + self.cont_b.unsqueeze(0)
            feat_embs.append(cont_e)
        if len(self.cat_cards) > 0:
            cat_e = [self.cat_embs[col](x_cat[col]).unsqueeze(1) for col in self.cat_cards.keys()]
            feat_embs.append(torch.cat(cat_e, dim=1))
            
        X = torch.cat(feat_embs, dim=1) # Shape: (B, F, D)
        
        # ==================================
        # 2. 控制组自然响应 y(0)
        # ==================================
        attn_out, _ = self.self_attn(X, X, X) # (B, F, D)
        # 展平后过 MLP: concat(e_x) -> MLP
        y0_logit = self.natural_mlp(attn_out.reshape(B, -1)).squeeze(-1) 
        
        # ==================================
        # 3. 实验组增益预测 y(1) = y(0) + tau_1
        # ==================================
        # 获取 T=1 的专属表征 e^t
        t1_idx = torch.ones((B,), dtype=torch.long, device=device)
        e_t = self.t_emb(t1_idx) # (B, D)
        
        # 交叉注意力打分: alpha = Softmax(W_t0 * ReLU(W_t1*e_t + W_t2*X))
        interact = F.relu(self.W_t1(e_t).unsqueeze(1) + self.W_t2(X)) # (B, F, D)
        alpha = F.softmax(self.W_t0(interact), dim=1) # (B, F, 1)
        
        # 加权求和获取敏感特征 e^{xt}
        e_xt = torch.sum(alpha * X, dim=1) # (B, D)
        
        # 预测纯增益 Uplift
        tau_1 = self.uplift_mlp(e_xt).squeeze(-1) # (B,)
        
        # 显式相加
        y1_logit = y0_logit + tau_1 
        
        # ==================================
        # 4. 干预平衡约束 (用于 Loss 惩罚)
        # ==================================
        t_constraint_logit = self.constraint_mlp(e_xt).squeeze(-1)
        
        return y0_logit, y1_logit, {"efin_constraint_logit": t_constraint_logit}



# ==========================================
# 🎯 经典基线: S-Learner (Single Learner)
# 核心思想：把干预 T 作为一种普通特征，拼接到连续特征中，用单个大模型拟合。
# ==========================================
class S_Learner(nn.Module):
    def __init__(self, continuous_dim, categorical_cardinalities, hidden_dims=[128, 64]):
        super().__init__()
        # ⚠️ 核心改动：连续特征维度 + 1 (因为要把 T 拼进去)
        cont_dim_with_t = (continuous_dim if continuous_dim is not None else 0) + 1
        self.encoder = FeatureEncoder(cont_dim_with_t, categorical_cardinalities)
        
        layers = []
        curr_dim = self.encoder.output_dim
        for dim in hidden_dims:
            layers.extend([nn.Linear(curr_dim, dim), nn.ReLU()])
            curr_dim = dim
        layers.append(nn.Linear(curr_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_cont, x_cat, t=None):
        # 💡 训练模式：直接传入真实的 T 参与特征拼接
        if t is not None:
            t_feat = t.unsqueeze(1).float()
            x_c = torch.cat([x_cont, t_feat], dim=1) if x_cont is not None else t_feat
            
            feat = self.encoder(x_c, x_cat)
            y_logit = self.mlp(feat).squeeze(-1)
            
            # 为了兼容你的 compute_stage3_loss (里面有 torch.where)，
            # 我们直接返回两个一模一样的 y_logit，loss 那边取哪个都对
            return y_logit, y_logit, {}
            
        # 💡 评估模式：Evaluator 会调用它，此时 T 是未知的
        # 我们必须显式构造 T=0 和 T=1，分别前向传播两次，算出真实的 Uplift
        else:
            device = x_cont.device if x_cont is not None else x_cat[list(x_cat.keys())[0]].device
            batch_size = x_cont.shape[0] if x_cont is not None else x_cat[list(x_cat.keys())[0]].shape[0]
            
            # 强制令 T = 0
            t0 = torch.zeros((batch_size, 1), device=device)
            x_c0 = torch.cat([x_cont, t0], dim=1) if x_cont is not None else t0
            y0_logit = self.mlp(self.encoder(x_c0, x_cat)).squeeze(-1)
            
            # 强制令 T = 1
            t1 = torch.ones((batch_size, 1), device=device)
            x_c1 = torch.cat([x_cont, t1], dim=1) if x_cont is not None else t1
            y1_logit = self.mlp(self.encoder(x_c1, x_cat)).squeeze(-1)
            
            return y0_logit, y1_logit, {}

# ==========================================
# 🎯 经典基线: T-Learner (Two Learner)
# 核心思想：不用共享底座，T=0 训练一套独立网络，T=1 训练另一套独立网络。
# ==========================================
class T_Learner(nn.Module):
    def __init__(self, continuous_dim, categorical_cardinalities, hidden_dims=[128, 64]):
        super().__init__()
        
        # 👑 第一套班子 (专职预测 Y0)
        self.encoder0 = FeatureEncoder(continuous_dim, categorical_cardinalities)
        layers0 = []
        curr_dim0 = self.encoder0.output_dim
        for dim in hidden_dims:
            layers0.extend([nn.Linear(curr_dim0, dim), nn.ReLU()])
            curr_dim0 = dim
        layers0.append(nn.Linear(curr_dim0, 1))
        self.mlp0 = nn.Sequential(*layers0)
        
        # 👑 第二套班子 (专职预测 Y1)
        self.encoder1 = FeatureEncoder(continuous_dim, categorical_cardinalities)
        layers1 = []
        curr_dim1 = self.encoder1.output_dim
        for dim in hidden_dims:
            layers1.extend([nn.Linear(curr_dim1, dim), nn.ReLU()])
            curr_dim1 = dim
        layers1.append(nn.Linear(curr_dim1, 1))
        self.mlp1 = nn.Sequential(*layers1)

    def forward(self, x_cont, x_cat):
        # 互不干扰，各自预测
        # 依赖 loss 里的 torch.where(t==1, y1, y0) 来自动阻断梯度
        # 这样 T=1 的样本只会更新 mlp1，T=0 的样本只会更新 mlp0
        y0_logit = self.mlp0(self.encoder0(x_cont, x_cat)).squeeze(-1)
        y1_logit = self.mlp1(self.encoder1(x_cont, x_cat)).squeeze(-1)
        return y0_logit, y1_logit, {}


# ==========================================
# ⚖️ 经典基线: CFRNet (Counterfactual Regression)
# 核心思想：在双塔的基础上，强制要求中间表征层 Z 的分布 (T=1 和 T=0) 对齐
# ==========================================
class CFRNet(nn.Module):
    def __init__(self, continuous_dim, categorical_cardinalities, hidden_dims=[128, 64, 32]):
        super().__init__()
        self.encoder = FeatureEncoder(continuous_dim, categorical_cardinalities)
        
        # 1. 共享表征层 Z
        layers = []
        curr_dim = self.encoder.output_dim
        for dim in hidden_dims:
            layers.extend([nn.Linear(curr_dim, dim), nn.ReLU()])
            curr_dim = dim
        self.shared_rep = nn.Sequential(*layers)
        
        # 2. 独立双头
        self.head_y0 = nn.Linear(curr_dim, 1) # nn.Sequential(nn.Linear(curr_dim, 32), nn.ReLU(), nn.Linear(32, 1))
        self.head_y1 = nn.Linear(curr_dim, 1) # nn.Sequential(nn.Linear(curr_dim, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x_cont, x_cat):
        # 提取公共表征
        z = self.shared_rep(self.encoder(x_cont, x_cat))
        
        # 分别预测
        y0_logit = self.head_y0(z).squeeze(-1)
        y1_logit = self.head_y1(z).squeeze(-1)
        
        # 💡 核心：把 z 塞进字典传出去，给 Loss 算分布距离用！
        return y0_logit, y1_logit, {"z": z}


# ==========================================
# 👑 工业界 SOTA: DESCN (Deep Entire Space Cross Networks - KDD 2022)
# ==========================================
class DESCN(nn.Module):
    def __init__(self, continuous_dim, categorical_cardinalities, shared_dims=[128], tower_dims=[64]):
        super().__init__()
        self.encoder = FeatureEncoder(continuous_dim, categorical_cardinalities)
        
        # 1. 共享底层 (Shared Network)
        # 论文 4.4 节指出：shared network 包含 128 hidden units
        shared_layers = []
        curr_dim = self.encoder.output_dim
        for dim in shared_dims:
            shared_layers.extend([nn.Linear(curr_dim, dim), nn.ReLU()])
            curr_dim = dim
        self.shared_net = nn.Sequential(*shared_layers)
        
        # 2. 四个独立的 Tower (论文 4.4 节指出：other sub-models 包含 64 hidden units)
        def build_tower(in_dim, dims):
            layers = []
            curr = in_dim
            for d in dims:
                layers.extend([nn.Linear(curr, d), nn.ReLU()])
                curr = d
            layers.append(nn.Linear(curr, 1))
            return nn.Sequential(*layers)
            
        self.propensity_net = build_tower(curr_dim, tower_dims) # 预测倾向分 pi
        self.tr_net = build_tower(curr_dim, tower_dims)         # 预测 Treated Response (mu_1)
        self.cr_net = build_tower(curr_dim, tower_dims)         # 预测 Control Response (mu_0)
        self.pte_net = build_tower(curr_dim, tower_dims)        # 预测 Pseudo Treatment Effect (tau')

    def forward(self, x_cont, x_cat):
        # 提取共享表征
        shared_rep = self.shared_net(self.encoder(x_cont, x_cat))
        
        # 所有的网络直接输出 Logit (不经过 Sigmoid)
        # 论文 Eq 6 指出：tau' 是加在 logit 上的，这样数学上更稳定
        pi_logit = self.propensity_net(shared_rep).squeeze(-1)
        mu1_logit = self.tr_net(shared_rep).squeeze(-1)
        mu0_logit = self.cr_net(shared_rep).squeeze(-1)
        tau_logit = self.pte_net(shared_rep).squeeze(-1)
        
        pi_dict = {
            "descn_pi_logit": pi_logit,
            "descn_tau_logit": tau_logit
        }
        
        # 推断时，直接返回 mu0 和 mu1 即可，Evaluator 会用它们相减算 Uplift
        return mu0_logit, mu1_logit, pi_dict


    



import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicPriorAlignmentLayer(nn.Module):
    def __init__(self, align_type='lift', momentum=0.05, eps=1e-7):
        """
        全自动大盘先验对齐层
        momentum: EMA 滑动平均的更新步长 (0.05 表示新 Batch 占 5%，历史占 95%)
        """
        super().__init__()
        self.align_type = align_type
        self.momentum = momentum
        self.eps = eps
        
        # 注册为 buffer，会保存在 state_dict 里，但不需要梯度
        self.register_buffer('running_mean', torch.zeros(3, dtype=torch.float32))
        self.register_buffer('running_var', torch.ones(3, dtype=torch.float32))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

    def forward(self, priors):
        """
        priors: Model C 吐出的原始概率, Shape (B, 3) -> [pi_00, pi_01, pi_11]
        """
        # 🌟 1. 全自动大盘状态更新 (仅在训练时)
        if self.training:
            # 算出当前 Batch 的均值和方差
            batch_mean = priors.mean(dim=0)
            batch_var = priors.var(dim=0, unbiased=False)
            
            # 阻断梯度，纯统计更新
            with torch.no_grad():
                if self.num_batches_tracked == 0:
                    # 第一个 Batch，直接覆盖初始化
                    self.running_mean.copy_(batch_mean)
                    self.running_var.copy_(batch_var)
                else:
                    # 平滑更新 EMA
                    self.running_mean.copy_((1 - self.momentum) * self.running_mean + self.momentum * batch_mean)
                    self.running_var.copy_((1 - self.momentum) * self.running_var + self.momentum * batch_var)
                
                self.num_batches_tracked += 1

        # 🌟 2. 为了保证锚点的绝对稳定性，前向对齐始终使用全局 running_mean
        # (不能用 batch_mean，否则一个小 Batch 里全是羊毛党会导致 Lift 计算错乱)
        global_mean = self.running_mean
        global_std = torch.sqrt(self.running_var + self.eps)

        # 🌟 3. 执行空间映射
        if self.align_type == 'lift':
            # 方案 B: Lift Log-Odds = ln( P_i / 全局期望 P )
            aligned_logits = torch.log((priors + self.eps) / (global_mean + self.eps))
            
        elif self.align_type == 'z_score':
            # 方案 A: Z-Score = (P_i - 全局期望) / 全局标准差
            aligned_logits = (priors - global_mean) / (global_std + self.eps)
            
        elif self.align_type == 'rank':
            # 方案 C: Batch 内百分位排序 (依赖大 Batch Size，直接用当前 Batch 算)
            B = priors.shape[0]
            if B > 1:
                ranks = priors.argsort(dim=0).argsort(dim=0).float() / (B - 1)
            else:
                ranks = torch.ones_like(priors) * 0.5
            aligned_logits = ranks
        else:
            aligned_logits = priors
            
        return aligned_logits
    



