import os
# ==========================================
# 🛡️ 终极防御：Ray 多进程与内核冲突解决指南
# ==========================================
# 禁用 GRPC fork 支持，防止在数据加载 (num_workers>0) 时死锁
os.environ['GRPC_ENABLE_FORK_SUPPORT'] = '0'
# 绕过磁盘空间检查，防止在存储受限节点被踢出
os.environ["RAY_DEBUG_DISABLE_USING_FREE_DISK_SPACE_CHECK"] = "1"
os.environ["RAY_local_fs_capacity_threshold"] = "0.99"
os.environ["RAY_RUNTIME_ENV_CREATE_TIMEOUT_PER_RESOURCE_KIB"] = "600"
os.environ["RAY_GCS_MAX_FINISHED_JOBS"] = "100"
os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0"

# 方案一：把 Ray 的容忍底线拉高到 99%（推荐这个，好歹给系统留 1% 喘口气）
os.environ["RAY_memory_usage_threshold"] = "0.995"

# 方案二：彻底蒙住 Ray 的眼睛，完全关闭自保（如果你选这个，把下面这行取消注释）
# os.environ["RAY_memory_monitor_refresh_ms"] = "0"

import time
import argparse
from pathlib import Path
import sys
import uuid
import shutil
import pandas as pd
import numpy as np
import json

# 将项目根目录加入环境
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 兼容新老版本 Ray 的存储参数差异
try:
    from ray.train import RunConfig
except ImportError:
    from ray.air.config import RunConfig
from ray import tune, train

# 导入咱们的五大基石
from data_pipeline import get_base_config, get_data_spec
from search_space import get_default_hyperparams, get_ray_search_space
from trainer import train_trial

def set_all_seeds(seed):
    """
    全方位锁定随机性，确保分布式/多卡环境下实验可复现
    """
    import random
    import torch # 假设你的项目基于 Torch
    
    # 1. 基本随机性锁定
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
    # 2. PyTorch 随机性锁定
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # 针对多卡环境
    
    # 3. 确定性算法锁定 (牺牲极小性能换取 100% 复现)
    # 强制 CUDNN 使用确定的卷积算法
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False
    
    # 针对新版本 Torch 的严苛模式（可选，如果某些算子不支持会报错）
    # torch.use_deterministic_algorithms(True, warn_only=True)
    
    print(f"🎯 随机种子已锁定为: {seed} (包含 Python, NumPy, Torch, CUDNN)")

def get_safe_run_config(exp_name, save_dir):
    """兼容新老版本 Ray 的存储参数差异"""
    try:
        return RunConfig(name=exp_name, storage_path=save_dir)
    except TypeError:
        return RunConfig(name=exp_name, local_dir=save_dir)

def generate_mock_data(data_dir):
    """如果NAS硬盘上没有数据，帮你自动造三份用来跑通流程的假数据"""
    os.makedirs(data_dir, exist_ok=True)
    if not os.path.exists(f"{data_dir}/train.parquet"):
        print("🔧 检测到本地无数据，正在自动生成 Mock Parquet 数据用于跑通代码...")
        np.random.seed(42)
        for split, size in [("train", 2000), ("valid", 500), ("test", 500)]:
            df = pd.DataFrame({
                "f0": np.random.randn(size), "f1": np.random.randn(size),
                "treatment": np.random.randint(0, 2, size),
                "conversion": np.random.randint(0, 2, size),
                "visit": np.random.randint(0, 2, size)
            })
            df.to_parquet(f"{data_dir}/{split}.parquet")

def build_parser():
    parser = argparse.ArgumentParser(description="主分层 Uplift 统一实验框架 (工业防弹版)")
    
    # 核心控制
    parser.add_argument("--mode", type=str, choices=["debug", "tune", "eval", 'reproduce', 'reproduce_eval', 'single_grid'], default="tune",
                        help="debug(极速单次验证) / tune(大规模搜参) / eval(直接加载测试)")
    parser.add_argument("--task", type=str, choices=["train_c", "train_y"], default="train_y")
    parser.add_argument("--model", type=str, default="TARNET", help="指定模型骨架，如 TARNET")
    parser.add_argument("--data_name", type=str, default="criteo")
    parser.add_argument("--config_path", type=str, default=None)
    
    # 实验命名与策略
    parser.add_argument("--exp_name", type=str, default="v1_uplift_baseline")
    parser.add_argument("--version", type=str, default="v1_baseline")
    
    # 预加载与指定权重
    parser.add_argument("--c_ckpt_path", type=str, default=None, help="指定Stage1 C模型的权重路径")
    parser.add_argument("--eval_ckpt", type=str, default=None, help="仅eval模式:指定要测试的权重路径")
    
    # 资源与运行时
    parser.add_argument("--cuda", type=str, default="0", help="指定可见的GPU ID")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader的worker数")
    parser.add_argument("--num_per_gpu", type=float, default=1, help="每个Trial分配多少GPU")
    parser.add_argument("--seed", type=int, default=42)

    return parser

def main():
    args = build_parser().parse_args()
    
    # 1. 显卡隔离 (最先执行)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    set_all_seeds(args.seed)
    # ==========================================
    # 🛡️ 启动 Ray 终极防御矩阵
    # ==========================================
    if args.mode in ["debug", "tune"]:
        print("\n" + "="*60)
        print("🛡️ 初始化 Ray 终极防御矩阵...")
        init_start_time = time.time()
        
        # 动态生成绝对唯一的目录名，彻底物理隔离 Socket 文件
        tmp_name = f"ray_{os.getuid()}_{uuid.uuid4().hex[:8]}"
        # /NAS/shith/ray_tmp
        RAY_TMP = os.path.join('/data/shith/ray_tmp', tmp_name) # 根据你的NAS结构修改前缀
        os.makedirs(RAY_TMP, exist_ok=True)
        
        # 强行重定向系统级临时环境变量
        os.environ["RAY_TMPDIR"] = RAY_TMP
        os.environ["TMPDIR"] = RAY_TMP
        os.environ["TEMP"] = RAY_TMP
        os.environ["TMP"] = RAY_TMP
        


        import ray
        if not ray.is_initialized():
            ray.init(
                address="local", 
                namespace=tmp_name,
                include_dashboard=False,
                ignore_reinit_error=True,
                _temp_dir=RAY_TMP,
                object_store_memory=20 * 1024 * 1024 * 1024,
            )
            # ray.init('auto')
        init_end_time = time.time()
        print(f"✅ Ray 初始化完毕，安全舱体建立于: {RAY_TMP}")
        print(f"⏱️ 耗时: {init_end_time - init_start_time:.2f} 秒")
        print("=" * 60 + "\n")

    # 2. 构建多级目录名 (data / task / model / version / exp)
    run_hierarchy = f"{args.data_name}/{args.task}/{args.model}/{args.version}/{args.exp_name}"

    print(f"🚀 启动: {args.exp_name} | 模式: {args.mode.upper()} | 模型: {args.model}")
    print(f"📂 存储层级: {run_hierarchy}")
    print("=" * 60)

    # 3. 组装全局 Config
    project_root = os.path.abspath(os.path.dirname(__file__))

    base_cfg = get_base_config()
    base_cfg.update({
        "task": args.task, "data_name": args.data_name,
        "model": args.model, 
        "exp_name": args.exp_name, "version": args.version,
        "c_ckpt_path": args.c_ckpt_path, "mode": args.mode,
        "run_hierarchy": run_hierarchy,
        # 👉 核心新增：把绝对根目录死死传进去
        "project_root": project_root, 
        "num_workers": args.num_workers,
        "num_per_gpu": args.num_per_gpu,
        "seed": args.seed
    })
    
    data_spec = get_data_spec(args.data_name)
    base_cfg["data"] = data_spec
    generate_mock_data(data_spec["data_dir"])

    # --- 逻辑分发 ---

    # 1. DEBUG 模式：挂入 Tune 管道，极速试错
    if args.mode == "debug":
        print("🛠️ 进入 DEBUG 模式：调用 Ray Tune 跑 5 个 Epoch...")
        trial_space = {**base_cfg, **get_default_hyperparams(args.task, args.version)}
        trial_space["num_epochs"] = 2
        trial_space["max_steps_per_epoch"] = 2  
        
        save_dir = os.path.abspath(f"./ray_results/debug/{run_hierarchy}")
        tuner = tune.Tuner(
            tune.with_resources(train_trial, resources={"cpu": 1, "gpu": args.num_per_gpu}),
            param_space=trial_space,
            run_config=get_safe_run_config(args.exp_name, save_dir),
            runtime_env={"working_dir": project_root,
                "excludes": ["results", "logs", "ckpts", 'old', 'ablation_logs', '__pycache__'], # 极其重要：排除大型目录 
                "env_vars": {"RAY_CHDIR_TO_TRIAL_DIR": "0"}   # 禁用目录切换以减少 I/O 开销 [22, 23]}
        }
        )
        tuner.fit()
        print("🎉 Debug 跑通！Ray Tune 流水线无异常。")
        
    # 2. TUNE 模式：火力全开
    elif args.mode == "tune":
        print("🌐 进入 TUNE 模式：正在唤醒集群搜参...")
        
        # 👉 在每次 Tune 之前，先清空一下之前可能残留的 global 打擂台文件，防止跨实验串改
        ckpt_dir = os.path.join(project_root, f"ckpts/{run_hierarchy}")
        os.makedirs(ckpt_dir, exist_ok=True)
        global_metric_path = os.path.join(ckpt_dir, "global_best_metric.txt")
        if os.path.exists(global_metric_path):
            os.remove(global_metric_path)
            
        ray_space = get_ray_search_space(args.task, args.version)
        trial_space = {**base_cfg, **ray_space}
        
        save_dir = os.path.abspath(f"./ray_results/tune/{run_hierarchy}")
        tuner = tune.Tuner(
            tune.with_resources(train_trial, resources={"cpu": max(1, args.num_workers), "gpu": args.num_per_gpu}),
            param_space=trial_space,
            tune_config=tune.TuneConfig(
                metric="Target_C_AUUC" if args.task=="train_c" else "Target_Y_AUUC", 
                mode="max", 
                num_samples=1 
            ),
            run_config=get_safe_run_config(args.exp_name, save_dir)
        )
        tuner.fit()
        
        # 🌟 核心收割逻辑：群雄混战结束，从擂台里请出蛊王
        print("\n🚀 正在从本地磁盘提取全局蛊王模型，进行最终全量验证...")
        best_config_path = os.path.join(ckpt_dir, "best_config.json")
        best_ckpt_save_path = os.path.join(ckpt_dir, "best_model.pth")
        
        if os.path.exists(best_config_path) and os.path.exists(best_ckpt_save_path):
            # import json
            with open(best_config_path, 'r') as f:
                best_cfg = json.load(f)
            # 装载 config 和路径，开启 eval 模式直通车
            eval_cfg = {**base_cfg, **best_cfg, "mode": "eval", "eval_ckpt_path": best_ckpt_save_path}
            train_trial(eval_cfg)
        else:
            print(f"⚠️ 找不到全局模型配置文件 {best_config_path}，请检查是否所有 Trial 都爆 0 提早结束了！")

    # 3. EVAL 模式：指定路径跑测试 (该模式下无须唤起 Ray)
    elif args.mode == "eval":
        ckpt_to_load = args.eval_ckpt or f"./ckpts/{run_hierarchy}/best_model.pth"
        config_to_load = ckpt_to_load.replace("best_model.pth", "best_config.json")
        print(f"🔍 进入 EVAL 模式：尝试加载权重 {ckpt_to_load}")
        # import json
        if os.path.exists(config_to_load):
            print(f"📄 找到配套的 Config 文件，正在按照最优网络结构构建模型...")
            with open(config_to_load, 'r') as f:
                saved_cfg = json.load(f)
        else:
            print(f"⚠️ 找不到配套的 {config_to_load}，退回使用默认参数 (可能会报维度不匹配错误！)")
            saved_cfg = get_default_hyperparams(args.task, args.version)
        trial_cfg = {**base_cfg, **saved_cfg}
        trial_cfg["eval_ckpt_path"] = ckpt_to_load  
        trial_cfg["mode"] = "eval" # 确保强制进入 eval 模式
        train_trial(trial_cfg)

        
    elif args.mode in ["reproduce", "reproduce_eval"]:
        if not os.path.exists(args.config_path):
            raise FileNotFoundError(f"❌ 找不到配置文件: {args.config_path}")
            
        print(f"\n🚀 启动 {args.mode.upper()} 模式 | 读取图纸: {args.config_path} | Seed: {args.seed}")
        
        with open(args.config_path, 'r', encoding='utf-8') as f:
            best_cfg = json.load(f)
            
        trial_cfg = {**base_cfg, **best_cfg}
        trial_cfg["seed"] = args.seed
        trial_cfg["mode"] = args.mode
        
        train_trial(trial_cfg)
        
        print(f"✅ Seed {args.seed} {args.mode} 流水线全部结束！")
    # =========================================================================
    # 👑 物理突围防线：新增免 Ray 前台串行/并发强训模式 (single_grid)
    # =========================================================================
# =========================================================================
    # 👑 物理突围防线：免 Ray 纯 Python 参数化全局动态调度器通道 (single_grid)
    # =========================================================================
    elif args.mode == "single_grid":
        from custom_grid_space import ALL_CUSTOM_SPACES
        # 🌟 核心修正：顺应你原生 trainer.py 的导出习惯，直接导入大盘原汁原味的 Trainer 类或核心函数
        # 根据你原有的 main.py，如果是从 trainer 导入 Trainer 类，就用下面这行：
        from trainer import Trainer  
        
        # 1. 拦截检查该特定版本配置是否存在
        if args.version not in ALL_CUSTOM_SPACES:
            raise ValueError(f"❌ [免Ray空间] 找不到你指定的空间版本名字: {args.version}")
            
        # 2. 强行拉出对应的纯净参数配置，不跟 Ray 产生半毛钱直积
        current_config = ALL_CUSTOM_SPACES[args.version]
        
        print("=" * 80)
        print(f"🚀 [NO-RAY ENGINE] 顺利切入免Ray强训通道！随机种子已被锁死为: {args.seed}")
        print(f"📂 实验保存文件夹: {args.exp_name}")
        print(f"⚙️  全通道灌入配置明细: {current_config}")
        print("=" * 80)
        
        # 3. 实时同步覆盖掉框架内 args 中的全局默认参数，完成超参侵入
        for k, v in current_config.items():
            if hasattr(args, k):
                setattr(args, k, v)
                
        # 4. 🌟 核心修正：用你最原始、真刀真枪跑通的 Trainer 初始化方式去唤醒训练！
        # 如果你底层的构造函数习惯直接收 args，这样写：
        trainer = Trainer(args)
        trainer.train()
        
        # 💡 注：如果你原本的 main.py 里有特殊的传入参数（比如 trainer = Trainer(config, device)）
        # 请在这里保持和你原生 main.py 第 50-100 行附近的实例化代码完全像素级一致即可！
        
        print(f"✅ [NO-RAY ENGINE] 专属版本 {args.version} | 种子 {args.seed} 顺利通过，新日志已刷出！")

    # 运行结束后清理临时目录 (保持服务器整洁)
    if args.mode in ["debug", "tune"]:
        try:
            ray.shutdown()
            shutil.rmtree(RAY_TMP, ignore_errors=True)
            print(f"🧹 已清理 Ray 安全舱体: {RAY_TMP}")
        except Exception as e:
            print(f"清理临时目录失败: {e}")

if __name__ == "__main__":
    main()