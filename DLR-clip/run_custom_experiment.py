#!/usr/bin/env python3

"""
运行所有数据集的few-shot实验，每个数据集shots=16，训练轮数=50
"""

import os
import sys
import argparse

# 添加项目路径到Python路径
project_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_path)

# 设置环境变量
os.environ['DATA_ROOT'] = "/media/yang/49f29042-389a-46e0-b8b1-94439dc013a5/data"
os.environ['IMAGENET_ROOT'] = "/media/yang/Elements SE"  # ImageNet 单独放在外接硬盘
os.environ['MODEL_CACHE_DIR'] = "./model/clip"
os.environ['LOG_ROOT'] = "./result/log"

def run_custom_experiment(cma_start=5, cma_end=12, cma_dim=32, cma_scale=0.01, 
                        use_skd=False, skd_temperature=2.0, skd_smoothing=0.2, skd_lambda=0.01,
                        shots_list=[1, 2, 4, 8, 16], exclude_imagenet=True, use_glr=True, fixed_alpha=None, fixed_weight=None, experiment_name=None, batch_size=16):
    """运行自定义实验：可配置CMA参数和shots列表"""
    print("开始运行自定义实验...")
    print(f"设置: shots_list={shots_list}, 训练轮数=50")
    print(f"CMA配置: 层{cma_start}-{cma_end}, 维度{cma_dim}, 缩放因子{cma_scale}")
    print(f"GLR模块: {'启用' if use_glr else '禁用'}")
    if use_skd:
        print(f"SKD配置: 温度={skd_temperature}, smoothing={skd_smoothing}, lambda={skd_lambda}")
    if fixed_alpha is not None:
        print(f"GLR融合门控: 固定 alpha={fixed_alpha}")
    if fixed_weight is not None:
        print(f"DLF融合权重: 固定 weight={fixed_weight}")
    if exclude_imagenet:
        print("注意: 排除ImageNet数据集")
    print("=" * 50)
    
    # 导入必要的模块
    from dlr_train import AllExperiments
    
    # 创建实验实例
    all_experiment = AllExperiments()
    
    # 如果排除ImageNet，修改数据集列表
    if exclude_imagenet:
        all_experiment.datasets = "fgvc/caltech101/stanford_cars/dtd/eurosat/oxford_flowers/food101/oxford_pets/sun397/ucf101"
    
    # 运行所有数据集的所有shots
    for shots in shots_list:
        # 生成简洁的日志文件名
        if experiment_name:
            log_filename = f"{experiment_name}_{shots}shot.txt"
        else:
            # 默认生成的文件名
            log_suffix = ""
            if use_skd:
                log_suffix += "_SKD"
            if use_glr:
                log_suffix += "_GLR"
            if not use_skd and not use_glr:
                log_suffix = "_CMA"
            log_filename = f"CMA{log_suffix}_{shots}shot.txt"
        
        log_txt_path = os.path.join(os.environ['LOG_ROOT'], log_filename)
        # 确保日志目录存在
        os.makedirs(os.path.dirname(log_txt_path), exist_ok=True)
        
        # 运行实验: 使用指定的CMA配置
        all_experiment.experiment_one(
            shots=shots, 
            backbone="ViT-B/16", 
            train_epoch=50, 
            has_ood=False,  # 修改默认值为False
            use_cma=True, 
            cma_start_layer=cma_start,
            cma_end_layer=cma_end,
            cma_dim=cma_dim,
            cma_scale=cma_scale,
            use_skd=use_skd,
            skd_temperature=skd_temperature,
            skd_smoothing=skd_smoothing,
            skd_lambda=skd_lambda,
            use_glr=use_glr,  # 添加GLR参数
            fixed_alpha=fixed_alpha,  # 添加固定alpha参数
            fixed_weight=fixed_weight,  # 添加固定weight参数
            batch_size=batch_size,  # 添加batch_size参数
            log_txt_path=log_txt_path
        )
    
    print("=" * 50)
    print("实验完成!")
    print(f"结果已保存到: {os.environ['LOG_ROOT']}")

def run_single_dataset(dataset_name, cma_start=5, cma_end=12, cma_dim=32, cma_scale=0.01,
                      use_skd=False, skd_temperature=2.0, skd_smoothing=0.2, skd_lambda=0.01,
                      shots_list=[1, 2, 4, 8, 16], use_glr=True, fixed_alpha=None, fixed_weight=None, batch_size=16, experiment_name=None):
    """运行单个数据集的实验"""
    print(f"开始运行 {dataset_name} 数据集实验...")
    print(f"设置: shots_list={shots_list}, 训练轮数=50")
    print(f"CMA配置: 层{cma_start}-{cma_end}, 维度{cma_dim}, 缩放因子{cma_scale}")
    print(f"GLR模块: {'启用' if use_glr else '禁用'}")
    if use_skd:
        print(f"SKD配置: 温度={skd_temperature}, smoothing={skd_smoothing}, lambda={skd_lambda}")
    print("=" * 50)
    
    # 导入必要的模块
    from dlr_train import AllExperiments
    
    # 创建实验实例
    all_experiment = AllExperiments()
    
    # 只运行指定数据集
    all_experiment.datasets = dataset_name
    
    # 运行所有shots
    for shots in shots_list:
        # 对于ImageNet数据集，使用batch size 32
        actual_batch_size = 32 if dataset_name == "imagenet" else batch_size
        
        # 生成简洁的日志文件名
        if experiment_name:
            log_filename = f"{experiment_name}_{shots}shot.txt"
        else:
            # 默认生成的文件名
            log_suffix = ""
            if use_skd:
                log_suffix = f"_skd_{skd_temperature}temp_{skd_smoothing}smooth_{skd_lambda}lambda"
            if not use_glr:
                log_suffix += "_no_glr"
            if fixed_alpha is not None:
                log_suffix += f"_fixed_alpha_{fixed_alpha}"
            
            log_filename = f"custom_{dataset_name}_cma_{cma_start}-{cma_end}_{cma_dim}dim_{cma_scale}scale_{shots}shot{log_suffix}.txt"
        
        log_txt_path = os.path.join(os.environ['LOG_ROOT'], log_filename)
        # 确保日志目录存在
        os.makedirs(os.path.dirname(log_txt_path), exist_ok=True)
        
        # 运行实验: 使用指定的CMA配置
        all_experiment.experiment_one(
            shots=shots, 
            backbone="ViT-B/16", 
            train_epoch=50, 
            has_ood=False,  # 修改默认值为False
            use_cma=True, 
            cma_start_layer=cma_start,
            cma_end_layer=cma_end,
            cma_dim=cma_dim,
            cma_scale=cma_scale,
            use_skd=use_skd,
            skd_temperature=skd_temperature,
            skd_smoothing=skd_smoothing,
            skd_lambda=skd_lambda,
            use_glr=use_glr,  # 添加GLR参数
            fixed_alpha=fixed_alpha,  # 添加固定alpha参数
            fixed_weight=fixed_weight,  # 添加固定weight参数
            batch_size=actual_batch_size,  # 设置batch size
            log_txt_path=log_txt_path
        )
    
    print("=" * 50)
    print(f"{dataset_name} 数据集实验完成!")
    print(f"结果已保存到: {os.environ['LOG_ROOT']}")

def run_all_datasets_separately(cma_start=5, cma_end=12, cma_dim=32, cma_scale=0.01,
                               use_skd=False, skd_temperature=2.0, skd_smoothing=0.2, skd_lambda=0.01,
                               shots_list=[1, 2, 4, 8, 16], exclude_imagenet=True, use_glr=True, fixed_alpha=None, fixed_weight=None, batch_size=16, experiment_name=None):
    """分别运行所有数据集的实验"""
    datasets = [
        "fgvc",
        "caltech101", 
        "stanford_cars",
        "dtd",
        "eurosat",
        "oxford_flowers",
        "food101",
        "oxford_pets",
        "sun397",
        "ucf101"
    ]
    
    print("开始分别运行所有数据集实验...")
    print(f"设置: shots_list={shots_list}, 训练轮数=50")
    print(f"CMA配置: 层{cma_start}-{cma_end}, 维度{cma_dim}, 缩放因子{cma_scale}")
    if use_skd:
        print(f"SKD配置: 温度={skd_temperature}, smoothing={skd_smoothing}, lambda={skd_lambda}")
    if exclude_imagenet:
        print("注意: 排除ImageNet数据集")
    print("=" * 50)
    
    for dataset in datasets:
        try:
            run_single_dataset(dataset, cma_start, cma_end, cma_dim, cma_scale,
                             use_skd, skd_temperature, skd_smoothing, skd_lambda, shots_list, use_glr, fixed_alpha, fixed_weight, batch_size, experiment_name)
        except Exception as e:
            print(f"运行 {dataset} 数据集时出错: {e}")

def list_available_datasets():
    """列出所有可用的数据集"""
    datasets = [
        "fgvc",
        "caltech101", 
        "stanford_cars",
        "dtd",
        "eurosat",
        "oxford_flowers",
        "food101",
        "oxford_pets",
        "sun397",
        "ucf101"
    ]
    
    print("可用的数据集:")
    for i, dataset in enumerate(datasets, 1):
        print(f"{i}. {dataset}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run custom experiments with configurable CMA parameters')
    parser.add_argument('--dataset', type=str, help='Run experiment on specific dataset')
    parser.add_argument('--all', action='store_true', help='Run all datasets together')
    parser.add_argument('--separate', action='store_true', help='Run all datasets separately')
    parser.add_argument('--list', action='store_true', help='List all available datasets')
    parser.add_argument('--cma-start', type=int, default=5, help='CMA start layer (default: 5)')
    parser.add_argument('--cma-end', type=int, default=12, help='CMA end layer (default: 12)')
    parser.add_argument('--cma-dim', type=int, default=32, help='CMA adapter dimension (default: 32)')
    parser.add_argument('--cma-scale', type=float, default=0.01, help='CMA adapter scale (default: 0.01)')
    
    # GLR argument
    parser.add_argument('--no-glr', action='store_true', help='Disable GLR module (use only CMA)')
    
    # Fixed alpha for GLR
    parser.add_argument('--fixed-alpha', type=float, default=None, help='Fixed alpha value for GLR fusion gate (None means adaptive)')
    # Fixed weight for DLF
    parser.add_argument('--fixed-weight', type=float, default=None, help='Fixed weight value for DLF fusion (None means dynamic)')
    
    # SKD arguments
    parser.add_argument('--skd', action='store_true', help='Enable SKD distillation')
    parser.add_argument('--skd-temp', type=float, default=2.0, help='SKD temperature (default: 2.0)')
    parser.add_argument('--skd-smoothing', type=float, default=0.2, help='SKD smoothing (default: 0.2)')
    parser.add_argument('--skd-lambda', type=float, default=0.01, help='SKD loss weight (default: 0.01)')
    
    parser.add_argument('--shots', type=str, default="1,2,4,8,16", help='Shots to run, comma separated (default: 1,2,4,8,16)')
    parser.add_argument('--include-imagenet', action='store_true', help='Include ImageNet dataset (requires special setup)')
    parser.add_argument('--experiment-name', type=str, default=None, help='Experiment name for output file (e.g., CMA, CMA+GLR, CMA+SKD, CMA+GLR+SKD)')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size (default: 16)')
    
    args = parser.parse_args()
    
    # 解析shots参数
    shots_list = [int(x) for x in args.shots.split(',')]
    
    # GLR设置（默认启用，除非指定了--no-glr）
    use_glr = not args.no_glr
    
    if args.list:
        list_available_datasets()
    elif args.all:
        run_custom_experiment(args.cma_start, args.cma_end, args.cma_dim, args.cma_scale,
                            args.skd, args.skd_temp, args.skd_smoothing, args.skd_lambda,
                            shots_list, not args.include_imagenet, use_glr, args.fixed_alpha, args.fixed_weight, args.experiment_name, args.batch_size)
    elif args.separate:
        run_all_datasets_separately(args.cma_start, args.cma_end, args.cma_dim, args.cma_scale,
                                  args.skd, args.skd_temp, args.skd_smoothing, args.skd_lambda,
                                  shots_list, not args.include_imagenet, use_glr, args.fixed_alpha, args.fixed_weight, args.batch_size, args.experiment_name)
    elif args.dataset:
        run_single_dataset(args.dataset, args.cma_start, args.cma_end, args.cma_dim, args.cma_scale,
                         args.skd, args.skd_temp, args.skd_smoothing, args.skd_lambda,
                         shots_list, use_glr, args.fixed_alpha, args.fixed_weight, args.batch_size, args.experiment_name)
    else:
        print("用法:")
        print("  python run_custom_experiment.py --all [--cma-start START] [--cma-end END] [--cma-dim DIM] [--cma-scale SCALE] [--no-glr] [--skd] [--skd-temp TEMP] [--skd-smoothing SMOOTHING] [--skd-lambda LAMBDA] [--shots SHOTS] [--include-imagenet] [--batch-size SIZE]")
        print("  python run_custom_experiment.py --separate [--cma-start START] [--cma-end END] [--cma-dim DIM] [--cma-scale SCALE] [--no-glr] [--skd] [--skd-temp TEMP] [--skd-smoothing SMOOTHING] [--skd-lambda LAMBDA] [--shots SHOTS] [--include-imagenet] [--batch-size SIZE]")
        print("  python run_custom_experiment.py --dataset DATASET [--cma-start START] [--cma-end END] [--cma-dim DIM] [--cma-scale SCALE] [--no-glr] [--skd] [--skd-temp TEMP] [--skd-smoothing SMOOTHING] [--skd-lambda LAMBDA] [--shots SHOTS] [--batch-size SIZE]")
        print("  python run_custom_experiment.py --list")
        print()
        print("参数说明:")
        print("  --cma-start START  CMA起始层 (默认: 5)")
        print("  --cma-end END      CMA结束层 (默认: 12)")
        print("  --cma-dim DIM      CMA适配器维度 (默认: 32)")
        print("  --cma-scale SCALE  CMA适配器缩放因子 (默认: 0.01)")
        print("  --no-glr           禁用GLR模块 (仅使用CMA)")
        print("  --fixed-alpha      固定GLR融合门控的alpha值 (默认: None，表示自适应)")
        print("  --skd              启用SKD蒸馏")
        print("  --skd-temp TEMP    SKD温度参数 (默认: 2.0)")
        print("  --skd-smoothing SMOOTHING  SKD平滑参数 (默认: 0.2)")
        print("  --skd-lambda LAMBDA SKD损失权重 (默认: 0.01)")
        print("  --shots SHOTS      要运行的shots，逗号分隔 (默认: 1,2,4,8,16)")
        print("  --include-imagenet 包含ImageNet数据集 (需要特殊设置)")
        print("  --batch-size SIZE  批大小 (默认: 16)")
        print()
        list_available_datasets()
        # python run_custom_experiment.py --dataset eurosat --cma-start 5 --cma-end 12 --cma-dim 32 --cma-scale 0.01 --skd --skd-temp 2.0 --skd-smoothing 0.2 --skd-lambda 0.01 --shots 16