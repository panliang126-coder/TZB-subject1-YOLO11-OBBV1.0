#!/usr/bin/env python3
"""
YOLO11-OBB 遥感车辆检测增强版 - 训练脚本

支持 P2 检测层、Focal Loss、Wise-IoU、KLD Angle Loss、
Scale-aware Weight、Slide Loss 等全部增强功能。

每个优化点通过配置独立开关，方便 Ablation 实验。

用法:
    # Baseline (标准 YOLO11-OBB)
    python train.py --fold 0 --baseline

    # 增强版 (所有优化开启)
    python train.py --fold 0 --enhanced

    # Ablation: 只开启 P2
    python train.py --fold 0 --model yolo11n-obb-p2.yaml

    # Ablation: P2 + Focal Loss
    python train.py --fold 0 --model yolo11n-obb-p2.yaml --use-focal

    # 完整配置
    python train.py --fold 0 --enhanced --epochs 300 --batch 64 --imgsz 800
"""

import argparse
import sys
import yaml
import torch
from pathlib import Path

# 将本地 ultralytics 添加到路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / 'ultralytics_src'))

from ultralytics import YOLO


def _load_yaml_config(yaml_path):
    """从 YAML 文件加载训练配置"""
    with open(yaml_path) as f:
        config = yaml.safe_load(f)
    return {k: v for k, v in config.items() if v is not None}


def get_baseline_args(fold=0):
    """Baseline 配置 (从 configs/baseline.yaml 加载)"""
    args = _load_yaml_config(str(PROJECT_ROOT / 'configs' / 'baseline.yaml'))
    args['data'] = str(PROJECT_ROOT / f'dataset_yolo/fold_{fold}/data.yaml')
    return args


def get_enhanced_args(fold=0):
    """增强版配置 (从 configs/enhanced.yaml 加载)"""
    args = _load_yaml_config(str(PROJECT_ROOT / 'configs' / 'enhanced.yaml'))
    args['data'] = str(PROJECT_ROOT / f'dataset_yolo/fold_{fold}/data.yaml')
    return args


def main():
    parser = argparse.ArgumentParser(
        description='YOLO11-OBB 遥感车辆检测增强版训练',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # Baseline 标准训练
  python train.py --fold 0 --baseline

  # 增强版 (P2 + Focal + 数据增强)
  python train.py --fold 0 --enhanced

  # Ablation: P2 only
  python train.py --fold 0 --model yolo11n-obb-p2.yaml

  # Ablation: P2 + Focal
  python train.py --fold 0 --model yolo11n-obb-p2.yaml --use-focal

  # 自定义配置
  python train.py --fold 0 --model yolo11n-obb-p2.yaml --epochs 300 --batch 128 --imgsz 1024
        """,
    )
    # ── 预设模式 ──
    preset = parser.add_mutually_exclusive_group()
    preset.add_argument('--baseline', action='store_true', help='使用 Baseline 配置 (configs/baseline.yaml)')
    preset.add_argument('--enhanced', action='store_true', help='使用增强版配置 (configs/enhanced.yaml)')

    # ── 自定义 YAML ──
    parser.add_argument('--cfg', type=str, default=None, help='使用自定义 YAML 配置文件')

    # ── 基础配置 ──
    parser.add_argument('--fold', type=int, default=0, help='使用哪个 fold (0-4)')
    parser.add_argument('--model', type=str, default=None, help='模型 YAML 路径或名称')
    parser.add_argument('--epochs', type=int, default=None, help='训练轮数')
    parser.add_argument('--batch', type=int, default=None, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=None, help='输入尺寸')
    parser.add_argument('--device', type=str, default=None, help='设备 (例如: 0,1,2,3)')
    parser.add_argument('--workers', type=int, default=None, help='数据加载进程数')
    parser.add_argument('--fraction', type=float, default=None, help='训练数据比例 (调试用, 例: 0.01)')
    parser.add_argument('--cache', type=str, default=None, choices=['ram', 'disk'], help='缓存图像到 ram/disk 加速训练')
    parser.add_argument('--profile-loader', action='store_true', default=None, help='开启 DataLoader Profiling')
    parser.add_argument('--resume', action='store_true', help='从检查点恢复')

    # ── 增强开关 (Ablation 用) ──
    loss_group = parser.add_argument_group('Loss 优化')
    loss_group.add_argument('--use-focal', action='store_true', default=None, help='开启 Focal Loss')
    loss_group.add_argument('--no-focal', action='store_true', help='关闭 Focal Loss')
    loss_group.add_argument('--use-wise-iou', action='store_true', default=None, help='开启 Wise-IoU')
    loss_group.add_argument('--use-kld-angle', action='store_true', default=None, help='开启 KLD Angle Loss')
    loss_group.add_argument('--use-slide-loss', action='store_true', default=None, help='开启 Slide Loss')
    loss_group.add_argument('--use-scale-aware', action='store_true', default=None, help='开启 Scale-aware Weight')

    aug_group = parser.add_argument_group('数据增强')
    aug_group.add_argument('--mosaic9', type=float, default=None, help='3x3 Mosaic 概率')
    aug_group.add_argument('--scale-aware-mosaic', type=float, default=None, help='Scale-aware Mosaic 概率')
    aug_group.add_argument('--small-obj-copy-paste', type=float, default=None, help='小目标 CopyPaste 概率')
    aug_group.add_argument('--mixup', type=float, default=None, help='MixUp 概率')

    # ── 实验名称 ──
    parser.add_argument('--name', type=str, default=None, help='实验名称')

    args_parsed = parser.parse_args()

    # ── 加载 YAML 配置 ──
    if args_parsed.cfg:
        args = _load_yaml_config(args_parsed.cfg)
        args['data'] = str(PROJECT_ROOT / f'dataset_yolo/fold_{args_parsed.fold}/data.yaml')
    elif args_parsed.enhanced:
        args = get_enhanced_args(fold=args_parsed.fold)
    elif args_parsed.baseline:
        args = get_baseline_args(fold=args_parsed.fold)
    else:
        # 默认使用 baseline
        args = get_baseline_args(fold=args_parsed.fold)

    # ── CLI 参数覆盖 YAML 值 ──
    if args_parsed.epochs:
        args['epochs'] = args_parsed.epochs
    if args_parsed.batch:
        args['batch'] = args_parsed.batch
    if args_parsed.imgsz:
        args['imgsz'] = args_parsed.imgsz
    if args_parsed.device:
        args['device'] = args_parsed.device
    if args_parsed.workers:
        args['workers'] = args_parsed.workers
    if args_parsed.fraction is not None:
        args['fraction'] = args_parsed.fraction
    if args_parsed.cache:
        args['cache'] = args_parsed.cache
    if args_parsed.profile_loader:
        args['profile_loader'] = True
    if args_parsed.model:
        args['model'] = args_parsed.model

    # Loss 开关覆盖
    if args_parsed.use_focal:
        args['use_focal'] = True
    if args_parsed.no_focal:
        args['use_focal'] = False
    if args_parsed.use_wise_iou is not None:
        args['use_wise_iou'] = args_parsed.use_wise_iou
    if args_parsed.use_kld_angle is not None:
        args['use_kld_angle'] = args_parsed.use_kld_angle
    if args_parsed.use_slide_loss is not None:
        args['use_slide_loss'] = args_parsed.use_slide_loss
    if args_parsed.use_scale_aware is not None:
        args['use_scale_aware'] = args_parsed.use_scale_aware

    # 增强参数覆盖
    if args_parsed.mosaic9 is not None:
        args['mosaic9'] = args_parsed.mosaic9
    if args_parsed.scale_aware_mosaic is not None:
        args['scale_aware_mosaic'] = args_parsed.scale_aware_mosaic
    if args_parsed.small_obj_copy_paste is not None:
        args['small_obj_copy_paste'] = args_parsed.small_obj_copy_paste
    if args_parsed.mixup is not None:
        args['mixup'] = args_parsed.mixup

    # 实验名称
    if args_parsed.name:
        args['name'] = args_parsed.name
    else:
        # 自动生成名称
        model_name = Path(args['model']).stem
        features = []
        if args['use_focal']:
            features.append('focal')
        if args.get('use_wise_iou'):
            features.append('wiou')
        if args.get('use_kld_angle'):
            features.append('kld')
        if args.get('use_slide_loss'):
            features.append('slide')
        if args.get('use_scale_aware'):
            features.append('scaleaw')
        if args.get('mosaic9', 0) > 0:
            features.append('m9')
        if args.get('small_obj_copy_paste', 0) > 0:
            features.append('socp')
        feature_str = '_'.join(features) if features else 'baseline'
        args['name'] = f"{model_name}_fold{args_parsed.fold}_{feature_str}"

    args['project'] = str(PROJECT_ROOT / 'runs')

    # ── 打印配置 ──
    print("=" * 70)
    print("🚀 YOLO11-OBB 遥感车辆检测增强版 - 训练配置")
    print("=" * 70)
    print(f"  实验名称:   {args['name']}")
    print(f"  模型:       {args['model']}")
    print(f"  数据:       {args['data']}")
    print(f"  Fold:       {args_parsed.fold}")
    print(f"  Epochs:     {args['epochs']}")
    print(f"  Batch:      {args['batch']}")
    print(f"  ImgSz:      {args['imgsz']}")
    print(f"  Device:     {args['device']}")
    print("-" * 70)
    print("  增强特性:")
    print(f"    P2 检测层:     {'p2' in str(args['model'])}")
    print(f"    Focal Loss:    {args['use_focal']}")
    print(f"    Wise-IoU:      {args.get('use_wise_iou', False)}")
    print(f"    KLD Angle:     {args.get('use_kld_angle', False)}")
    print(f"    Slide Loss:    {args.get('use_slide_loss', False)}")
    print(f"    Scale-aware:   {args.get('use_scale_aware', False)}")
    print(f"    3x3 Mosaic:    {args.get('mosaic9', 0)}")
    print(f"    ScaleAw Mosaic:{args.get('scale_aware_mosaic', 0)}")
    print(f"    SmallObj Copy: {args.get('small_obj_copy_paste', 0)}")
    print(f"    Class PW:      {args.get('cls_pw', 0)}")
    print(f"    Cos LR:        {args['cos_lr']}")
    print(f"    Multi-scale:   {args.get('multi_scale', 0)}")
    print("=" * 70)

    # ── 断点续训 ──
    if args_parsed.resume:
        resume_dir = args.get('project', str(PROJECT_ROOT / 'runs'))
        last_pt = Path(resume_dir) / (args['name'] + '/weights/last.pt' if args.get('name') else '')

        if last_pt.exists():
            # 1. 先备份已有权重，防止意外覆盖
            import shutil
            backup_dir = last_pt.parent / '.backup'
            backup_dir.mkdir(exist_ok=True)
            for f in list(last_pt.parent.glob('*.pt')) + list(last_pt.parent.parent.glob('results.csv')):
                if f.exists():
                    shutil.copy2(f, backup_dir / f.name)
            print(f"📦 已备份权重到 {backup_dir}")

            # 2. 检查 checkpoint 是否包含 optimizer（可续训）
            ckpt = torch.load(str(last_pt), map_location='cpu', weights_only=False)
            has_optimizer = 'optimizer' in ckpt

            if has_optimizer:
                # 完整 checkpoint → 使用 ultralytics 原生 resume
                print(f"🔄 断点续训 (含优化器状态): {last_pt}")
                model = YOLO(str(last_pt))
                args['resume'] = True
                args['exist_ok'] = True
            else:
                # 训练已完成，只有权重 → 加载权重，作为新训练起点
                print(f"🔁 训练已完成，加载权重继续训练: {last_pt}")
                print(f"   原 epoch: {ckpt.get('epoch', '未知')}")
                model = YOLO(str(last_pt))
                args['resume'] = False
                # 不改动已有目录，让 ultralytics 自动生成新目录名
        else:
            print(f"⚠️ 未找到 {last_pt}，从模型配置重新开始")
            model = YOLO(args['model'])
            args['resume'] = False
    else:
        model = YOLO(args['model'])

    results = model.train(**args)

    print("\n✅ 训练完成!")
    print(f"  最佳模型: {results}")

    return results


if __name__ == '__main__':
    main()
