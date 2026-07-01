#!/usr/bin/env python3
"""
TODO01: 官方预训练模型 + Enhanced 配置微调

Experiment 1-3: yolo11n → yolo11s → yolo11m
全部使用 enhanced.yaml 训练参数 (P2 + imgsz=800 + cos_lr + multi_scale)
唯一变量: 模型容量 + 预训练权重

用法:
    python train_pretrain_enhanced.py --model n    # yolo11n + yolo11n.pt
    python train_pretrain_enhanced.py --model s    # yolo11s + yolo11s.pt
    python train_pretrain_enhanced.py --model m    # yolo11m + yolo11m.pt
    python train_pretrain_enhanced.py --model all  # 依次跑 n → s → m
"""

import sys
import yaml
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / 'ultralytics_src'))

from ultralytics import YOLO


MODEL_CONFIGS = {
    'n': {
        'model_yaml': 'yolo11n-obb-p2.yaml',
        'pretrained': 'yolo11n.pt',
        'name': 'pretrain_enhanced_yolo11n_fold0',
    },
    's': {
        'model_yaml': 'yolo11s-obb-p2.yaml',
        'pretrained': 'yolo11s.pt',
        'name': 'pretrain_enhanced_yolo11s_fold0',
    },
    'm': {
        'model_yaml': 'yolo11m-obb-p2.yaml',
        'pretrained': 'yolo11m.pt',
        'name': 'pretrain_enhanced_yolo11m_fold0',
    },
    'l': {
        'model_yaml': 'yolo11l-obb-p2.yaml',
        'pretrained': 'yolo11l.pt',
        'name': 'pretrain_enhanced_yolo11l_fold0',
    },
}


def run_experiment(model_size: str, fold: int = 0, device: str = '0,1,2,3'):
    cfg = MODEL_CONFIGS[model_size]
    pretrained_path = str(PROJECT_ROOT / 'models' / cfg['pretrained'])

    # ── 加载 enhanced 训练参数 ──
    with open(PROJECT_ROOT / 'configs' / 'enhanced.yaml') as f:
        args = yaml.safe_load(f)
    args = {k: v for k, v in args.items() if v is not None}
    args['model'] = cfg['model_yaml']
    args['data'] = str(PROJECT_ROOT / f'dataset_yolo/fold_{fold}/data.yaml')
    args['device'] = device
    args['name'] = cfg['name']
    args['project'] = str(PROJECT_ROOT / 'runs')
    args['exist_ok'] = True
    args['pretrained'] = pretrained_path

    # ── 打印配置 ──
    print("=" * 70)
    print(f"🚀 TODO01 Experiment: yolo11{model_size} + pretrained + enhanced")
    print("=" * 70)
    print(f"  实验名称:     {cfg['name']}")
    print(f"  模型架构:     {cfg['model_yaml']}")
    print(f"  预训练权重:   {pretrained_path}")
    print(f"  配置来源:     configs/enhanced.yaml")
    print(f"  Epochs:       {args['epochs']}")
    print(f"  Batch:        {args['batch']}")
    print(f"  ImgSz:        {args['imgsz']}")
    print(f"  Device:       {device}")
    print(f"  cos_lr:       {args['cos_lr']}")
    print(f"  multi_scale:  {args['multi_scale']}")
    print(f"  close_mosaic: {args['close_mosaic']}")
    print("=" * 70)

    # ── 创建模型 ──
    model = YOLO(cfg['model_yaml'], task='obb')

    # ── 训练 ──
    results = model.train(**args)
    print(f"\n✅ yolo11{model_size} 训练完成! 最佳: {results}")
    return results


def main():
    parser = argparse.ArgumentParser(description='TODO01: pretrained + enhanced 微调实验')
    parser.add_argument('--model', type=str, required=True,
                        choices=['n', 's', 'm', 'l', 'all'],
                        help='模型容量 (n/s/m/l 或 all 依次执行)')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--device', type=str, default='0,1,2,3')
    args = parser.parse_args()

    if args.model == 'all':
        for size in ['n', 's', 'm']:
            print(f"\n{'#'*70}")
            print(f"# 开始 Experiment: yolo11{size}")
            print(f"{'#'*70}\n")
            run_experiment(size, fold=args.fold, device=args.device)
    else:
        run_experiment(args.model, fold=args.fold, device=args.device)


if __name__ == '__main__':
    main()
