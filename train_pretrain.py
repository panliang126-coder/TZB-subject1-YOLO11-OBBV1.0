#!/usr/bin/env python3
"""
Experiment 1: 官方预训练 yolo11n → OBB 微调 (TODO01)

通过 pretrained 参数，框架 setup_model() 自动处理 detect→OBB 权重迁移:
- OBB 架构从 yolo11n-obb.yaml 创建 (正确 Head)
- backbone 权重从 models/yolo11n.pt intersect 加载 (COCO 预训练)
- OBB head 自动随机初始化

所有训练参数与 baseline.yaml 保持一致，唯一变量: 预训练权重。
"""

import sys
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / 'ultralytics_src'))

from ultralytics import YOLO


def main():
    fold = 0
    device = '0,1,2,3'

    # ── 1. 加载 baseline 训练参数 ──
    with open(PROJECT_ROOT / 'configs' / 'baseline.yaml') as f:
        args = yaml.safe_load(f)
    args = {k: v for k, v in args.items() if v is not None}
    args['data'] = str(PROJECT_ROOT / f'dataset_yolo/fold_{fold}/data.yaml')
    args['device'] = device
    args['name'] = 'pretrain_yolo11n_fold0'
    args['project'] = str(PROJECT_ROOT / 'runs')
    args['exist_ok'] = True

    # ── 2. 指定官方预训练权重 (detect → OBB 迁移) ──
    pretrained_path = str(PROJECT_ROOT / 'models' / 'yolo11n.pt')
    args['pretrained'] = pretrained_path

    # ── 3. 创建 OBB 模型 (架构正确，但最终会被 setup_model() 重建并加载权重) ──
    model = YOLO('yolo11n-obb.yaml', task='obb')

    # ── 4. 打印配置 ──
    print("=" * 70)
    print("🚀 Experiment 1: 官方预训练 yolo11n → OBB 微调 (TODO01)")
    print("=" * 70)
    print(f"  实验名称:     {args['name']}")
    print(f"  预训练权重:   {pretrained_path}")
    print(f"  架构:         yolo11n-obb.yaml")
    print(f"  Backbone:     COCO 预训练 (yolo11n.pt)")
    print(f"  OBB Head:     随机初始化")
    print(f"  数据:         {args['data']}")
    print(f"  Fold:         0")
    print(f"  Epochs:       {args['epochs']}")
    print(f"  Batch:        {args['batch']}")
    print(f"  ImgSz:        {args['imgsz']}")
    print(f"  Device:       {device}")
    print(f"  Workers:      {args['workers']}")
    print("=" * 70)

    # ── 5. 训练 ──
    results = model.train(**args)

    print("\n✅ 训练完成!")
    print(f"  最佳模型: {results}")

    return results


if __name__ == '__main__':
    main()
