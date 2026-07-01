#!/usr/bin/env python3
"""
准备下一个训练 chunk 的配置文件。
每完成一个 chunk，根据最新 best.pt 生成下一个 25-epoch chunk 配置。
"""

import sys
import yaml
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.parent
CHUNK_EPOCHS = 25  # 每个 chunk 的 epoch 数

# 基础配置模板 (与 exp6_test50 保持一致)
BASE_CONFIG = {
    'task': 'obb',
    'mode': 'train',
    'data': 'dataset_yolo/fold_0/data.yaml',
    'epochs': CHUNK_EPOCHS,
    'batch': 96,
    'imgsz': 800,
    'workers': 24,
    'device': '0,1,2,3,4,5,6,7',
    'seed': 42,
    'deterministic': True,
    'optimizer': 'MuSGD',
    'lr0': 0.0003,
    'lrf': 0.5,
    'momentum': 0.9,
    'weight_decay': 0.0005,
    'warmup_epochs': 1.0,
    'warmup_momentum': 0.8,
    'warmup_bias_lr': 0.0001,
    'cos_lr': True,
    'nbs': 64,
    'close_mosaic': 30,
    'amp': True,
    'patience': 100,
    'save_period': CHUNK_EPOCHS,
    'box': 7.5,
    'cls': 0.5,
    'cls_pw': 0.5,
    'dfl': 1.5,
    'angle': 1.0,
    'mosaic': 1.0,
    'mosaic9': 0.2,
    'mixup': 0.1,
    'hsv_h': 0.015,
    'hsv_s': 0.3,
    'hsv_v': 0.2,
    'translate': 0.1,
    'scale': 0.5,
    'fliplr': 0.5,
    'use_focal': False,
    'use_wise_iou': False,
}


def create_chunk_config(chunk_num, source_best_pt, total_epochs_so_far):
    """生成第 N 个 chunk 的训练配置"""
    target_total = total_epochs_so_far + CHUNK_EPOCHS
    name = f'exp6_ep{target_total}'
    config_path = PROJECT_ROOT / 'configs' / f'exp6_chunk{chunk_num}_{name}.yaml'

    config = BASE_CONFIG.copy()
    config['model'] = str(source_best_pt)
    config['epochs'] = CHUNK_EPOCHS

    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"✅ 配置已生成: {config_path}")
    print(f"   源模型:    {source_best_pt}")
    print(f"   目标:      总 {target_total} epoch")
    print(f"   实验名:    {name}")

    return config_path, name


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("用法: python prepare_next_chunk.py <chunk_num> <source_best_pt> <total_epochs_so_far>")
        sys.exit(1)

    chunk_num = int(sys.argv[1])
    source_best_pt = Path(sys.argv[2])
    total_epochs = int(sys.argv[3])

    create_chunk_config(chunk_num, source_best_pt, total_epochs)
