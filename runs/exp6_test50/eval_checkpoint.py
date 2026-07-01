#!/usr/bin/env python3
"""
Exp6 续训评估脚本
对单个 checkpoint 进行 val 和 test 评估，输出：
- 整体指标 (mAP50, mAP50-95, Precision, Recall)
- 分类指标 (所有类别 AP)
- Head/Middle/Tail AP 分组统计
- COCO Scale AP (Small/Medium/Large)
"""

import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'ultralytics_src'))

from ultralytics import YOLO

# ── 类别定义 ──
CLASS_NAMES = {
    0: 'Bus', 1: 'Cargo Truck', 2: 'Dump Truck', 3: 'Excavator',
    4: 'Small Car', 5: 'Tractor', 6: 'Trailer', 7: 'Truck Tractor',
    8: 'Van', 9: 'other-vehicle',
}

# 按训练样本量分组 (来自 Exp6 分析)
HEAD_CLASSES = {'Small Car', 'Van'}                          # >20000 样本
MIDDLE_CLASSES = {'Dump Truck', 'Cargo Truck', 'other-vehicle'}  # 500-6500
TAIL_CLASSES = {'Bus', 'Excavator', 'Tractor', 'Trailer', 'Truck Tractor'}  # <500

# COCO Scale 阈值 (像素面积)
# 注意: OBB 使用 bbox 面积 (width * height), 非 mask 面积
SCALE_THRESHOLDS = {
    'Small':  (0,     32**2),   # area < 1024
    'Medium': (32**2, 96**2),   # 1024 <= area < 9216
    'Large':  (96**2, float('inf')),  # area >= 9216
}


def classify_scale(area):
    """根据 bbox 面积判断 COCO scale"""
    if area < 32**2:
        return 'Small'
    elif area < 96**2:
        return 'Medium'
    else:
        return 'Large'


def run_evaluation(weights_path, data_yaml, split='val', imgsz=800, batch=16, name=''):
    """运行评估并返回原始结果字典"""
    model = YOLO(str(weights_path))
    metrics = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=imgsz,
        batch=batch,
        plots=False,
        verbose=False,
    )
    return metrics


def parse_metrics(metrics, split_name):
    """从 ultralytics Metrics 对象提取结构化数据"""
    # f1 可能是 per-class array 而非标量
    f1_val = float(metrics.box.f1.mean()) if hasattr(metrics.box.f1, '__len__') else float(metrics.box.f1)
    result = {
        'split': split_name,
        'mAP50': float(metrics.box.map50),
        'mAP50-95': float(metrics.box.map),
        'precision': float(metrics.box.mp),
        'recall': float(metrics.box.mr),
        'f1': f1_val,
        'per_class_ap50': {},    # 每个类别的 AP50
        'per_class_ap50_95': {}, # 每个类别的 AP50-95
    }

    # 提取各类别 AP
    # ultralytics Metrics 中 ap_class_index 是类别索引列表
    # box.ap 是各类别的 AP50-95, box.ap50 是各类别的 AP50 (如果存在)
    if hasattr(metrics.box, 'ap_class_index') and metrics.box.ap_class_index is not None:
        for i, cls_idx in enumerate(metrics.box.ap_class_index):
            cls_name = CLASS_NAMES.get(int(cls_idx), f'class_{cls_idx}')
            if hasattr(metrics.box, 'ap') and i < len(metrics.box.ap):
                result['per_class_ap50_95'][cls_name] = float(metrics.box.ap[i])
            if hasattr(metrics.box, 'ap50') and i < len(metrics.box.ap50):
                result['per_class_ap50'][cls_name] = float(metrics.box.ap50[i])

    return result


def compute_group_metrics(parsed):
    """计算 Head/Middle/Tail 分组指标"""
    groups = {'Head': [], 'Middle': [], 'Tail': []}

    for cls_name, ap in parsed['per_class_ap50_95'].items():
        if cls_name in HEAD_CLASSES:
            groups['Head'].append(ap)
        elif cls_name in MIDDLE_CLASSES:
            groups['Middle'].append(ap)
        elif cls_name in TAIL_CLASSES:
            groups['Tail'].append(ap)

    result = {}
    for group_name, aps in groups.items():
        if aps:
            result[f'{group_name}_AP'] = sum(aps) / len(aps)
            result[f'{group_name}_count'] = len(aps)
        else:
            result[f'{group_name}_AP'] = 0.0
            result[f'{group_name}_count'] = 0

    return result


def evaluate_checkpoint(weights_path, fold=0, checkpoint_label=''):
    """完整评估一个 checkpoint: val + test"""
    data_dir = PROJECT_ROOT / 'dataset_yolo' / f'fold_{fold}'

    print(f"\n{'='*60}")
    print(f"📊 评估 Checkpoint: {checkpoint_label or weights_path.stem}")
    print(f"   权重: {weights_path}")
    print(f"{'='*60}")

    all_results = {}

    # Val 评估
    print("\n🔍 Val 集评估...")
    val_metrics = run_evaluation(weights_path, data_dir / 'data.yaml', split='val')
    val_parsed = parse_metrics(val_metrics, 'val')
    all_results['val'] = val_parsed

    # Test 评估
    print("🔍 Test 集评估...")
    test_metrics = run_evaluation(weights_path, data_dir / 'data.yaml', split='test')
    test_parsed = parse_metrics(test_metrics, 'test')
    all_results['test'] = test_parsed

    # 打印结果
    for split_name, parsed in [('Val', val_parsed), ('Test', test_parsed)]:
        groups = compute_group_metrics(parsed)
        print(f"\n{'─'*40}")
        print(f"📋 {split_name} 集结果:")
        print(f"   mAP50:     {parsed['mAP50']:.4f}")
        print(f"   mAP50-95:  {parsed['mAP50-95']:.4f}")
        print(f"   Precision: {parsed['precision']:.4f}")
        print(f"   Recall:    {parsed['recall']:.4f}")
        print(f"\n   📊 分组 AP50-95:")
        print(f"   Head   ({groups.get('Head_count', 0)} 类): {groups.get('Head_AP', 0):.4f}")
        print(f"   Middle ({groups.get('Middle_count', 0)} 类): {groups.get('Middle_AP', 0):.4f}")
        print(f"   Tail   ({groups.get('Tail_count', 0)} 类): {groups.get('Tail_AP', 0):.4f}")
        print(f"\n   📋 各类别 AP50-95:")
        for cls_name in sorted(parsed['per_class_ap50_95'].keys()):
            ap = parsed['per_class_ap50_95'][cls_name]
            tag = ''
            if cls_name in HEAD_CLASSES:
                tag = ' [Head]'
            elif cls_name in MIDDLE_CLASSES:
                tag = ' [Mid]'
            elif cls_name in TAIL_CLASSES:
                tag = ' [Tail]'
            print(f"     {cls_name:20s}: {ap:.4f}{tag}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description='Exp6 Checkpoint 评估')
    parser.add_argument('--weights', type=str, required=True, help='Checkpoint 路径')
    parser.add_argument('--fold', type=int, default=0, help='Fold 编号')
    parser.add_argument('--label', type=str, default='', help='Checkpoint 标签 (如 ep75)')
    parser.add_argument('--output', type=str, default=None, help='输出 JSON 路径')
    args = parser.parse_args()

    results = evaluate_checkpoint(
        weights_path=Path(args.weights),
        fold=args.fold,
        checkpoint_label=args.label or Path(args.weights).stem,
    )

    # 保存 JSON
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(args.weights).parent.parent / f'eval_{args.label or Path(args.weights).stem}.json'

    # 为 JSON 序列化转换
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n📁 评估结果已保存至: {output_path}")

    return results


if __name__ == '__main__':
    main()
