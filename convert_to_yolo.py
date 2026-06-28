#!/usr/bin/env python3
"""
数据集转换脚本: JSON → YOLO OBB 格式

将 dataset_enhanced_new.json 中的 Polygon (4-point) 标注转换为 YOLO OBB 格式:
    class_id x_center y_center width height angle
    (所有坐标归一化到 [0, 1], angle 为弧度)

输入: dataset/fold_0/train.json 等 JSON 文件
输出: dataset_yolo/ 目录下的标准 YOLO OBB 数据集

用法:
    python convert_to_yolo.py --fold 0
    python convert_to_yolo.py --all  # 转换所有 fold
"""

import json
import math
import os
import sys
from pathlib import Path
import argparse

import cv2
import numpy as np

# 类别映射 (按字母顺序, 与 JSON 中的 lab 对应)
CLASS_NAMES = [
    "Bus",
    "Cargo Truck",
    "Dump Truck",
    "Excavator",
    "Small Car",
    "Tractor",
    "Trailer",
    "Truck Tractor",
    "Van",
    "other-vehicle",
]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}


def polygon_to_obb(points):
    """将 4 点 Polygon 转为 OBB (cx, cy, w, h, angle_rad)

    Args:
        points: list of [x, y], 前 4 个点 (已排除第 5 个重复点)

    Returns:
        (cx, cy, w, h, angle_rad) in pixel coordinates
    """
    pts = np.array(points[:4], dtype=np.float32)
    rect = cv2.minAreaRect(pts)  # ((cx,cy), (w,h), angle)
    (cx, cy), (w, h), angle = rect

    # cv2.minAreaRect 返回的 angle 范围是 [-90, 0), w 是最先遇到的边
    # 我们需要统一: angle in [-pi/2, pi/2), w >= h
    # 转换: OpenCV angle -> YOLO OBB angle (弧度, [-pi/2, pi/2))
    # 如果 w < h, 交换 w/h 并调整 angle
    # if w < h:
    #     w, h = h, w
    #     angle = angle + 90

    # 转换到 YOLO 约定: angle = -angle_deg * pi / 180
    angle_rad = -angle * math.pi / 180.0

    # 处理 w < h 的情况
    if w < h:
        w, h = h, w
        # 如果 angle < 0 加 pi/2 否则减 pi/2
        angle_rad = angle_rad + math.pi / 2 if angle_rad < 0 else angle_rad - math.pi / 2

    # 归一化 angle 到 [-pi/4, 3*pi/4)
    if angle_rad < -math.pi / 4:
        angle_rad += math.pi / 2
        w, h = h, w
    elif angle_rad >= 3 * math.pi / 4:
        angle_rad -= math.pi / 2
        w, h = h, w

    return float(cx), float(cy), float(w), float(h), angle_rad


def convert_json_to_yolo(json_path, output_dir, img_size=None):
    """转换单个 JSON 文件到 YOLO OBB 格式

    Args:
        json_path: JSON 文件路径
        output_dir: 输出目录 (图片 + labels)
        img_size: 默认图像大小 (w, h), 如果无法读取图片则使用

    Returns:
        (num_images, num_annotations)
    """
    with open(json_path, 'r') as f:
        dataset = json.load(f)

    annotations = dataset['data'] if isinstance(dataset, dict) else dataset

    # 按图片分组
    img_annotations = {}
    for ann in annotations:
        img_path = ann['data_path']
        if img_path not in img_annotations:
            img_annotations[img_path] = []
        img_annotations[img_path].append(ann)

    labels_dir = Path(output_dir) / 'labels'
    images_dir = Path(output_dir) / 'images'
    labels_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    num_images = 0
    num_annotations = 0
    errors = 0

    for img_path, anns in img_annotations.items():
        # 读取图片获取尺寸
        try:
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                if img_size:
                    h, w = img_size[1], img_size[0]
                else:
                    errors += 1
                    continue
            else:
                h, w = img.shape[:2]
        except Exception:
            if img_size:
                h, w = img_size[1], img_size[0]
            else:
                errors += 1
                continue

        # 生成唯一文件名
        img_name = Path(img_path).stem
        safe_name = img_path.replace('/', '_').replace('\\', '_').replace('.tif', '').replace(' ', '_')
        safe_name = safe_name.replace(':', '_')

        # 创建符号链接或复制图像到 images 目录
        src_img = Path(img_path)
        dst_img = images_dir / f"{safe_name}.tif"
        if not dst_img.exists():
            try:
                os.symlink(src_img.absolute(), dst_img)
            except OSError:
                try:
                    # 如果符号链接失败, 尝试复制
                    import shutil
                    shutil.copy2(src_img, dst_img)
                except Exception:
                    pass

        # 写入 YOLO OBB 标签文件 (9列格式: class_id x1 y1 x2 y2 x3 y3 x4 y4)
        label_file = labels_dir / f"{safe_name}.txt"
        with open(label_file, 'w') as f:
            for ann in anns:
                class_name = ann['lab']
                class_id = CLASS_TO_ID.get(class_name, -1)
                if class_id < 0:
                    continue

                # 取前 4 个点 (points 是 5 点闭合, 最后一点与第一点相同)
                pts = np.array(ann['points'][:4], dtype=np.float32)

                # 归一化坐标
                pts_norm = pts.copy()
                pts_norm[:, 0] /= w
                pts_norm[:, 1] /= h

                # 写入 YOLO OBB 格式: class_id x1 y1 x2 y2 x3 y3 x4 y4
                pts_flat = pts_norm.reshape(-1)
                f.write(f"{class_id} {' '.join(f'{p:.6f}' for p in pts_flat)}\n")
                num_annotations += 1

        num_images += 1

    if errors > 0:
        print(f"  ⚠️  {errors} images could not be read (no size info)")

    return num_images, num_annotations


def create_data_yaml(output_dir, train_path, val_path, test_path=None):
    """创建 YOLO data.yaml 配置文件"""
    yaml_content = f"""# YOLO OBB 遥感车辆检测数据集配置

# 数据集路径
path: {Path(output_dir).absolute()}
train: {train_path}
val: {val_path}
test: {test_path if test_path else ''}

# 类别
names:
  0: Bus
  1: Cargo Truck
  2: Dump Truck
  3: Excavator
  4: Small Car
  5: Tractor
  6: Trailer
  7: Truck Tractor
  8: Van
  9: other-vehicle

# 类别数量
nc: 10
"""
    yaml_path = Path(output_dir) / 'data.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f"✅ 创建 data.yaml: {yaml_path}")
    return yaml_path


def main():
    parser = argparse.ArgumentParser(description='转换数据集为 YOLO OBB 格式')
    parser.add_argument('--fold', type=int, default=None, help='指定 fold (0-4)')
    parser.add_argument('--all', action='store_true', help='转换所有 fold')
    parser.add_argument('--output', type=str, default='dataset_yolo', help='输出根目录')
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent / 'dataset'
    output_root = Path(__file__).parent / args.output
    output_root.mkdir(parents=True, exist_ok=True)

    if args.all:
        folds = list(range(5))
    elif args.fold is not None:
        folds = [args.fold]
    else:
        folds = [0]  # 默认 fold 0

    for fold in folds:
        fold_dir = base_dir / f'fold_{fold}'
        if not fold_dir.exists():
            print(f"⚠️  Fold {fold} 不存在: {fold_dir}")
            continue

        train_json = fold_dir / 'train.json'
        val_json = fold_dir / 'val.json'

        fold_output = output_root / f'fold_{fold}'

        # 转换训练集
        print(f"\n{'='*60}")
        print(f"Fold {fold} - 训练集转换...")
        print(f"{'='*60}")
        train_dir = fold_output / 'train'
        n_train_img, n_train_ann = convert_json_to_yolo(str(train_json), str(train_dir))

        # 转换验证集
        print(f"\nFold {fold} - 验证集转换...")
        val_dir = fold_output / 'val'
        n_val_img, n_val_ann = convert_json_to_yolo(str(val_json), str(val_dir))

        # 创建 data.yaml
        create_data_yaml(str(fold_output), 'train/images', 'val/images')

        print(f"\n📊 Fold {fold} 统计:")
        print(f"  Train: {n_train_img} 图片, {n_train_ann} 标注")
        print(f"  Val:   {n_val_img} 图片, {n_val_ann} 标注")

    # 转换测试集
    test_json = base_dir / 'test.json'
    if test_json.exists():
        print(f"\n{'='*60}")
        print(f"测试集转换...")
        print(f"{'='*60}")
        test_dir = output_root / 'test'
        n_test_img, n_test_ann = convert_json_to_yolo(str(test_json), str(test_dir))
        print(f"  Test:  {n_test_img} 图片, {n_test_ann} 标注")


if __name__ == '__main__':
    main()
