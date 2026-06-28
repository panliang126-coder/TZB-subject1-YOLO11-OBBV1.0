# YOLO11-OBB 遥感车辆检测增强版

> 基于 Ultralytics YOLO11-OBB 框架，针对遥感街景车辆检测任务进行全面优化。
> 目标: 比赛级 SOTA 精度 (mAP 最高)

## 快速开始

```bash
# 1. 转换数据集 (只需首次运行)
python convert_to_yolo.py --all

# 2. Baseline 训练 (标准 YOLO11-OBB)
python train.py --fold 0 --baseline --epochs 200

# 3. 增强版训练 (P2 + Focal + cls_pw + 增强数据)
python train.py --fold 0 --enhanced --epochs 300

# 4. Ablation: 单独测试某个优化
python train.py --fold 0 --model yolo11n-obb-p2.yaml --use-focal
```

## 项目结构

```
YOLOv11/
├── ultralytics_src/          # ultralytics 源码 (已修改)
│   └── ultralytics/
│       ├── cfg/
│       │   ├── default.yaml              # ★ 新增增强超参数
│       │   ├── models/11/
│       │   │   └── yolo11-obb-p2.yaml    # ★ P2 检测层模型
│       │   └── __init__.py               # ★ 新增参数验证
│       ├── utils/
│       │   ├── loss.py                   # ★ v8OBBLoss 增强
│       │   └── patches.py                # ★ 4通道TIFF修复
│       └── data/augment.py               # (增强待实施)
├── dataset/                  # 原始数据集 (JSON格式)
├── dataset_yolo/             # YOLO格式数据集 (自动生成)
│   └── fold_0~4/
│       ├── data.yaml
│       ├── train/{images,labels}/
│       └── val/{images,labels}/
├── configs/
│   ├── baseline.yaml         # Baseline 配置
│   └── enhanced.yaml         # 增强版配置
├── train.py                  # 训练脚本
├── convert_to_yolo.py        # 数据转换脚本
└── ABLATION_PLAN.md          # Ablation 实验计划
```

## 已实施的优化清单

### ★★★★★ 已完成 (最高优先级)

| # | 优化点 | 修改文件 | 配置开关 | 预计提升 |
|---|--------|----------|----------|----------|
| 1 | **P2 检测层** | `cfg/models/11/yolo11-obb-p2.yaml` | `--model yolo11n-obb-p2.yaml` | +1.5~3.0 mAP |
| 2 | **Focal Loss** | `utils/loss.py` L1008-1028 | `--use-focal` | +0.5~1.5 mAP |
| 3 | **Class Weight (cls_pw)** | `utils/loss.py` | `--cls-pw 0.5` | +0.3~1.0 mAP |
| 4 | **4通道TIFF支持** | `utils/patches.py` L43-49 | 自动 | 基础功能 |

### ★★★★☆ 已完成 (高优先级)

| # | 优化点 | 修改文件 | 配置开关 | 预计提升 |
|---|--------|----------|----------|----------|
| 5 | **KLD Angle Loss** | `utils/loss.py` L1130-1180 | `--use-kld-angle` | ±0.1~0.5 mAP |
| 6 | **Wise-IoU** | `utils/loss.py` L1182-1215 | `--use-wise-iou` | ±0.1~0.5 mAP |
| 7 | **Slide Loss** | `utils/loss.py` (infra ready) | `--use-slide-loss` | ±0.1~0.5 mAP |
| 8 | **Scale-aware Weight** | `utils/loss.py` L1217-1240 | `--use-scale-aware` | +0.3~0.8 mAP |
| 9 | **Cosine LR** | `default.yaml` | `--cos-lr` | +0.3~0.8 mAP |
| 10 | **Multi-Scale Train** | `default.yaml` | `--multi-scale 0.1` | +0.3~0.8 mAP |

### 配置参数完整列表

```yaml
# 增强 OBB 超参数 (在 default.yaml 中已注册)
use_focal: false          # Focal Loss 替代 BCE
focal_gamma: 2.0          # Focal gamma
focal_alpha: 0.25         # Focal alpha
use_wise_iou: false       # Wise-IoU
wise_iou_beta: 6.0        # Wise-IoU beta
use_kld_angle: false      # KLD Angle Loss
kld_tau: 1.0              # KLD temperature
use_slide_loss: false     # Slide Loss
slide_mu: 0.5             # Slide Loss center
use_scale_aware: false    # Scale-aware Weight
scale_aware_gamma: 1.0    # Scale-aware gamma
use_dynamic_topk: false   # Dynamic TopK
dynamic_topk_min: 5       # TopK min
dynamic_topk_max: 20      # TopK max
scale_aware_mosaic: 0.0   # Scale-aware Mosaic
small_obj_copy_paste: 0.0 # Small Object CopyPaste
rotation_augment: 0.0     # Rotation augmentation
rotation_deg: 15.0        # Max rotation degree
mosaic9: 0.0              # 3x3 Mosaic
```

## 推荐实验顺序

按收益递减执行:

```bash
# 1. Baseline - 建立基线
python train.py --fold 0 --baseline

# 2. P2 检测层 ★★★★★
python train.py --fold 0 --model yolo11n-obb-p2.yaml

# 3. P2 + Focal Loss ★★★★★
python train.py --fold 0 --model yolo11n-obb-p2.yaml --use-focal

# 4. P2 + Focal + cls_pw ★★★★★
python train.py --fold 0 --model yolo11n-obb-p2.yaml --use-focal --cls-pw 0.5

# 5. 增大输入尺寸 ★★★★☆
python train.py --fold 0 --model yolo11n-obb-p2.yaml --use-focal --imgsz 800

# 6. + Cosine LR ★★★★☆
python train.py --fold 0 --model yolo11n-obb-p2.yaml --use-focal --imgsz 800 --cos-lr --epochs 300

# 7. + 数据增强 ★★★★☆
python train.py --fold 0 --enhanced
```

## 完整冲榜配置

```bash
python train.py \
    --fold 0 \
    --model yolo11x-obb-p2.yaml \
    --use-focal --focal-gamma 2.0 --cls-pw 0.5 \
    --cos-lr --epochs 500 \
    --imgsz 1024 --batch 32 \
    --multi-scale 0.1 --mosaic9 0.2 --mixup 0.1 \
    --close-mosaic 20 \
    --device 0,1,2,3,4,5,6,7 \
    --name final_submission
```

## 开发原则

1. ✅ 一次 Commit 只实现一个优化点
2. ✅ 每个优化点通过配置项独立开关
3. ✅ 保留 Baseline 对照实验能力
4. ✅ 每个优化点有独立 Ablation 计划
5. ✅ 不允许多个优化点耦合
6. ✅ 所有新增模块兼容 YOLO11-OBB
7. ✅ 按收益最高优先实施
