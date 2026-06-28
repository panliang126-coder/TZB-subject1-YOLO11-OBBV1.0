# YOLO11-OBB 遥感车辆检测增强版 — Ablation 实验计划

> 版本: v1.0 | 日期: 2026-06-27
> 数据集: 遥感街景车辆检测 (9442张, 468979标注, 10类)
> 硬件: 8×A100 40G

---

## 一、实验设计原则

1. **一次只改一个变量** — 每个实验单独开启/关闭一个优化
2. **保留 Baseline 对照** — 每个实验都有对应的 Baseline 结果
3. **配置开关控制** — 全部通过 `--use-xxx` / `--no-xxx` 控制
4. **固定随机种子** — 所有实验使用 `seed=42, deterministic=True`

---

## 二、Baseline 实验

### EXP-00: Baseline (YOLO11n-OBB 标准配置)

```bash
python train.py --fold 0 --baseline --epochs 200 --batch 64 --imgsz 640
```

**配置要点:**
- 模型: `yolo11n-obb.yaml` (P3/P4/P5, stride [8,16,32])
- Loss: BCE + ProbIoU + sin(2θ)² angle loss
- 增强: Mosaic(1.0) + HSV + Flip
- cos_lr: False (线性衰减)

**目标**: 建立 mAP 基线, 预期 mAP50 ~0.55-0.60

---

## 三、优化实验矩阵

### 优先级 ★★★★★ (预计 +2~5 mAP)

#### EXP-01: P2 检测层
```bash
python train.py --fold 0 --model yolo11n-obb-p2.yaml --epochs 200 --batch 64 --imgsz 640
```
| 配置 | 值 |
|------|-----|
| 模型 | yolo11n-obb-p2.yaml |
| 检测层 | P2/4, P3/8, P4/16, P5/32 |
| 为什么有效 | ~45% 目标 < 18px, P2 stride=4 保留更多空间信息 |
| 预计提升 | **+1.5~3.0 mAP** |

#### EXP-02: Focal Loss 替代 BCE
```bash
python train.py --fold 0 --model yolo11n-obb-p2.yaml --use-focal --focal-gamma 2.0 --epochs 200 --batch 64 --imgsz 640
```
| 配置 | 值 |
|------|-----|
| 为什么有效 | Small Car+Van 占 87%, Focal Loss 抑制简单样本,关注难样本 |
| 预计提升 | **+0.5~1.5 mAP** (长尾类别收益更大) |

#### EXP-03: Class Weight (cls_pw)
```bash
python train.py --fold 0 --model yolo11n-obb-p2.yaml --cls-pw 0.5 --epochs 200 --batch 64 --imgsz 640
```
| 配置 | 值 |
|------|-----|
| cls_pw | 0.5 (逆频率权重, 提升尾类损失) |
| 为什么有效 | Tractor 仅 288 样本, 需要更高损失权重 |
| 预计提升 | **+0.3~1.0 mAP** (尾类 mAP 提升明显) |

#### EXP-04: P2 + Focal + cls_pw (组合测试)
```bash
python train.py --fold 0 --model yolo11n-obb-p2.yaml --use-focal --focal-gamma 2.0 --cls-pw 0.5 --epochs 200 --batch 64 --imgsz 640
```
| 预计提升 | **+2.0~4.5 mAP** (叠加收益) |

---

### 优先级 ★★★★☆ (预计 +0.5~2 mAP)

#### EXP-05: Larger Image Size
```bash
python train.py --fold 0 --model yolo11n-obb-p2.yaml --use-focal --imgsz 800 --epochs 200 --batch 48
python train.py --fold 0 --model yolo11n-obb-p2.yaml --use-focal --imgsz 1024 --epochs 200 --batch 32
```
| 配置 | 值 |
|------|-----|
| imgsz=800 | 15px 目标→~19px in P2, 更好分辨 |
| imgsz=1024 | 15px 目标→~24px in P2, 显著提升 |
| 预计提升 | **+0.5~2.0 mAP** |

#### EXP-06: Multi-Scale Training
```bash
python train.py --fold 0 --model yolo11n-obb-p2.yaml --use-focal --multi-scale 0.1 --imgsz 800 --epochs 200
```
| 配置 | 值 |
|------|-----|
| multi_scale | 0.1 (±10% 尺度变化) |
| 为什么有效 | 遥感目标尺度变化大 (2.85~266px) |
| 预计提升 | **+0.3~0.8 mAP** |

#### EXP-07: 3×3 Mosaic (Mosaic9)
```bash
python train.py --fold 0 --model yolo11n-obb-p2.yaml --use-focal --mosaic9 0.2 --epochs 200 --batch 64 --imgsz 640
```
| 配置 | 值 |
|------|-----|
| mosaic9 | 0.2 (20% 使用 3×3 布局, 9 图拼接) |
| 为什么有效 | 单图最高 1074 目标, 3×3 拼接增加密集场景多样性 |
| 预计提升 | **+0.3~0.8 mAP** |

#### EXP-08: MixUp Augmentation
```bash
python train.py --fold 0 --model yolo11n-obb-p2.yaml --use-focal --mixup 0.1 --epochs 200 --batch 64 --imgsz 640
```
| 配置 | 值 |
|------|-----|
| mixup | 0.1 (遥感场景适度 MixUp) |
| 为什么有效 | 增加类间混合, 改善尾类泛化 |
| 预计提升 | **+0.2~0.5 mAP** |

#### EXP-09: Cosine LR Schedule
```bash
python train.py --fold 0 --model yolo11n-obb-p2.yaml --use-focal --cos-lr --epochs 300 --batch 64 --imgsz 640
```
| 配置 | 值 |
|------|-----|
| cos_lr | True |
| epochs | 300 (Cosine 需要更多 epoch) |
| 预计提升 | **+0.3~0.8 mAP** |

---

### 优先级 ★★★☆☆ (预计 +0.1~0.5 mAP, 实验性)

#### EXP-10: KLD Angle Loss
```bash
python train.py --fold 0 --model yolo11n-obb-p2.yaml --use-focal --use-kld-angle --epochs 200 --batch 64 --imgsz 640
```
| 配置 | 值 |
|------|-----|
| use_kld_angle | True |
| 为什么有效 | KLD 对接近正方形的目标更稳定 |
| 注意 | 可能与现有 sin² 损失接近, 收益不确定 |

#### EXP-11: Wise-IoU
```bash
python train.py --fold 0 --model yolo11n-obb-p2.yaml --use-focal --use-wise-iou --epochs 200 --batch 64 --imgsz 640
```
| 配置 | 值 |
|------|-----|
| use_wise_iou | True |
| 为什么有效 | 降低低质量预测的梯度, 让模型关注高质量匹配 |
| 注意 | 需要仔细调参 wise_iou_beta |

#### EXP-12: Slide Loss
```bash
python train.py --fold 0 --model yolo11n-obb-p2.yaml --use-focal --use-slide-loss --epochs 200 --batch 64 --imgsz 640
```
| 配置 | 值 |
|------|-----|
| use_slide_loss | True |
| slide_mu | 0.5 |
| 为什么有效 | 关注中等 IoU 样本, 减少过难样本的噪声 |

---

## 四、推荐实验顺序 (按收益递减)

| 优先级 | 实验 | 命令 | 预计提升 | 预计时间 |
|--------|------|------|----------|----------|
| 1 | EXP-00 Baseline | `--baseline` | - | 2h |
| 2 | EXP-01 P2 | `--model yolo11n-obb-p2.yaml` | +1.5~3.0 | 3h |
| 3 | EXP-02 P2+Focal | `--model yolo11n-obb-p2.yaml --use-focal` | +0.5~1.5 | 3h |
| 4 | EXP-03 P2+cls_pw | `--model yolo11n-obb-p2.yaml --cls-pw 0.5` | +0.3~1.0 | 3h |
| 5 | EXP-04 P2+Focal+cls_pw | 组合 | +2.0~4.5 | 3h |
| 6 | EXP-05 ImgSz 800 | `--imgsz 800` | +0.5~2.0 | 4h |
| 7 | EXP-06 MultiScale | `--multi-scale 0.1` | +0.3~0.8 | 3h |
| 8 | EXP-07 Mosaic9 | `--mosaic9 0.2` | +0.3~0.8 | 3h |
| 9 | EXP-08 MixUp | `--mixup 0.1` | +0.2~0.5 | 3h |
| 10 | EXP-09 CosLR 300ep | `--cos-lr --epochs 300` | +0.3~0.8 | 4.5h |
| 11 | EXP-10 KLD | `--use-kld-angle` | ±0.1~0.5 | 3h |
| 12 | EXP-11 Wise-IoU | `--use-wise-iou` | ±0.1~0.5 | 3h |
| 13 | EXP-12 Slide Loss | `--use-slide-loss` | ±0.1~0.5 | 3h |

---

## 五、自动化 Ablation 脚本

```bash
#!/bin/bash
# 一键运行所有 Ablation 实验

PROJECT=/data/work1/panliang/2026/00_TZB_game/YOLOv11

# 实验列表: "名称|额外参数"
experiments=(
    "exp00_baseline|--baseline"
    "exp01_p2|--model yolo11n-obb-p2.yaml"
    "exp02_p2_focal|--model yolo11n-obb-p2.yaml --use-focal"
    "exp03_p2_clspw|--model yolo11n-obb-p2.yaml --cls-pw 0.5"
    "exp04_p2_focal_clspw|--model yolo11n-obb-p2.yaml --use-focal --cls-pw 0.5"
    "exp05_imgsz800|--model yolo11n-obb-p2.yaml --use-focal --imgsz 800 --batch 48"
    "exp06_multiscale|--model yolo11n-obb-p2.yaml --use-focal --multi-scale 0.1"
    "exp07_mosaic9|--model yolo11n-obb-p2.yaml --use-focal --mosaic9 0.2"
    "exp08_mixup|--model yolo11n-obb-p2.yaml --use-focal --mixup 0.1"
    "exp09_coslr|--model yolo11n-obb-p2.yaml --use-focal --cos-lr --epochs 300"
    "exp10_kld|--model yolo11n-obb-p2.yaml --use-focal --use-kld-angle"
)

for exp in "${experiments[@]}"; do
    name="${exp%%|*}"
    extra_args="${exp##*|}"
    echo "=== Running $name ==="
    python $PROJECT/train.py --fold 0 $extra_args --name "$name"
done
```

---

## 六、结果追踪表

| 实验 | mAP50 | mAP50-95 | Small Car AP | Van AP | Tail AP | 备注 |
|------|-------|----------|-------------|--------|---------|------|
| EXP-00 Baseline | | | | | | 基线 |
| EXP-01 P2 | | | | | | |
| EXP-02 P2+Focal | | | | | | |
| EXP-03 P2+cls_pw | | | | | | |
| EXP-04 P2+Focal+cls_pw | | | | | | |
| EXP-05 ImgSz 800 | | | | | | |
| EXP-06 MultiScale | | | | | | |
| EXP-07 Mosaic9 | | | | | | |
| EXP-08 MixUp | | | | | | |
| EXP-09 CosLR 300ep | | | | | | |
| EXP-10 KLD | | | | | | |
| EXP-11 Wise-IoU | | | | | | |
| EXP-12 Slide Loss | | | | | | |

---

## 七、风险分析

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| P2 层导致显存溢出 | 中 | 中 | 减小 batch size 或使用 gradient accumulation |
| Focal Loss 训练不稳定 | 低 | 低 | focal_gamma 从 1.0 开始逐步增加 |
| KLD Loss 与 ProbIoU 冲突 | 中 | 低 | 关闭 KLD 回退到 sin² loss |
| 数据增强过拟合 | 低 | 中 | 监控 val loss / mAP 曲线 |
| 小目标被 P2 层过拟合 | 低 | 中 | 验证大目标精度是否下降 |

---

## 八、最终推荐配置 (冲榜用)

```bash
python train.py \
    --fold 0 \
    --model yolo11x-obb-p2.yaml \
    --use-focal --focal-gamma 2.0 --cls-pw 0.5 \
    --cos-lr --epochs 500 \
    --imgsz 1024 --batch 32 \
    --multi-scale 0.1 --mosaic9 0.2 --mixup 0.1 \
    --close-mosaic 20 \
    --name final_submission \
    --device 0,1,2,3,4,5,6,7
```

**预计总提升: +3~8 mAP** (相比 Baseline)

**推理时额外提升:**
- TTA: 水平翻转 + 多尺度 (0.8×, 1.0×, 1.2×)
- WBF: 5-fold 模型加权融合
- 预计再提升 +1~3 mAP
