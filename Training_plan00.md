# YOLO11-OBB 遥感车辆检测 — Training Plan v0.0

> 日期：2026-06-27 | 版本：v0.0（初始训练计划）

---

## 一、任务总览

| 项 | 内容 |
|---|------|
| **任务** | 遥感街景车辆 OBB 检测，比赛冲榜 |
| **目标** | 排行榜 mAP 最高（唯一指标） |
| **硬件** | 8×A100 40G（训练速度/显存不是瓶颈） |
| **框架** | Ultralytics 8.4.80（本地修改版，兼容 YOLO11/YOLO26） |

### 1.1 数据集

| 指标 | 数值 |
|------|------|
| 图片数 | 9,442 张（1000×1000 为主） |
| 标注数 | 468,979 |
| 类别数 | 10 类 |
| 标注类型 | OBB（4 点旋转框） |
| 核心难点 | 45% 目标 < 18px / 密集（最高 1074/图）/ 严重长尾（Small Car+Van 占 87%） |
| 划分 | 5 折交叉验证 + 10% 固定 test 集 |

### 1.2 预训练模型决策

**结论：预训练权重 Fine-tuning 优于从头训练。**

- YOLO11-OBB 的 `.pt` 权重包含 **COCO 预训练 backbone + DOTA 遥感 OBB 预训练 head**
- DOTA 和本数据集同为遥感旋转框检测，域匹配度高
- 例外：**P2 增强模型必须从头训练**（网络结构不同，多了 P2/4 检测层）

---

## 二、模型选型

### 2.1 可用预训练权重

| 版本 | 参数量 | GFLOPs | 预训练路径 | 推荐度 |
|------|--------|--------|-----------|--------|
| yolo11n-obb.pt | 2.7M | 6.9 | COCO + DOTA | ❌ 容量不足 |
| yolo11s-obb.pt | 9.7M | 22.7 | COCO + DOTA | ⚠️ 勉强 |
| yolo11m-obb.pt | 21.0M | 72.2 | COCO + DOTA | ✅ 推荐 |
| **yolo11l-obb.pt** | **26.2M** | **91.3** | **COCO + DOTA** | ✅✅ **首推** |
| yolo11x-obb.pt | 58.9M | 204.3 | COCO + DOTA | ⚠️ 冲榜冲刺 |
| yolo26s-obb.pt | 10.6M | 63.5 | COCO + DOTA | 🔬 架构对照 |
| yolo26m-obb.pt | 23.6M | 211.9 | COCO + DOTA | ❌ 计算量大/边际收益低 |

### 2.2 选型理由

**首推 `yolo11l-obb.pt`：**

- 26M 参数 + DOTA OBB 预训练，容量和先验知识兼顾
- 对 9K 图不会严重过拟合，对 10 类 + 密集小目标有足够拟合空间
- 比 m 只多 25% 参数，但密集场景收益远超这个比例
- YOLO26 同尺寸计算量大 3x，在 9K 图数据量下边际收益低，暂不主攻

---

## 三、训练实验矩阵

### Phase 1 — Baseline 建立（预计 1 天）

建立多个 Baseline，对比模型规模和预训练的效果。

| 实验 | 命令 | 预计 mAP50 | 时间 |
|------|------|-----------|------|
| **EXP-01a** Baseline-n | `--model yolo11n-obb.pt --epochs 200 --batch 128 --imgsz 640` | ~0.55 | ~1.5h |
| **EXP-01b** Baseline-m | `--model yolo11m-obb.pt --epochs 200 --batch 96 --imgsz 640` | ~0.57 | ~2h |
| **EXP-01c** Baseline-l | `--model yolo11l-obb.pt --epochs 200 --batch 64 --imgsz 640` | ~0.58 | ~2.5h |
| EXP-01d 从头训练对照 | `--model yolo11n-obb.yaml --epochs 300` | 待观察 | ~2h |

> **目标**：确认预训练收益、l vs m 的规模收益，确定后续实验的基准模型。

### Phase 2 — 核心优化叠加（预计 2-3 天）

以 Phase 1 最佳模型为基础，按收益递减顺序叠加优化。

| # | 实验 | 新增内容 | 关键参数 | 预计收益 |
|---|------|---------|---------|---------|
| EXP-02 | +Focal Loss | BCE→Focal | `--use-focal --focal-gamma 2.0` | +0.5~1.5 |
| EXP-03 | +cls_pw | 逆频率类别权重 | `--cls-pw 0.5` | +0.3~1.0 |
| EXP-04 | +Focal+cls_pw 组合 | 叠加 EXP-02+03 | `--use-focal --cls-pw 0.5` | +0.5~2.0 |
| EXP-05 | +大图输入 800 | 提升小目标分辨率 | `--imgsz 800` | +0.5~1.5 |
| EXP-06 | +大图输入 1024 | 进一步提分辨率 | `--imgsz 1024` | +0.3~1.0 |
| EXP-07 | +Cosine LR | 余弦学习率 | `--cos-lr --epochs 300` | +0.3~0.8 |
| EXP-08 | +Multi-Scale | 多尺度训练 | `--multi-scale 0.1` | +0.3~0.8 |

> 每步保留最佳配置，形成 `Phase2_best` 配置作为后续基准。

### Phase 3 — 数据增强优化（预计 1-2 天）

基于 Phase 2 最佳配置，逐个测试数据增强。

| # | 实验 | 新增内容 | 关键参数 | 预计收益 |
|---|------|---------|---------|---------|
| EXP-09 | +Mosaic9 | 3×3 Mosaic | `--mosaic9 0.2` | +0.3~0.8 |
| EXP-10 | +MixUp | 遥感适度 MixUp | `--mixup 0.1` | +0.2~0.5 |
| EXP-11 | +Rotation Aug | 遥感旋转增强 | `--rotation-augment 0.1` | ±0.1~0.5 |
| EXP-12 | +CopyPaste | 小目标复制粘贴 | `--small-obj-copy-paste 0.2` | +0.3~0.8 |

### Phase 4 — 高级 Loss 实验（预计 1-2 天）

基于当前最佳配置，测试备用 Loss 是否有额外收益。

| # | 实验 | 新增内容 | 预计收益 |
|---|------|---------|---------|
| EXP-13 | +Wise-IoU | 关注高质量匹配 | ±0.1~0.5 |
| EXP-14 | +KLD Angle | 正方形目标角度稳定 | ±0.1~0.5 |
| EXP-15 | +Slide Loss | 中等 IoU 样本关注 | ±0.1~0.5 |
| EXP-16 | +Scale-aware | 利用 scale_id 加权 | +0.3~0.8 |

### Phase 5 — P2 增强模型（预计 2-3 天）

P2 模型必须从头训练（结构不同），但可以用 partial loading 加载 backbone 预训练。

| # | 实验 | 模型 | 预计收益 |
|---|------|------|---------|
| EXP-17a | P2 从头训练 n | `yolo11n-obb-p2.yaml` (scratch) | vs EXP-01a |
| EXP-17b | P2 partial loading n | `yolo11n-obb-p2.yaml` + backbone 权重 | vs EXP-17a |
| EXP-18a | P2 从头训练 l | `yolo11l-obb-p2.yaml` (scratch) | vs EXP-01c |
| EXP-18b | P2 partial loading l | `yolo11l-obb-p2.yaml` + backbone 权重 | vs EXP-18a |
| EXP-19 | P2 + 全优化组合 | P2 + Phase2~4 最佳配置 | — |

### Phase 6 — 冲榜冲刺（预计 2-3 天）

| # | 实验 | 内容 | 预计收益 |
|---|------|------|---------|
| EXP-20 | yolo11x-obb.pt | 最大模型（如果未过拟合） | +0.5~1.5 |
| EXP-21 | 5-fold 完整训练 | 收集所有 fold 模型 | — |
| EXP-22 | TTA | 水平翻转 + 多尺度推理 | +0.5~1.0 |
| EXP-23 | WBF Ensemble | 5-fold 加权框融合 | +0.5~2.0 |

---

## 四、推荐执行顺序（按优先级）

```
Phase 1 ──→ Phase 2 ──→ Phase 3 ──→ Phase 4 ──→ Phase 5 ──→ Phase 6
  │            │           │           │           │           │
  │            │           │           │           │           └── WBF + TTA
  │            │           │           │           └── P2 从头训练 + Partial Loading
  │            │           │           └── 高级 Loss（不确定收益，先不做）
  │            │           └── 数据增强（Mosaic9/MixUp/CopyPaste）
  │            └── 核心优化（Focal/cls_pw/大图/Cosine/MultiScale）
  └── 建立 Baseline（确认预训练收益 + 模型规模选择）
```

**原则**：
- 每步只改一个变量，方便 Ablation 归因
- 发现负收益立即回退，不累积
- Phase 4 如果前三项无收益则整阶段跳过
- Phase 6 仅在确定不 overfit 时进行

---

## 五、快速启动命令

```bash
PROJECT=/data/work1/panliang/2026/00_TZB_game/YOLOv11
cd $PROJECT

# Phase 1 — 建立 Baseline（第一个要跑的）
python train.py --fold 0 --model yolo11l-obb.pt --epochs 200 --batch 64 --imgsz 640 --name exp01c_baseline_l

# 同时并行一个对照（不同 GPU 组）
python train.py --fold 0 --model yolo11m-obb.pt --epochs 200 --batch 96 --imgsz 640 --device 4,5,6,7 --name exp01b_baseline_m

# Phase 2 — 核心优化（Baseline 确定后）
python train.py --fold 0 --model yolo11l-obb.pt --epochs 200 --batch 64 --imgsz 640 --use-focal --focal-gamma 2.0 --name exp02_focal
python train.py --fold 0 --model yolo11l-obb.pt --epochs 200 --batch 64 --imgsz 640 --use-focal --cls-pw 0.5 --name exp04_focal_clspw
python train.py --fold 0 --model yolo11l-obb.pt --epochs 300 --batch 48 --imgsz 800 --use-focal --cls-pw 0.5 --cos-lr --name exp07_large_cos

# Phase 5 — P2 模型（独立分支）
python train.py --fold 0 --model yolo11n-obb-p2.yaml --epochs 300 --batch 64 --imgsz 640 --name exp17a_p2_scratch
python train.py --fold 0 --model yolo11l-obb-p2.yaml --epochs 300 --batch 32 --imgsz 640 --name exp18a_p2_l_scratch
```

---

## 六、结果追踪表

| 实验 | 模型 | mAP50 | mAP50-95 | Small Car AP | Van AP | Tail mAP | 备注 |
|------|------|-------|----------|-------------|--------|----------|------|
| **Phase 1** | | | | | | | |
| EXP-01c l-baseline | yolo11l-obb.pt | | | | | | |
| EXP-01b m-baseline | yolo11m-obb.pt | | | | | | |
| EXP-01d scratch | yolo11n-obb.yaml | | | | | | 从头训练对照 |
| **Phase 2** | | | | | | | |
| EXP-02 +Focal | | | | | | | |
| EXP-03 +cls_pw | | | | | | | |
| EXP-04 +Focal+cls_pw | | | | | | | |
| EXP-05 +imgsz800 | | | | | | | |
| EXP-06 +imgsz1024 | | | | | | | |
| EXP-07 +CosLR | | | | | | | |
| EXP-08 +MultiScale | | | | | | | |
| **Phase 3** | | | | | | | |
| EXP-09 +Mosaic9 | | | | | | | |
| EXP-10 +MixUp | | | | | | | |
| EXP-11 +Rotation | | | | | | | |
| EXP-12 +CopyPaste | | | | | | | |
| **Phase 4** | | | | | | | |
| EXP-13 +Wise-IoU | | | | | | | |
| EXP-14 +KLD | | | | | | | |
| EXP-15 +Slide Loss | | | | | | | |
| EXP-16 +Scale-aware | | | | | | | |
| **Phase 5** | | | | | | | |
| EXP-17a P2-n scratch | | | | | | | |
| EXP-17b P2-n partial | | | | | | | |
| EXP-18a P2-l scratch | | | | | | | |
| EXP-18b P2-l partial | | | | | | | |
| EXP-19 P2 全优化 | | | | | | | |
| **Phase 6** | | | | | | | |
| EXP-20 x-obb | yolo11x-obb.pt | | | | | | |
| EXP-22 TTA | — | | | | | | 推理时 |
| EXP-23 WBF | — | | | | | | 推理时 |

---

## 七、风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| l 模型 9K 图过拟合 | 低 | 中 | 监控 val mAP 曲线，开启 Strong Aug |
| x 模型严重过拟合 | 中 | 中 | 先看 l 的 val 曲线再决定，加重正则化 |
| P2 从头训练收敛慢 | 中 | 中 | 用 partial loading 加载 backbone，epoch 拉到 500 |
| Focal Loss 不稳定 | 低 | 低 | gamma 从 1.0 逐步增加 |
| KLD 与 ProbIoU 冲突 | 中 | 低 | 关闭回退 |
| 数据增强破坏 OBB | 中 | 中 | Rotation 在 OBB 场景要谨慎，监控角度 loss |
| 不同 fold 结果差异大 | 中 | 低 | 至少跑 2 个 fold 确认稳定性 |

---

## 八、日常流程

```bash
# 1. 启动训练
python train.py --fold 0 --model yolo11l-obb.pt --epochs 200 --batch 64 --imgsz 640 --name exp_xxx

# 2. 监控训练（另一个终端）
tensorboard --logdir runs/obb/train/

# 3. 完成后查看结果
cat runs/obb/train/exp_xxx/results.csv | tail -1

# 4. 记录到结果追踪表
# 更新本文件的「六、结果追踪表」
```

---

## 九、预期总收益路线

```
                   yolo11n-obb.pt (DOTA预训练)
                   mAP50 ≈ 0.55-0.60
                          │
          ┌───────────────┼───────────────┐
          ▼                                ▼
   yolo11l-obb.pt                    yolo11l-obb-p2.yaml
   + Focal + cls_pw                 (Partial Loading)
   + imgsz 1024 + CosLR             + Focal + cls_pw
   + Mosaic9 + MixUp                + imgsz 1024
   + Scale-aware                    + 从头训练 500 epoch
   mAP50 ≈ 0.62-0.67               mAP50 ≈ 0.60-0.68
          │                                │
          └───────────────┬───────────────┘
                          ▼
                  5-fold × TTA × WBF
                  mAP50 ≈ 0.65-0.72  ← 冲榜最终分数
```

---

*本计划为初始版本，每个 Phase 完成后根据实际结果调整后续策略。*
