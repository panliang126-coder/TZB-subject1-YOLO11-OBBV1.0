# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 常用命令

```bash
# 分析单个实验的最新结果
python3 -c "
import pandas as pd
df = pd.read_csv('runs/<exp_name>/results.csv')
df = df.dropna(subset=['metrics/mAP50(B)'])
last = df.iloc[-1]; best = df.loc[df['metrics/mAP50(B)'].idxmax()]
print(f'Epoch {int(last[\"epoch\"])}/{len(df)}  mAP50={last[\"metrics/mAP50(B)\"]:.4f}  best={best[\"metrics/mAP50(B)\"]:.4f} @ ep{int(best[\"epoch\"])}')
"

# 横向对比多个实验
python3 -c "
import pandas as pd
from pathlib import Path
for d in sorted(Path('runs').iterdir()):
    f = d / 'results.csv'
    if f.exists():
        df = pd.read_csv(f).dropna(subset=['metrics/mAP50(B)'])
        if len(df) > 10:
            best = df.loc[df['metrics/mAP50(B)'].idxmax()]
            print(f'{d.name:<35s} ep={int(best[\"epoch\"]):3d}/{len(df)}  mAP50={best[\"metrics/mAP50(B)\"]:.4f}  mAP50-95={best[\"metrics/mAP50-95(B)\"]:.4f}')
"

# 实时监控 training 进度
watch -n 10 'tail -5 /tmp/train_enhanced_yolo11n.log'
# DDP 模式下，子进程输出不走 tee，直接看 results.csv：
watch -n 30 'python3 -c "import pandas as pd; df=pd.read_csv(\"runs/EXP_NAME/results.csv\").dropna(subset=[\"metrics/mAP50(B)\"]); print(f\"epoch {int(df.iloc[-1][\\\"epoch\\\"])} mAP50={df.iloc[-1][\\\"metrics/mAP50(B)\\\"]:.4f}\")"'

# 查看 GPU 分配情况
nvidia-smi --query-compute-apps=pid,gpu_name,used_memory --format=csv
```

## 目录结构

```
runs/
  <exp_name>/
    args.yaml      # 训练超参数（完整配置快照）
    results.csv    # epoch 级训练日志（loss, mAP50, mAP50-95, precision, recall, lr）
    weights/
      best.pt      # 最佳 checkpoint（基于 val mAP50）
      last.pt      # 最终 epoch checkpoint
      .backup/     # 部分实验有备份
```

## 关键实验结果速查

### 当前最佳

| 实验 | Epochs | mAP50 | mAP50-95 | 配置特征 |
|------|--------|-------|----------|---------|
| `exp2_high_lr_v4` | 413 | **0.4967** | — | P2 + high_lr |
| `enhanced_fold0_1000ep-10` | 1000 | **0.4922** | — | P2 + enhanced |
| `exp5_finetune` | 79 | 0.4742 | — | 微调 |
| `enhanced_fold0-2` | 500 | 0.4493 | — | P2 + enhanced |

### Pretrain 对比 (TODO01 核心)

| 实验 | 初始化 | 配置 | mAP50 | 状态 |
|------|--------|------|-------|------|
| `baseline_fold0_1000ep` | Scratch | baseline | 0.2900 | 199/1000ep |
| `pretrain_yolo11n_fold0` | yolo11n.pt | **baseline** | 0.3801 | 264/300ep ✅ |
| `pretrain_enhanced_yolo11n_fold0` | yolo11n.pt | **enhanced** | — | 启动失败 |

### 结论

- **Pretrain > Scratch**: 同 baseline 配置下，pretrain (0.38) 显著优于 scratch (0.29)
- **baseline 配置太弱**: 缺 P2、imgsz=640、无 cos_lr，上限远低于 enhanced (0.49+)
- **Enhanced + Pretrain 未跑成**: 需要排查 P2 架构下 detect→OBB 权重加载问题

## 实验命名规范

| 前缀 | 含义 | 配置来源 |
|------|------|---------|
| `baseline_` | 标准 yolo11n-obb.yaml | configs/baseline.yaml |
| `enhanced_` | yolo11n-obb-p2.yaml | configs/enhanced.yaml |
| `exp1_focal_` | Focal Loss ablation | 自定义 |
| `exp2_high_lr_` | 高学习率 ablation | 自定义 |
| `exp3_focal_high_lr_` | Focal + 高 LR 组合 | 自定义 |
| `exp4_1024` | imgsz=1024 | 自定义 |
| `exp5_finetune` | 微调实验 | 自定义 |
| `pretrain_yolo11n_` | 官方预训练 OBB | baseline |
| `pretrain_enhanced_` | 官方预训练 OBB | enhanced |
| `test_` | 调试/验证 | 测试用 |

## TODO01 预训练实验计划

目标：验证官方 COCO 预训练权重对遥感 OBB 数据集的价值，选出最优模型容量。

**约束**：只允许改变预训练权重和模型容量，其余参数（架构、Loss、增强、策略）全部保持一致。

| Exp | 模型 | 预训练权重 | 参数量 | 状态 |
|-----|------|-----------|--------|------|
| Exp1 | yolo11n-obb-p2 | yolo11n.pt | 2.8M | ⏳ 待重启 |
| Exp2 | yolo11s-obb-p2 | yolo11s.pt | 9.8M | ⏳ |
| Exp3 | yolo11m-obb-p2 | yolo11m.pt | 21.0M | ⏳ |

训练脚本：`train_pretrain_enhanced.py --model n/s/m`

## 重要注意事项

### DDP 日志不可见
DDP 模式下子进程 stdout 不被 tee 捕获。必须通过 `results.csv` 或 `runs/<exp>/` 目录判断训练是否在正常运行。

### GPU 分配
- **GPU 0-3**：当前 pretrain 实验使用（前面 4 张卡）
- **GPU 4-7**：用户其他任务使用，**严禁动**
- imgsz=800 + batch=96 是 P2 架构的安全上限（CLAUDE.md 记录）

### 跨任务权重迁移
detect → OBB 通过 `model.load(ckpt)` + `intersect_dicts()` 实现：
- backbone 层：key 匹配，权重加载 ✅
- OBB head 层：key 不匹配，随机初始化 ✅
- P2 额外层：detect 模型没有，随机初始化 ✅

P2 模型权重迁移率约 45%（297/649），低于标准 OBB 的 83%（448/541），因为 P2 新增了更多 head 层。

### 实验结果分析模板
`TODO00_实验结果分析.md` 包含 12 维度的完整分析框架：分尺度表现、小目标检测、长尾类别、混淆矩阵等。
