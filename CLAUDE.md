# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 常用命令

```bash
# 训练
python train.py --enhanced --fold 0                          # enhanced 配置
python train.py --baseline --fold 0                          # baseline 配置
python train.py --cfg configs/exp_mosaic_base.yaml --fold 0  # 自定义 YAML 配置
python train.py --fold 0 --model yolo11n-obb-p2.yaml --use-focal  # ablation

# CLI 覆盖常用参数
python train.py --cfg configs/exp_mosaic_base.yaml --fold 0 --name my_exp \
    --close-mosaic 30 --device 0,1,2,3 --batch 48

# 预训练权重微调
python train_pretrain.py            # 官方 yolo11n.pt → OBB (4×A100, baseline.yaml)
python train_pretrain_enhanced.py   # 官方 yolo11n.pt → OBB (enhanced 配置)

# 断点续训（必须用完整目录名）
python train.py --enhanced --fold 0 --epochs 1000 --name enhanced_fold0_1000ep-10 --resume

# Profiling
python tools/benchmark_dataloader.py --config enhanced --full

# 训练后手动 test 评估
python -c "from ultralytics import YOLO; m=YOLO('runs/xxx/weights/best.pt'); m.val(data='dataset_yolo/fold_0/data.yaml', split='test')"

# Checkpoint 评估 (val+test, per-class, Head/Mid/Tail 分组)
python runs/exp6_test50/eval_checkpoint.py --weights runs/xxx/weights/best.pt --label epXX

# Mosaic 策略对比实验 (全自动串行)
bash run_mosaic_experiments.sh

# 生成对比报告
python tools/benchmark_mosaic.py

# 监控
python plot_metrics.py runs/enhanced_fold0_1000ep-10/results.csv --watch 120
watch -n2 nvidia-smi
```

## 架构要点

完整架构见 `CLAUDE.system.md`。这里只记关键关系：

- `train.py` → `ultralytics.YOLO` → `task_map["obb"]` → `OBBTrainer/OBBValidator/OBBPredictor`
- DDP 通过 `generate_ddp_file()` 序列化 trainer 到临时 `.py`，再 `torch.distributed.run` 启动 8 个子进程
- `BaseTrainer._do_train()` 是核心训练循环（`engine/trainer.py:390`）
- `v8OBBLoss` 是 OBB Loss 入口（`utils/loss.py:986`），含 ProbIoU + BCE/Focal + DFL + Angle
- `RotatedTaskAlignedAssigner` 用点积测试判断锚点是否在旋转框内（`utils/tal.py:359`）

## 重要注意事项

### DDP 内禁止调用 model.val()
训练中调用 `YOLO().val()` 会尝试再次启动 DDP 子进程，与已在运行的训练进程 GPU 冲突导致死锁。test 评估回调已因此移除，改为训练结束后手动评估。

### cache=ram 禁止用于 DDP
DDP fork 触发 copy-on-write，8 进程 × 10GB+ 图像缓存 = 系统 OOM。

### cache=disk 在 128 workers 下极慢
8495 个 `.npy` 小文件随机读取导致磁盘 I/O 饱和，每 epoch 10-18 分钟。使用默认 `cache=false`（JPEG 实时解码）即可。

### batch 上限
enhanced 配置（P2+800 分辨率）下 batch=96 是安全上限。更大的 batch 会导致 TaskAlignedAssigner CUDA OOM（319K 锚点 × 目标数矩阵过大）。

### yolo11n 在 A100 上吃不饱
2.6M 参数在 A100 上几毫秒算完一个 batch，nvidia-smi 显示 100% 但功耗仅 ~60W（实际负载 ~5%）。不影响训练正确性，~40s/epoch 是正常速度。

### 数据集
5-fold 交叉验证，每个 fold 的 data.yaml 共享同一个 test 集（`../test/images`）。

## 项目对 ultralytics 的修改

| 文件 | 变更 |
|------|------|
| `train.py` | **训练入口**: 支持 `--cfg`/`--close-mosaic`/`--device`/`--batch` CLI 覆盖; 断点续训保护 |
| `utils/loss.py` | `v8OBBLoss` 增强：Focal/WiseIoU/KLD/SlideLoss/ScaleAware 开关 |
| `utils/tal.py` | `RotatedTaskAlignedAssigner`：点积测试判断锚点在旋转框内 |
| `utils/callbacks/base.py` | 移除训练内 test 回调（DDP 死锁），改为 no-op |
| `data/build.py` | `persistent_workers=True` + worker 数量公式优化 |
| `utils/profile_loader.py` | **新增** DataLoader Profiling 框架 |
| `engine/trainer.py` | Profiling 集成（profile_loader 开关） |
| `cfg/default.yaml` | 新增增强超参数注册 |
| `cfg/models/11/yolo11-obb-p2.yaml` | **新增** P2 检测层模型定义 |
| `utils/loss.py` | **Bug修复**: FocalLoss标量→per-element; WiseIoU rbox2dist参数修复 |
| `models/utils/loss.py` | 适配FocalLoss per-element返回值 |
| `tools/benchmark_mosaic.py` | **新增** Mosaic 策略对比分析和报告生成 |
| `run_mosaic_experiments.sh` | **新增** 串行执行多组 Mosaic 实验的自动化脚本 |
| `runs/exp6_test50/eval_checkpoint.py` | **新增** Checkpoint val+test 评估 (per-class, Head/Mid/Tail) |
| `runs/exp6_test50/auto_continue.py` | **新增** 续训自动化监控与决策脚本 |
| `runs/exp6_test50/prepare_next_chunk.py` | **新增** 训练 chunk 配置生成器 |

## 调优经验 (2026-06-29)

### 实验总结

| 实验 | 策略 | 结果 |
|------|------|------|
| Exp2 | 高lr warm restart (MuSGD, lr0=0.02, 从best.pt) | ✅ +0.014, mAP50-95=0.388 |
| Exp3 | Focal gamma=1.0 + 高lr | ❌ 收敛慢一倍 |
| Exp1 | Focal gamma=2.0 从零训练 | ❌ cls_loss=1e-5, 分类不学 |
| Exp5 | 低lr精调Exp2 best (lr0=0.005) | ❌ 未超Exp2, 模型容量到顶 |
| **Exp6** | **Mosaic重开续训50ep (lr=0.0003)** | **✅ +0.043, mAP50-95=0.417, 所有类无下降** |
| **Exp6+** | **续训冲分 (25ep×2, 逐块评估)** | **ep75峰值 0.4178, ep100回退** |

### ⭐ 容量天花板确认 (Exp6+, 2026-06-29)

Exp6 后继续续训，每 25 epoch 评估一次：
- ep50 (起点): test mAP50-95=0.4149
- **ep75 (🏆 最佳): test mAP50-95=0.4178** (+0.0029, Truck Tractor +0.026)
- ep100: test mAP50-95=0.4144 (-0.0034 回退)

**结论**: yolo11n (2.7M params) 在 ep75 达到容量上限 0.418。再训 25 epoch 即出现 overfitting（Tractor -0.024, Bus -0.006）。
Head 类别完全饱和 (Small Car/Van ±0.001)，唯一持续受益的是极长尾 Truck Tractor (24 test 样本, +24.7%)。
最佳模型: `runs/exp6_ep75/weights/best.pt`

### 关键Bug及修复

1. **FocalLoss返回标量** (`loss.py:86`): `return loss.mean(1).sum()` → `return loss`。标量再除target_scores_sum(~5100万)导致cls_loss=1e-5。
2. **WiseIoU rbox2dist调用** (`loss.py:1263`): 参数`(anchor_points, target_bboxes)`反了，且缺`[...,:4]`剥离角度。修复为`(target_bboxes[...,:4], anchor_points, ..., reg_max=...)`。
3. **RTDETR适配** (`models/utils/loss.py:114`): FocalLoss改为per-element后需补`.mean(1).sum()`。

### optimizer=auto隐藏行为

`engine/trainer.py:1112-1120`: 当optimizer=auto时强制覆盖:
- `lr0` → 0.01 (忽略自定义值)
- `momentum` → 0.9
- `warmup_bias_lr` → **0.0** (硬编码!)

要自定义lr/warmup_bias_lr，**必须用 `optimizer=MuSGD`**。

### DDP pickle截断

症状: `_pickle.UnpicklingError: pickle data was truncated`

原因: 多次DDP崩溃后multiprocessing forkserver僵尸进程干扰IPC。

修复: 每次启动DDP前清理:
```bash
ps aux | grep -E "multiprocessing|forkserver|resource_tracker" | grep $(whoami) | awk '{print $2}' | xargs kill -9
```

### GPU隔离

YAML `device: 4,5,6,7` + ultralytics内部`select_device()`自动设CUDA_VISIBLE_DEVICES。**不要额外export CUDA_VISIBLE_DEVICES**。8卡切4+4并行不可靠，建议串行。

### 当前实验配置

| 文件 | 用途 | 关键参数 |
|------|------|----------|
| `configs/baseline.yaml` | 基线 yolo11n-obb 640 | 标准 |
| `configs/enhanced.yaml` | P2+800增强 | 当前基线 |
| `configs/exp_mosaic_base.yaml` | **Mosaic策略对比基准** (基于exp1_focal_v10) | Focal γ=2.0, 唯一变量close_mosaic |
| `configs/exp5_finetune.yaml` | 低lr精调 | lr0=0.005, 从Exp2 best.pt |
| `configs/exp6_test.yaml` | Mosaic重开续训 | lr=0.0003, close_mosaic=30, 效果最好 |

### ⭐ Mosaic 策略对比实验 (TODO02, 进行中)

| 方案 | close_mosaic | 实验名 | 状态 |
|------|-------------|--------|------|
| A | 15 | exp_mosaic_A_close15 | ✅ 完成 |
| B | 30 | exp1_focal_v10 | ✅ 复用 |
| C | 100 | exp_mosaic_C_close100 | 🔄 进行中 |
| D | 300 | exp_mosaic_D_close300 | ⏳ |
| E | 500 | exp_mosaic_E_close500 | ⏳ |
| F | 0 (永不关闭) | exp_mosaic_F_never | ⏳ |

实验链: `run_mosaic_experiments.sh` 自动串行 C→D→E→F，完成后 `tools/benchmark_mosaic.py` 生成对比报告。

> ⚠️ `exp_mosaic_base.yaml` 使用 `optimizer: auto` → lr0 被强制 0.01、warmup_bias_lr 被强制 0.0。如需自定义这些值，改用 `optimizer: MuSGD`。

### ⭐ Mosaic 关键发现 (Exp6, 2026-06-29)

原训练 `close_mosaic=15` → ep16 后永久关 Mosaic → 984 轮无增强 → 模型学僵。
从 best.pt 续训 + 重开 Mosaic (`close_mosaic=30`), 仅 50 epoch, lr=0.0003:
- **val mAP50-95: 0.374 → 0.417 (+0.043)**
- **test mAP50-95: 0.371 → 0.415 (+0.044)**
- **全部 10 个类别无一下降**
- 增益与样本量成反比: Bus (+0.105) >> Small Car (+0.019)

**教训**: close_mosaic 不宜过早。Mosaic 对长尾类别是"场景解耦"手段，不仅是数据增强。
