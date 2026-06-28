# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 常用命令

```bash
# 训练
python train.py --enhanced --fold 0                          # enhanced 配置
python train.py --baseline --fold 0                          # baseline 配置
python train.py --fold 0 --model yolo11n-obb-p2.yaml --use-focal  # ablation

# 断点续训（必须用完整目录名）
python train.py --enhanced --fold 0 --epochs 1000 --name enhanced_fold0_1000ep-10 --resume

# Profiling
python tools/benchmark_dataloader.py --config enhanced --full

# 训练后手动 test 评估
python -c "from ultralytics import YOLO; m=YOLO('runs/xxx/weights/best.pt'); m.val(data='dataset_yolo/fold_0/data.yaml', split='test')"

# 监控
tail -f /tmp/train_final.log
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
| `utils/loss.py` | `v8OBBLoss` 增强：Focal/WiseIoU/KLD/SlideLoss/ScaleAware 开关 |
| `utils/tal.py` | `RotatedTaskAlignedAssigner`：点积测试判断锚点在旋转框内 |
| `utils/callbacks/base.py` | 移除训练内 test 回调（DDP 死锁），改为 no-op |
| `data/build.py` | `persistent_workers=True` + worker 数量公式优化 |
| `utils/profile_loader.py` | **新增** DataLoader Profiling 框架 |
| `engine/trainer.py` | Profiling 集成（profile_loader 开关） |
| `cfg/default.yaml` | 新增增强超参数注册 |
| `cfg/models/11/yolo11-obb-p2.yaml` | **新增** P2 检测层模型定义 |
