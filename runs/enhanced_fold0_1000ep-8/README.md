# enhanced_fold0_1000ep-8

**实验目的**: 调试 cache=disk 性能问题

**配置**: batch=256, imgsz=800, cache=disk, 8×A100

**结果**: ⚠️ 跑了 4 个 epoch 后手动停止

**原因**: 磁盘 I/O 瓶颈，方案放弃
