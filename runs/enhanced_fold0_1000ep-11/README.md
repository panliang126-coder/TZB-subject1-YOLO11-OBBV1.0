# enhanced_fold0_1000ep-11

**实验目的**: 无，误创建

**原因**: resume 时 name 参数写错（写了 `enhanced_fold0_1000ep` 而非 `enhanced_fold0_1000ep-10`），ultralytics 找不到 checkpoint 自动创建了新目录从头训练

**结果**: 立即停止并删除
