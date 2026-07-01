# enhanced_fold0_1000ep-9

**实验目的**: 回到 cache=false，验证 batch=256 是否可行

**配置**: batch=256, imgsz=800, cache=false, 8×A100

**结果**: ❌ TaskAlignedAssigner CUDA OOM，确认 batch=256 不可行

**结论**: enhanced 配置（P2+800）最大稳定 batch=96 (12/GPU)
