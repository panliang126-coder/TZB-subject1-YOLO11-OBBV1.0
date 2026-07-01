# PRETRAIN_FINETUNE_PLAN.md

# YOLO11-OBB 官方预训练模型微调实验方案

## 一、任务目标

当前 YOLO11n-OBB 模型采用 **Scratch（随机初始化）** 训练。

为了验证官方预训练模型对于当前遥感 OBB 数据集的收益，建立一套完整的迁移学习基准实验。

本阶段目标不是追求最终最高分，而是回答：

1. 官方预训练是否明显优于 Scratch？
2. 哪个模型容量（n/s/m/l/x）最适合作为后续一个月的主力模型？
3. 后续所有实验应基于哪个预训练模型继续优化？

---

# 二、实验原则

本阶段只允许改变：

* 预训练权重
* 模型容量

除此之外：

所有训练参数保持一致。

严禁：

* 修改网络结构
* 修改 Neck
* 修改 Head
* 修改 Loss
* 修改数据增强
* 修改 Dataset
* 修改 Label
* 修改训练策略

目的：

确保唯一变量为：

> Official Pretrained Weight

---

# 三、实验设计

## Baseline

当前结果：

YOLO11n-OBB-P2

Scratch Training

作为 Baseline。

---

## Experiment 1

模型：

YOLO11n

初始化：

官方预训练权重

目的：

验证：

Pretrained vs Scratch

这是最重要的一组实验。

---

## Experiment 2

模型：

YOLO11s

初始化：

官方预训练权重

目的：

验证：

增加模型容量是否带来收益。

---

## Experiment 3

模型：

YOLO11m

初始化：

官方预训练权重

目的：

继续验证模型容量。

---

## Experiment 4（可选）

模型：

YOLO11l

初始化：

官方预训练权重

如果 GPU 时间充足再执行。

---

## Experiment 5（探索）

如果框架支持：

YOLO26x

官方预训练权重

作为容量上限探索。

如果实验初期表现明显落后，

立即停止。

不要浪费 GPU。

---

# 四、训练策略

所有实验：

保持：

* 相同 Epoch
* 相同 imgsz
* 相同 Optimizer
* 相同 LR
* 相同 Scheduler
* 相同 Batch（尽可能一致）
* 相同数据增强
* 相同随机种子

除模型容量和预训练权重外，

任何变量不得修改。

---

# 五、自动记录

每个实验自动记录：

* Model
* Weight
* Params
* GFLOPs
* Batch
* GPU Memory
* Epoch Time
* Best Epoch
* mAP50
* mAP50-95
* Precision
* Recall
* Train Loss
* Val Loss

输出统一 Markdown 表格。

---

# 六、自动分析

训练结束以后，

从以下几个方面分析：

## 1

Pretrained 是否明显优于 Scratch？

提升：

* mAP50
* mAP50-95
* Recall
* 收敛速度

---

## 2

模型容量是否已经达到瓶颈？

分析：

n

↓

s

↓

m

↓

l

↓

x

收益曲线。

不要只比较最终 mAP。

同时比较：

* 收敛速度
* Recall
* 小目标表现
* GPU 开销
* 每 Epoch 时间

---

## 3

收益分析

计算：

每增加一个模型等级，

带来的：

* mAP 提升
* GPU 时间增加
* 显存增加

评估：

Accuracy / Cost

是否值得。

---

# 七、输出

输出：

## 排名

Top1

Top2

Top3

...

---

## 推荐

推荐：

最终模型：

推荐理由：

预计最终潜力：

预计继续优化空间：

是否建议作为比赛最终 Backbone。

---

# 八、下一阶段建议

根据实验结果，

自动制定下一阶段计划。

例如：

如果：

YOLO11m

明显最好，

则后续所有：

学习率

数据增强

Loss

imgsz

Optimizer

全部基于：

YOLO11m

继续优化。

不要再继续维护多个模型。

集中资源优化最佳模型。

---

# 九、决策原则

你不是论文作者。

你的目标不是证明更大的模型一定更好。

你的目标只有一个：

> 找到未来一个月最值得持续投入的模型。

所有结论必须由实验数据支撑。

如果：

YOLO11s 与 YOLO11m 的差距极小，

但训练成本增加很多，

请优先推荐：

YOLO11s。

如果：

YOLO11m 带来明显收益，

则直接建议：

后续所有实验全部迁移至 YOLO11m。

不要输出理论分析。

不要泛泛而谈。

直接给出：

实验结论

推荐模型

下一步实验计划。
