# 任务：生成系统架构文档（CLAUDE.system.md）

## 目标

请完整阅读整个项目源码（不仅限于 train.py），分析整个训练系统的架构，并在项目根目录生成一份

```
CLAUDE.system.md
```

该文档的目标不是代码注释，而是作为整个项目的"系统说明书（System Architecture）"。

要求：

> 任何新的 Claude Code Agent、Codex Agent 或开发者，只阅读这一份 md 文件，就能够在 5~10 分钟内快速理解整个系统，而无需阅读大量源码。

文档要求全面、准确、可维护，并与源码保持一致。

---

# 一、项目概览（Overview）

描述：

* 项目用途
* 支持哪些任务（Detect / Segment / Pose / OBB / Classify 等）
* 当前默认训练流程
* 当前分支主要针对什么任务
* 整个工程目录结构

例如：

```
Project
│
├── ultralytics/
├── datasets/
├── models/
├── engine/
├── utils/
├── nn/
├── cfg/
└── ...
```

并说明每个目录负责什么。

---

# 二、系统整体流程（Pipeline）

绘制完整流程（Mermaid）。

例如：

```
Dataset
    │
    ▼
DataLoader
    │
    ▼
Data Augmentation
    │
    ▼
Model
    │
    ▼
Forward
    │
    ▼
Loss
    │
    ▼
Backward
    │
    ▼
Optimizer
    │
    ▼
Scheduler
    │
    ▼
EMA
    │
    ▼
Validation
    │
    ▼
Metrics
    │
    ▼
Checkpoint
```

并逐步解释。

---

# 三、系统输入

说明：

## 数据输入

包括：

* Dataset yaml
* images
* labels
* OBB Label格式
* Polygon格式
* JSON/XML转换流程（如果存在）

说明：

数据如何一步步进入模型。

---

## 配置输入

列出：

* train参数
* val参数
* predict参数
* cfg
* model yaml
* hyp

说明：

配置最终流向哪里。

---

# 四、DataLoader

完整说明：

DataLoader

↓

Dataset

↓

Transforms

↓

Collate

↓

Batch

包括：

* worker
* cache
* mosaic
* mixup
* copypaste
* affine
* hsv
* flip

说明：

每一步在哪里实现。

对应源码：

```
文件

类

函数
```

---

# 五、模型结构

说明：

Model 构建流程：

```
yaml

↓

parse_model

↓

Backbone

↓

Neck

↓

Head
```

说明：

每层来自哪里。

包括：

* Detect Head
* OBB Head
* DFL
* Anchor
* Decoder

说明：

Forward流程。

---

# 六、输入输出

整理：

模型输入：

```
Tensor Shape

dtype

device
```

模型输出：

例如：

```
cls

box

angle

score
```

每一项：

shape

含义

单位

范围

---

# 七、Loss

这是重点。

完整分析：

Loss组成。

例如：

```
Loss

├── Box Loss

├── Cls Loss

├── DFL

├── Angle Loss

└── ...
```

说明：

每个Loss：

来源

计算方式

输入

输出

权重

在哪里调用。

最好给出公式。

---

# 八、Optimizer

整理：

支持：

* SGD
* Adam
* AdamW

说明：

默认使用哪个。

在哪里创建。

哪些参数不参与weight decay。

如何分组。

---

# 九、LR Scheduler

说明：

支持：

* Cosine
* Linear
* OneCycle

默认：

如何更新。

Warmup流程。

---

# 十、EMA

说明：

EMA：

什么时候更新。

保存什么。

推理是否使用EMA。

---

# 十一、Validation

说明：

验证流程：

```
Model

↓

Inference

↓

NMS

↓

Metrics
```

包括：

mAP

Precision

Recall

F1

OBB IoU

Conf

NMS

---

# 十二、Checkpoint

说明：

保存：

```
best.pt

last.pt

ema.pt
```

分别保存哪些内容。

恢复训练流程。

---

# 十三、训练循环

画完整流程：

```
Epoch

↓

Batch

↓

Forward

↓

Loss

↓

Backward

↓

Optimizer

↓

EMA

↓

Log

↓

Validation

↓

Save
```

注明：

每一步在哪个文件。

哪个函数。

---

# 十四、配置系统

整理：

所有影响训练的重要参数。

例如：

```
epochs

batch

workers

cache

device

amp

mosaic

mixup

close_mosaic

optimizer

lr0

lrf

weight_decay

warmup

box

cls

dfl
```

每项：

默认值

作用

读取位置。

---

# 十五、日志系统

说明：

训练日志：

TensorBoard

CSV

Console

Weights&Biases（如果支持）

保存位置。

---

# 十六、源码调用关系

建立模块依赖图。

例如：

```
train.py

↓

Trainer

↓

Model

↓

Dataset

↓

Loss

↓

Optimizer

↓

Validator
```

不要只写文件名，要写调用关系。

---

# 十七、扩展点（非常重要）

整理：

如果以后要修改：

## 新Loss

改哪里。

## 新Head

改哪里。

## 新Backbone

改哪里。

## 新Augmentation

改哪里。

## 新Dataset

改哪里。

## 新Metrics

改哪里。

## 新Optimizer

改哪里。

## 新Scheduler

改哪里。

方便其他Agent快速开发。

---

# 十八、关键类索引（Index）

建立索引。

格式：

| 类 | 文件 | 作用 |
| - | -- | -- |

例如：

Trainer

DetectionTrainer

OBBTrainer

BaseModel

OBBModel

DetectionLoss

TaskAlignedAssigner

Validator

Dataset

等等。

---

# 十九、关键函数索引

建立：

| 函数 | 文件 | 作用 |

例如：

train()

build_dataset()

build_dataloader()

forward()

loss()

validator()

save_model()

parse_model()

等。

---

# 二十、修改建议

最后增加：

```
System Optimization Suggestions
```

总结：

目前代码：

哪些模块耦合较高。

哪些模块建议解耦。

哪些模块适合重构。

哪些模块适合Profiling。

哪些地方容易成为性能瓶颈（DataLoader、Augmentation、DDP、Loss、NMS等）。

---

# 文档要求

1. 所有内容必须依据源码生成，不允许猜测。
2. 所有重要结论必须标明对应源码位置（文件+类+函数）。
3. 使用 Markdown 编写。
4. 使用 Mermaid 绘制整体流程图。
5. 建立目录（Table of Contents）。
6. 文档长度不限，越完整越好。
7. 最终保存到：

```
CLAUDE.system.md
```

放在项目根目录。

要求该文件能够作为整个项目唯一的系统架构说明书，供后续所有 AI Agent 和开发者阅读使用。

