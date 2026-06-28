# 任务

请完整阅读当前整个 Ultralytics (YOLO11-OBB) 训练框架源码。

Github：

https://github.com/ultralytics/ultralytics

我的目标不是保持官方默认实现，而是：

**针对我的数据集做比赛级(SOTA)优化，以最终排行榜分数最高为唯一目标。**

请不要考虑代码改动是否大，也不要考虑是否兼容官方版本，只考虑最终精度（mAP）最高。

所有修改必须真正修改源码，而不是仅仅修改yaml参数。

最后请给出：

1. 修改原因
2. 修改位置
3. 修改后的代码
4. 是否影响推理速度
5. 是否建议开启
6. ablation建议

---------------------------------------

# 数据集特点

这是遥感车辆OBB检测。

图片：

9442张

目标：

468979个

类别：

10类

全部都是OBB

Small Car
Van

占87%

属于严重长尾。

目标尺寸：

绝大部分目标

polygon_size

15~20px

约45%的目标小于18px。

属于典型遥感小目标。

单图平均：

49.7个目标

最高：

1074个目标

属于密集检测。

已经提供：

scale_id

0

1

2

3

表示目标尺度。

图片大小：

1000×1000

800×600

训练资源：

8*A100 40G

因此：

训练速度不是限制。

显存不是限制。

最终排行榜成绩才是目标。

---------------------------------------

# 修改原则

请不要局限于官方实现。

请阅读整个框架。

包括：

ultralytics/

下面所有源码。

包括：

dataset

augment

loss

head

trainer

validator

assigner

tal

metrics

engine

model

cfg

全部分析。

寻找所有可以提升比赛成绩的位置。

不是调参。

而是真正修改源码。

---------------------------------------

# 请重点分析以下方向

## 1

是否需要新增P2检测层

对于15px左右目标是否有帮助。

如果有：

请直接修改网络。

包括：

yaml

Detect Head

Stride

Anchor

Loss

全部同步修改。

---------------------------------------

## 2

Loss

目前官方Loss是否最优。

请分析：

DFL

BCE

VFL

QFL

Focal

Slide Loss

Wise-IoU

MPDIoU

SIoU

Inner-IoU

PIoU

Shape-IoU

是否值得替换。

如果值得：

请直接替换。

---------------------------------------

## 3

Rotated Loss

分析OBB Loss。

角度Loss是否还能优化。

Rotated IoU

KLD

GWD

KFIoU

是否值得加入。

---------------------------------------

## 4

Label Assignment

阅读：

TaskAlignedAssigner

请分析：

是否适合

密集遥感

小目标

长尾

是否需要修改。

例如：

Dynamic TopK

SimOTA

OTA

PAA

ATSS

是否更优。

---------------------------------------

## 5

Data Augmentation

官方：

Mosaic

Mixup

CopyPaste

RandomPerspective

HSV

是否最优。

请重新设计。

例如：

Scale-aware Mosaic

Rotation Mosaic

Small Object CopyPaste

Dense CopyPaste

MultiScale Crop

等等。

---------------------------------------

## 6

Scale-aware

我的数据拥有：

scale_id

请充分利用。

不要浪费。

例如：

Loss Weight

Sampling

Curriculum Learning

Scale-aware Batch

Scale-aware Mosaic

Scale-aware Head

任选。

---------------------------------------

## 7

Class Imbalance

目前：

Small Car

Van

87%

其它类别极少。

请重新设计：

Loss

Sampler

Weight

Repeat Sampling

Class Balanced Sampling

Equalization Loss

Balanced Softmax

LDAM

等等。

---------------------------------------

## 8

Dense Detection

单图1000多个目标。

请分析：

NMS

Rotated NMS

Soft NMS

Cluster NMS

Matrix NMS

Weighted Box Fusion

是否值得修改。

---------------------------------------

## 9

训练策略

重新设计：

EMA

Warmup

LR

Cosine

Batch

AMP

Gradient Accumulation

Gradient Checkpoint

Multi-scale

Multi-stage Training

Fine-tuning

TTA

全部重新分析。

---------------------------------------

## 10

推理

是否建议：

Multi-scale Test

TTA

Flip

Rotate

Model Ensemble

WBF

Soft Voting

等等。

---------------------------------------

## 11

YOLO11结构

请重新分析：

Backbone

SPPF

C2PSA

Attention

Neck

PAN

BiFPN

DyHead

EMA Attention

DCNv4

是否值得替换。

---------------------------------------

## 12

遥感优化

请结合：

DOTA

FAIR1M

HRSC2016

DIOR

RSOD

公开比赛经验。

哪些改动最有效。

---------------------------------------

# 输出要求

不要泛泛而谈。

请按照：

优先级★★★★★

★★★★★

★★★★☆

★★★☆☆

排序。

对于每一项：

说明：

为什么有效。

修改哪些文件。

修改哪些函数。

修改哪些类。

预计提升：

+0.2 mAP

+0.5 mAP

+1 mAP

等等。

---------------------------------------

# 最终目标

最终请形成一个：

"YOLO11-OBB 遥感车辆比赛增强版"

要求：

这是一个真正可以用于比赛冲榜的版本。

不是论文Demo。

不是参数修改。

而是整个训练框架针对遥感OBB检测进行全面升级。

请最终输出：

1. 修改路线图
2. 每一步修改内容
3. 修改优先级
4. 风险分析
5. 推荐实施顺序

要求每一步都尽可能独立，方便逐步验证Ablation。

不要一次修改所有内容，而是按照"收益最高优先"进行排序。



## 开发要求（必须遵守）

任何修改都必须满足以下原则：

1. 一次 Commit 只实现一个优化点。
2. 每个优化点必须能够单独开启/关闭（通过配置项控制）。
3. 每个优化点都需要保留 Baseline 对照实验。
4. 修改完成后生成对应的 ablation 实验计划。
5. 不允许多个优化点耦合在一起，否则无法分析收益来源。
6. 所有新增模块必须兼容 YOLO11-OBB，不破坏已有接口。
7. 优先修改收益最大的模块，再进行次优模块开发。


