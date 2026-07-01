# 任务：重新设计完整训练流程，验证 Mosaic 调度策略，构建最终比赛训练方案

## 背景

已有实验表明：

原始训练：

close_mosaic=15

导致：

1000 epoch中绝大多数训练没有 Mosaic。

重新打开 Mosaic 后，

仅50 epoch，

mAP50-95 提升约0.044。

因此怀疑：

原训练策略存在系统性问题。

现在需要重新训练，而不是继续修补旧模型。

目标：

重新寻找最优训练策略。

---

# 第一部分：实验目标

验证：

不同 Mosaic 调度策略，

对最终成绩的影响。

最终确定：

以后比赛统一采用哪一种训练策略。

不是追求一次实验最好，

而是建立长期可复用方案。

---

# 第二部分：实验变量

保持：

模型

数据集

优化器

batch

输入尺寸

全部一致。

唯一改变：

Mosaic 调度。

建议至少比较：

方案A：

close_mosaic=15

方案B：

close_mosaic=30

方案C：

close_mosaic=100

方案D：

close_mosaic=300

方案E：

close_mosaic=500

方案F：

始终开启 Mosaic（Never Close）

如果框架支持，

增加：

动态 Mosaic 概率衰减方案。

---

# 第三部分：训练计划

所有实验：

统一训练。

epoch保持一致。

不要因为策略不同，

训练epoch不同。

确保：

公平比较。

建议：

使用统一随机种子。

统一保存：

best

last

checkpoint。

---

# 第四部分：评估内容

每个实验：

输出：

整体：

mAP50

mAP50-95

Precision

Recall

Loss

分类：

所有类别 AP。

重点统计：

Head

Middle

Tail

Scale0

Scale1

Scale2

Scale3

全部记录。

---

# 第五部分：重点分析

回答：

Mosaic 是否：

真正改善：

Tail。

还是：

改善全部类别。

分析：

不同 close_mosaic

分别影响：

Head

Tail

Scale0

Scale3

哪个受益最大。

输出：

完整对比表。

---

# 第六部分：自动生成对比报告

所有实验完成后，

自动生成：

《Mosaic Strategy Benchmark.md》

包括：

总表：

| Mosaic策略 | mAP | Tail AP | Scale0 AP | FPS |

排序：

最好

↓

最差

另外：

绘制：

不同 Mosaic 策略

↓

mAP 曲线。

Tail AP 曲线。

Scale AP 曲线。

---

# 第七部分：自动推荐最终训练方案

根据实验结果，

自动给出：

推荐：

★★★★★

推荐理由。

例如：

如果：

close_mosaic=300

最好，

则分析：

为什么。

如果：

Never Close

最好，

分析：

为什么。

最后输出：

建议以后比赛统一采用的训练策略。

---

# 第八部分：进一步思考

除了 Mosaic，

请结合本项目特点（遥感、小目标、OBB、长尾类别、密集场景），分析还有哪些训练策略值得作为下一阶段实验。

按收益排序，

列出：

* 优先级
* 实现难度
* 预计收益
* 是否建议立即开展

最终形成：

《下一阶段训练路线图.md》。

要求整个实验过程高度自动化，并保证所有实验具有可复现性和公平可比性。

