#!/usr/bin/env python3
"""
Mosaic 调度策略对比分析脚本
=============================
自动收集所有 Mosaic 实验的 results.csv 和 test 评估结果，
生成对比表、曲线图、和最终报告。

用法:
    # 所有实验完成后运行
    python tools/benchmark_mosaic.py

    # 指定实验目录
    python tools/benchmark_mosaic.py --exp-dirs runs/exp_mosaic_A_close15 runs/exp1_focal_v10 ...
"""

import argparse
import sys
from pathlib import Path
import yaml
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime

# Matplotlib 后端设置
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# 实验定义
MOSAIC_EXPERIMENTS = {
    'A': {'name': 'exp_mosaic_A_close15', 'close_mosaic': 15,  'label': 'A: close=15 (early)',
          'desc': '方案A: 早期关闭 (close_mosaic=15)'},
    'B': {'name': 'exp1_focal_v10',        'close_mosaic': 30,  'label': 'B: close=30 (v10 baseline)',
          'desc': '方案B: v10基线 (close_mosaic=30)'},
    'C': {'name': 'exp_mosaic_C_close100', 'close_mosaic': 100, 'label': 'C: close=100 (mid)',
          'desc': '方案C: 中期关闭 (close_mosaic=100)'},
    'D': {'name': 'exp_mosaic_D_close300', 'close_mosaic': 300, 'label': 'D: close=300 (late)',
          'desc': '方案D: 晚期关闭 (close_mosaic=300)'},
    'E': {'name': 'exp_mosaic_E_close500', 'close_mosaic': 500, 'label': 'E: close=500 (equiv never)',
          'desc': '方案E: 等价永不关闭 (close_mosaic=500)'},
    'F': {'name': 'exp_mosaic_F_never',    'close_mosaic': 0,   'label': 'F: close=0 (never)',
          'desc': '方案F: 永不关闭 (close_mosaic=0)'},
}

# 类别定义（按 Head/Middle/Tail 分组）
CLASS_GROUPS = {
    'Head':  ['Small Car', 'Large Bus', 'Bus', 'Truck', 'Van'],
    'Middle': ['Cargo Truck', 'Dump Truck', 'Trailer'],
    'Tail':   ['Tractor', 'Truck Tractor'],
}

# Scale 分组 (根据像素面积)
# Scale0: area < 32², Scale1: 32²~96², Scale2: 96²~160², Scale3: > 160²


def find_run_dir(project_root, exp_name):
    """查找实验目录"""
    run_dir = project_root / 'runs' / exp_name
    if run_dir.exists():
        return run_dir
    # 尝试查找近似匹配
    for d in (project_root / 'runs').iterdir():
        if d.is_dir() and d.name.startswith(exp_name):
            return d
    return None


def load_results_csv(run_dir):
    """加载 results.csv"""
    csv_path = run_dir / 'results.csv'
    if not csv_path.exists():
        return None
    return pd.read_csv(csv_path)


def load_args_yaml(run_dir):
    """加载 args.yaml"""
    args_path = run_dir / 'args.yaml'
    if not args_path.exists():
        return None
    with open(args_path) as f:
        return yaml.safe_load(f)


def load_test_metrics(run_dir):
    """从 test 评估结果加载指标。
    查找 val 目录下的 test 评估结果，或运行评估。
    """
    # 首先检查是否有专门的 test 评估结果
    test_dirs = sorted(run_dir.glob('val*'), reverse=True)
    for val_dir in test_dirs:
        # 检查是否是 test split 的结果
        val_args = val_dir / 'args.yaml'
        if val_args.exists():
            with open(val_args) as f:
                va = yaml.safe_load(f)
            if va.get('split') == 'test':
                # 找到 test 评估结果
                metrics = {}
                # 读取 per-class 指标
                for csv_file in val_dir.glob('*.csv'):
                    df = pd.read_csv(csv_file)
                    metrics['csv_data'] = df
                return metrics

    # 检查 val split 的结果
    for val_dir in test_dirs:
        val_args = val_dir / 'args.yaml'
        if val_args.exists():
            with open(val_args) as f:
                va = yaml.safe_load(f)
            if va.get('split') == 'val':
                metrics = {}
                for csv_file in val_dir.glob('*.csv'):
                    df = pd.read_csv(csv_file)
                    metrics['csv_data'] = df
                return metrics

    return None


def extract_best_metrics(df):
    """从 results.csv 提取最佳指标"""
    if df is None or len(df) == 0:
        return None

    # 列名可能带空格或不同格式
    map50_col = None
    map50_95_col = None
    for col in df.columns:
        if 'mAP50' in col and '95' not in col:
            map50_col = col
        if 'mAP50-95' in col or 'mAP50_95' in col:
            map50_95_col = col

    if map50_95_col is None:
        return None

    best_idx = df[map50_95_col].idxmax()
    best_row = df.iloc[best_idx]

    metrics = {
        'best_epoch': int(best_idx) + 1,
        'best_mAP50': float(best_row[map50_col]) if map50_col else None,
        'best_mAP50_95': float(best_row[map50_95_col]),
    }

    # 提取其他列
    for col in df.columns:
        col_lower = col.strip().lower()
        if 'precision' in col_lower:
            metrics['best_precision'] = float(best_row[col])
        if 'recall' in col_lower:
            metrics['best_recall'] = float(best_row[col])
        if 'box_loss' in col_lower and 'train' in col_lower:
            metrics['final_box_loss'] = float(df.iloc[-1][col])
        if 'cls_loss' in col_lower and 'train' in col_lower:
            metrics['final_cls_loss'] = float(df.iloc[-1][col])
        if 'dfl_loss' in col_lower and 'train' in col_lower:
            metrics['final_dfl_loss'] = float(df.iloc[-1][col])
        if 'angle_loss' in col_lower and 'train' in col_lower:
            metrics['final_angle_loss'] = float(df.iloc[-1][col])

    # 计算达到最佳 epoch 的时间（假设均匀）
    if 'epoch' in df.columns:
        total_time = None
        for col in df.columns:
            if 'time' in col.lower():
                total_time = df.iloc[-1][col]
                break
        if total_time is not None:
            metrics['total_time_hours'] = round(total_time, 2)
            metrics['time_to_best_hours'] = round(total_time * (best_idx + 1) / len(df), 2)

    return metrics


def plot_mAP_curves(results, output_dir):
    """绘制 mAP50-95 对比曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    sorted_exps = sorted(results.keys())

    # 1. mAP50-95 曲线
    ax = axes[0, 0]
    for i, exp_id in enumerate(sorted_exps):
        r = results[exp_id]
        if r['df'] is not None:
            map50_95_col = None
            for col in r['df'].columns:
                if 'mAP50-95' in col or 'mAP50_95' in col:
                    map50_95_col = col
                    break
            if map50_95_col:
                ax.plot(r['df'][map50_95_col], color=colors[i], label=r['label'],
                        linewidth=1.5, alpha=0.9)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('mAP50-95')
    ax.set_title('mAP50-95 vs Epoch')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)

    # 2. mAP50 曲线
    ax = axes[0, 1]
    for i, exp_id in enumerate(sorted_exps):
        r = results[exp_id]
        if r['df'] is not None:
            map50_col = None
            for col in r['df'].columns:
                if 'mAP50' in col and '95' not in col:
                    map50_col = col
                    break
            if map50_col:
                ax.plot(r['df'][map50_col], color=colors[i], label=r['label'],
                        linewidth=1.5, alpha=0.9)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('mAP50')
    ax.set_title('mAP50 vs Epoch')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)

    # 3. Box Loss 曲线
    ax = axes[1, 0]
    for i, exp_id in enumerate(sorted_exps):
        r = results[exp_id]
        if r['df'] is not None:
            box_col = None
            for col in r['df'].columns:
                if 'box_loss' in col.lower() and 'train' in col.lower():
                    box_col = col
                    break
            if box_col:
                # Smooth with moving average
                loss = r['df'][box_col].values
                smooth = pd.Series(loss).rolling(10, min_periods=1).mean()
                ax.plot(smooth, color=colors[i], label=r['label'],
                        linewidth=1.0, alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Train Box Loss (smoothed)')
    ax.set_title('Box Loss vs Epoch')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)

    # 4. Best mAP50-95 对比柱状图
    ax = axes[1, 1]
    exp_labels = []
    best_maps = []
    bar_colors = []
    for i, exp_id in enumerate(sorted_exps):
        r = results[exp_id]
        if r['metrics']:
            exp_labels.append(f"{exp_id}\n(close={r['close_mosaic']})")
            best_maps.append(r['metrics']['best_mAP50_95'])
            bar_colors.append(colors[i])
        else:
            exp_labels.append(f"{exp_id}\n(close={r['close_mosaic']})")
            best_maps.append(0)
            bar_colors.append(colors[i])

    bars = ax.bar(range(len(exp_labels)), best_maps, color=bar_colors, edgecolor='white')
    ax.set_xticks(range(len(exp_labels)))
    ax.set_xticklabels(exp_labels, fontsize=8)
    ax.set_ylabel('Best mAP50-95')
    ax.set_title('Best mAP50-95 Comparison')

    # 在柱状图上标注数值
    for bar, val in zip(bars, best_maps):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.002,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 设置 y 轴范围留出标注空间
    if best_maps:
        y_max = max(best_maps) * 1.03
        y_min = min(best_maps) * 0.97
        ax.set_ylim(y_min, y_max)

    fig.suptitle('Mosaic Strategy Comparison — Training Curves', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(output_dir / 'mosaic_training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Plot] Training curves saved: {output_dir / 'mosaic_training_curves.png'}")


def plot_lr_curves(results, output_dir):
    """绘制学习率曲线对比"""
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    sorted_exps = sorted(results.keys())

    for i, exp_id in enumerate(sorted_exps):
        r = results[exp_id]
        if r['df'] is not None:
            lr_col = None
            for col in r['df'].columns:
                if 'lr/pg0' in col or 'lr0' in col.lower():
                    lr_col = col
                    break
            if lr_col:
                ax.plot(r['df'][lr_col], color=colors[i], label=r['label'],
                        linewidth=1.5, alpha=0.8)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate (pg0)')
    ax.set_title('LR Schedule Comparison')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'mosaic_lr_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Plot] LR curves saved: {output_dir / 'mosaic_lr_curves.png'}")


def generate_markdown_report(results, output_dir):
    """生成 Mosaic Strategy Benchmark.md 报告"""
    report_path = output_dir / 'Mosaic Strategy Benchmark.md'
    sorted_exps = sorted(results.keys())

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Mosaic 调度策略对比基准测试报告\n\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")

        # 实验概述
        f.write("## 1. 实验概述\n\n")
        f.write("| 方案 | close_mosaic | 含义 | 实验名 | 状态 |\n")
        f.write("|------|-------------|------|--------|------|\n")
        for exp_id in sorted_exps:
            r = results[exp_id]
            status = '✅ 完成' if r['metrics'] else ('🔄 进行中' if r['df'] is not None else '❌ 未开始')
            f.write(f"| {exp_id} | {r['close_mosaic']} | {r.get('desc', r['label'])} | {r['name']} | {status} |\n")
        f.write("\n")

        # 整体指标对比
        f.write("## 2. 整体指标对比\n\n")
        available = {k: v for k, v in results.items() if v['metrics']}
        if available:
            f.write("| 方案 | close_mosaic | Best Epoch | mAP50 | mAP50-95 | Precision | Recall | 训练时间(h) |\n")
            f.write("|------|-------------|------------|-------|----------|-----------|--------|------------|\n")

            # 按 mAP50-95 排序
            ranked = sorted(available.items(), key=lambda x: x[1]['metrics']['best_mAP50_95'], reverse=True)
            best_map = ranked[0][1]['metrics']['best_mAP50_95']

            for rank, (exp_id, r) in enumerate(ranked, 1):
                m = r['metrics']
                crown = '🏆' if rank == 1 and len(ranked) > 1 else ''
                diff = f" (+{m['best_mAP50_95'] - best_map:.4f})" if rank > 1 else " (基准)"
                f.write(f"| {crown} **{exp_id}** | {r['close_mosaic']} | {m['best_epoch']} | "
                       f"{m['best_mAP50']:.4f} | {m['best_mAP50_95']:.4f}{diff} | "
                       f"{m.get('best_precision', 'N/A')} | {m.get('best_recall', 'N/A')} | "
                       f"{m.get('total_time_hours', 'N/A')} |\n")
            f.write("\n")

            # 排名总结
            f.write("### 排名\n\n")
            f.write("| 排名 | 方案 | close_mosaic | mAP50-95 | 相对基准 |\n")
            f.write("|------|------|-------------|----------|----------|\n")
            for rank, (exp_id, r) in enumerate(ranked, 1):
                m = r['metrics']
                diff = m['best_mAP50_95'] - best_map
                diff_str = f"+{diff:.4f}" if diff >= 0 else f"{diff:.4f}"
                crown = '⭐⭐⭐⭐⭐' if rank == 1 else ('⭐⭐⭐⭐' if rank == 2 else ('⭐⭐⭐' if rank == 3 else '⭐⭐'))
                f.write(f"| {rank} | {crown} {exp_id} | {r['close_mosaic']} | {m['best_mAP50_95']:.4f} | {diff_str} |\n")
            f.write("\n")

        # Loss 对比
        f.write("## 3. Loss 对比\n\n")
        f.write("| 方案 | Final Box Loss | Final Cls Loss | Final DFL Loss | Final Angle Loss |\n")
        f.write("|------|---------------|---------------|---------------|------------------|\n")
        for exp_id in sorted_exps:
            r = results[exp_id]
            if r['metrics']:
                m = r['metrics']
                f.write(f"| {exp_id} | {m.get('final_box_loss', 'N/A')} | {m.get('final_cls_loss', 'N/A')} | "
                       f"{m.get('final_dfl_loss', 'N/A')} | {m.get('final_angle_loss', 'N/A')} |\n")
        f.write("\n")

        # 分析结论
        f.write("## 4. 分析结论\n\n")
        if available:
            ranked = sorted(available.items(), key=lambda x: x[1]['metrics']['best_mAP50_95'], reverse=True)
            best = ranked[0]
            worst = ranked[-1]
            f.write(f"### 最佳策略: {best[0]} (close_mosaic={best[1]['close_mosaic']})\n\n")
            f.write(f"- mAP50-95: **{best[1]['metrics']['best_mAP50_95']:.4f}**\n")
            f.write(f"- 最佳 epoch: {best[1]['metrics']['best_epoch']}\n\n")

            if len(ranked) > 1:
                gap = best[1]['metrics']['best_mAP50_95'] - worst[1]['metrics']['best_mAP50_95']
                f.write(f"### 策略间差距\n\n")
                f.write(f"- 最佳 vs 最差差距: **{gap:.4f} mAP50-95**\n")
                rank_str = ' > '.join(f"{eid}({rinfo['close_mosaic']})" for eid, rinfo in ranked)
                f.write(f"- 排名: {rank_str}\n\n")

            f.write("### 推荐训练策略\n\n")
            f.write(f"★★★★★ **推荐方案**: close_mosaic={best[1]['close_mosaic']}\n\n")
            f.write(f"理由: mAP50-95 最高 ({best[1]['metrics']['best_mAP50_95']:.4f})\n\n")
        else:
            f.write("⚠️ 暂无实验完成，分析待补充。\n\n")

        # 图表引用
        f.write("## 5. 图表\n\n")
        f.write("### 训练曲线\n\n")
        f.write("![训练曲线](mosaic_training_curves.png)\n\n")
        f.write("### 学习率曲线\n\n")
        f.write("![LR曲线](mosaic_lr_curves.png)\n\n")

        f.write("---\n\n")
        f.write(f"*报告由 benchmark_mosaic.py 自动生成*\n")

    print(f"  [Report] Benchmark report saved: {report_path}")


def generate_roadmap(output_dir):
    """生成下一阶段训练路线图"""
    roadmap_path = output_dir / '下一阶段训练路线图.md'

    recommendations = [
        {
            'title': '动态 Mosaic 概率衰减',
            'description': '不是突然关闭 Mosaic，而是按 epoch 线性/余弦衰减 Mosaic 概率。'
                           '例如：epoch 0-50 概率 1.0 → epoch 51-200 线性衰减至 0.1 → epoch 200+ 保持 0.1。'
                           '这比刚性 close_mosaic 更平滑，可能同时改善 Head 和 Tail 类别。',
            'priority': '⭐⭐⭐⭐⭐',
            'difficulty': '中',
            'gain': '预计 +0.01~0.02 mAP50-95',
            'action': '建议立即开展。需要修改 ultralytics/data/dataset.py 的 mosaic 概率计算逻辑。',
        },
        {
            'title': 'EMA 衰减因子调优',
            'description': '当前 EMA decay 使用默认值。对于长尾类别，更慢的 EMA 衰减可能帮助模型更好记忆稀有样本。'
                           '建议试验 decay=0.9997, 0.9998, 0.9999。',
            'priority': '⭐⭐⭐⭐',
            'difficulty': '低',
            'gain': '预计 +0.005~0.01 mAP50-95 (主要改善 Tail)',
            'action': '建议在 Mosaic 实验完成后开展。只需修改配置文件。',
        },
        {
            'title': 'Copy-Paste 小目标增强',
            'description': '遥感影像中小目标（Small Car, Van）占比较高。Copy-Paste 增强可将小目标实例复制到不同背景，'
                           '提高模型对稀有目标的泛化能力。ultralytics 框架已内置支持。',
            'priority': '⭐⭐⭐⭐',
            'difficulty': '低',
            'gain': '预计 +0.005~0.015 mAP50-95 (主要改善 Scale0)',
            'action': '建议立即开展。配置 `copy_paste=0.1~0.3` 试验。',
        },
        {
            'title': '多尺度训练范围扩大',
            'description': '当前 multi_scale=0.1 (imgsz 在 ±10% 范围内变化)。遥感小目标可能需要更大的尺度变化范围。'
                           '建议试验 multi_scale=0.3~0.5。',
            'priority': '⭐⭐⭐',
            'difficulty': '低',
            'gain': '预计 +0.003~0.01 mAP50-95',
            'action': '可在当前实验后快速验证。',
        },
        {
            'title': '长尾类别重采样',
            'description': '对 Tail 类别（Tractor, Truck Tractor）进行过采样或加权采样，'
                           '使每个 batch 中稀有类别的出现频率接近均匀分布。',
            'priority': '⭐⭐⭐',
            'difficulty': '中',
            'gain': '预计 +0.01~0.03 Tail AP (但可能轻微降低 Head)',
            'action': '需要自定义 Dataset sampler。建议在确定 Mosaic 策略后作为专项实验。',
        },
        {
            'title': '两阶段训练策略',
            'description': '第一阶段：长 Mosaic + 较高 lr (如 close_mosaic=300, lr0=0.01) 探索。'
                           '第二阶段：关闭 Mosaic + 低 lr (lr0=0.001) 精调收敛。'
                           '类似 Exp6 的续训策略但更加系统化。',
            'priority': '⭐⭐⭐',
            'difficulty': '中',
            'gain': '预计 +0.01~0.02 mAP50-95',
            'action': '建议作为 Mosaic 实验的后续优化。',
        },
        {
            'title': 'YOLO11n → YOLO11s 模型容量升级',
            'description': '当前 yolo11n (2.7M params) 已接近容量上限 (mAP50-95 ~0.418)。'
                           '升级到 yolo11s (9.7M params, 3.6×) 预期带来显著提升。'
                           'batch size 需要降低（8卡约 batch=48~64）。',
            'priority': '⭐⭐⭐',
            'difficulty': '低',
            'gain': '预计 +0.02~0.04 mAP50-95',
            'action': '在确定最优训练策略后，用该策略训练 yolo11s 验证容量上限。',
        },
        {
            'title': 'Rotated NMS 参数调优',
            'description': 'OBB 的 ProbIoU + NMS 参数 (iou=0.7) 可能不是最优。'
                           '建议在验证集上对 iou 阈值 (0.5~0.9) 进行网格搜索。',
            'priority': '⭐⭐',
            'difficulty': '低',
            'gain': '预计 +0.002~0.005 mAP50-95',
            'action': '低优先级，可在比赛最终提交前快速调优。',
        },
    ]

    with open(roadmap_path, 'w', encoding='utf-8') as f:
        f.write("# 下一阶段训练路线图\n\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("基于 Mosaic 调度策略实验结论和本项目特点（遥感、小目标、OBB、长尾类别、密集场景），"
                "以下是按收益排序的后续实验建议。\n\n")
        f.write("---\n\n")

        f.write("## 优先级总览\n\n")
        f.write("| 优先级 | 方向 | 实现难度 | 预计收益 (mAP50-95) | 建议 |\n")
        f.write("|--------|------|----------|---------------------|------|\n")
        for i, rec in enumerate(recommendations):
            f.write(f"| {i+1} | {rec['priority']} {rec['title']} | {rec['difficulty']} | {rec['gain']} | {rec['action'][:60]}... |\n")
        f.write("\n---\n\n")

        for i, rec in enumerate(recommendations, 1):
            f.write(f"## {i}. {rec['title']}\n\n")
            f.write(f"| 属性 | 值 |\n")
            f.write(f"|------|-----|\n")
            f.write(f"| 优先级 | {rec['priority']} |\n")
            f.write(f"| 实现难度 | {rec['difficulty']} |\n")
            f.write(f"| 预计收益 | {rec['gain']} |\n")
            f.write(f"| 是否建议立即开展 | {rec['action'][:50]}... |\n\n")
            f.write(f"**描述**: {rec['description']}\n\n")
            f.write(f"**建议**: {rec['action']}\n\n")
            f.write("---\n\n")

        f.write(f"\n*报告由 benchmark_mosaic.py 自动生成*\n")

    print(f"  [Report] Roadmap saved: {roadmap_path}")


def main():
    parser = argparse.ArgumentParser(description='Mosaic 调度策略对比分析')
    parser.add_argument('--project-root', type=str, default=None,
                        help='项目根目录 (默认: 脚本所在目录的父目录)')
    parser.add_argument('--exp-dirs', type=str, nargs='*', default=None,
                        help='手动指定实验目录')
    parser.add_argument('--output', type=str, default=None,
                        help='输出目录 (默认: runs/mosaic_benchmark)')
    parser.add_argument('--skip-plots', action='store_true', help='跳过绘图')
    args = parser.parse_args()

    # 确定项目根目录
    if args.project_root:
        project_root = Path(args.project_root)
    else:
        project_root = Path(__file__).parent.parent

    # 输出目录
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = project_root / 'runs' / 'mosaic_benchmark'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  Mosaic Strategy Benchmark Analysis")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Project: {project_root}")
    print(f"  Output: {output_dir}")
    print("=" * 70)

    # 收集实验结果
    results = {}
    for exp_id, exp_info in MOSAIC_EXPERIMENTS.items():
        run_dir = find_run_dir(project_root, exp_info['name'])
        print(f"\n[Exp] {exp_id}: {exp_info['name']}")

        if run_dir is None:
            print(f"   WARN: experiment directory not found")
            results[exp_id] = {
                **exp_info,
                'run_dir': None,
                'df': None,
                'args': None,
                'metrics': None,
            }
            continue

        print(f"   Dir: {run_dir}")

        # 加载 results.csv
        df = load_results_csv(run_dir)
        if df is not None:
            print(f"   OK: results.csv loaded ({len(df)} epochs)")
        else:
            print(f"   WARN: results.csv not found (training may be in progress)")

        # 提取最佳指标
        metrics = extract_best_metrics(df)
        if metrics:
            print(f"   Best mAP50-95: {metrics['best_mAP50_95']:.4f} @ epoch {metrics['best_epoch']}")
        else:
            print(f"   WARN: cannot extract metrics")

        results[exp_id] = {
            **exp_info,
            'run_dir': run_dir,
            'df': df,
            'args': load_args_yaml(run_dir),
            'metrics': metrics,
        }

    # 打印汇总表
    print("\n" + "=" * 70)
    print("  Metrics Summary")
    print("=" * 70)
    available = {k: v for k, v in results.items() if v['metrics']}
    if available:
        ranked = sorted(available.items(), key=lambda x: x[1]['metrics']['best_mAP50_95'], reverse=True)
        print(f"{'Rank':<5} {'Exp':<6} {'close_mosaic':<14} {'Best Epoch':<12} {'mAP50-95':<14} {'mAP50':<14}")
        print("-" * 70)
        for rank, (exp_id, r) in enumerate(ranked, 1):
            m = r['metrics']
            print(f"{rank:<5} {exp_id:<6} {r['close_mosaic']:<14} {m['best_epoch']:<12} {m['best_mAP50_95']:<14.4f} {m.get('best_mAP50', 0):<14.4f}")
    else:
        print("  WARN: No completed experiments yet")

    # 绘图
    if not args.skip_plots:
        print("\n[Plot] Generating charts...")
        plot_mAP_curves(results, output_dir)
        plot_lr_curves(results, output_dir)

    # 生成报告
    print("\n[Report] Generating reports...")
    generate_markdown_report(results, output_dir)
    generate_roadmap(output_dir)

    print("\n" + "=" * 70)
    print("  Analysis complete!")
    print(f"  Output directory: {output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
