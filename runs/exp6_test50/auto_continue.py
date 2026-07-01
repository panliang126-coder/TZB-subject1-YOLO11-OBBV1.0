#!/usr/bin/env python3
"""
Exp6 续训冲分 - 自动化监控与决策脚本
每25 epoch 评估一次，自动判断是否继续训练。

用法:
  python runs/exp6_test50/auto_continue.py --checkpoint runs/exp6_ep75/weights/best.pt --label ep75
"""

import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.parent
EVAL_SCRIPT = Path(__file__).parent / 'eval_checkpoint.py'

# ── 阈值配置 ──
PLATEAU_THRESHOLD = 0.002     # mAP50-95 提升 < 0.002 视为平台期
PLATEAU_CONSECUTIVE = 3       # 连续 N 次平台期则建议停止
MIN_TAIL_IMPROVEMENT = 0.001  # Tail AP 最小提升阈值


def load_history(history_path):
    """加载历史评估记录"""
    if history_path.exists():
        with open(history_path) as f:
            return json.load(f)
    return []


def save_history(history_path, history):
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def check_plateau(history, current):
    """判断是否进入平台期"""
    if len(history) < 2:
        return False, 0

    # 取最近 3 次评估的 test mAP50-95
    recent = [h['test']['mAP50-95'] for h in history[-3:]]
    recent.append(current['test']['mAP50-95'])

    # 计算连续增长
    plateau_count = 0
    for i in range(1, len(recent)):
        improvement = recent[i] - recent[i-1]
        if improvement < PLATEAU_THRESHOLD:
            plateau_count += 1
        else:
            plateau_count = 0

    return plateau_count >= PLATEAU_CONSECUTIVE, plateau_count


def check_tail_rising(history, current):
    """判断 Tail AP 是否仍在持续提升"""
    if len(history) < 1:
        return True, 0.0

    prev_tail = history[-1]['test']['Tail_AP']
    curr_tail = current['test']['Tail_AP']
    improvement = curr_tail - prev_tail

    return improvement > MIN_TAIL_IMPROVEMENT, improvement


def generate_report(history, output_path):
    """生成 Markdown 趋势报告"""
    lines = []
    lines.append("# Exp6 续训冲分 - 趋势报告")
    lines.append(f"\n> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"> 总评估点数: {len(history)}")
    lines.append("")

    # 整体趋势表
    lines.append("## 整体趋势 (Test 集)")
    lines.append("")
    lines.append("| Checkpoint | mAP50 | mAP50-95 | Precision | Recall | Head AP | Middle AP | Tail AP |")
    lines.append("|-----------|-------|----------|-----------|--------|---------|-----------|---------|")
    for h in history:
        t = h['test']
        lines.append(f"| {h['label']} | {t['mAP50']:.4f} | {t['mAP50-95']:.4f} | "
                     f"{t['precision']:.4f} | {t['recall']:.4f} | "
                     f"{t.get('Head_AP', 0):.4f} | {t.get('Middle_AP', 0):.4f} | {t.get('Tail_AP', 0):.4f} |")
    lines.append("")

    # 最新 vs 起点对比
    if len(history) >= 2:
        first = history[0]
        last = history[-1]
        delta_map = last['test']['mAP50-95'] - first['test']['mAP50-95']
        delta_tail = last['test']['Tail_AP'] - first['test']['Tail_AP']
        delta_head = last['test']['Head_AP'] - first['test']['Head_AP']
        lines.append("## 累计变化 (起点 → 最新)")
        lines.append("")
        lines.append(f"| 指标 | {first['label']} | {last['label']} | Δ |")
        lines.append(f"|------|------|------|------|")
        lines.append(f"| mAP50-95 | {first['test']['mAP50-95']:.4f} | {last['test']['mAP50-95']:.4f} | {delta_map:+.4f} |")
        lines.append(f"| Head AP | {first['test']['Head_AP']:.4f} | {last['test']['Head_AP']:.4f} | {delta_head:+.4f} |")
        lines.append(f"| Middle AP | {first['test']['Middle_AP']:.4f} | {last['test']['Middle_AP']:.4f} | {last['test']['Middle_AP'] - first['test']['Middle_AP']:+.4f} |")
        lines.append(f"| Tail AP | {first['test']['Tail_AP']:.4f} | {last['test']['Tail_AP']:.4f} | {delta_tail:+.4f} |")
        lines.append("")

    # 各类别详细变化
    if len(history) >= 2:
        lines.append("## 各类别变化 (Test 集)")
        lines.append("")
        first_classes = history[0]['test']['per_class_ap50_95']
        last_classes = history[-1]['test']['per_class_ap50_95']
        all_classes = sorted(set(list(first_classes.keys()) + list(last_classes.keys())))

        lines.append("| 类别 | 分组 | 起点 | 最新 | Δ | 趋势 |")
        lines.append("|------|------|------|------|------|------|")
        for cls in all_classes:
            f_ap = first_classes.get(cls, 0)
            l_ap = last_classes.get(cls, 0)
            delta = l_ap - f_ap
            trend = '📈' if delta > 0.005 else ('➡️' if abs(delta) <= 0.005 else '📉')
            head_mid_tail = 'Head' if cls in {'Small Car', 'Van'} else ('Middle' if cls in {'Dump Truck', 'Cargo Truck', 'other-vehicle'} else 'Tail')
            lines.append(f"| {cls} | {head_mid_tail} | {f_ap:.4f} | {l_ap:.4f} | {delta:+.4f} | {trend} |")
        lines.append("")

    # 决策建议
    lines.append("## 决策建议")
    lines.append("")
    if len(history) >= 2:
        last_delta = history[-1]['test']['mAP50-95'] - history[-2]['test']['mAP50-95']
        is_plateau, plateau_n = check_plateau(history[:-1], history[-1])
        tail_rising, tail_delta = check_tail_rising(history[:-1], history[-1])

        lines.append(f"- **最新提升**: {last_delta:+.4f} mAP50-95")
        lines.append(f"- **平台期检测**: {'⚠️ 已进入平台期' if is_plateau else '✅ 仍在提升'} (连续 {plateau_n} 次 < {PLATEAU_THRESHOLD})")
        lines.append(f"- **Tail 趋势**: {'📈 仍在提升' if tail_rising else '➡️ 趋平'} (Δ = {tail_delta:+.4f})")

        if is_plateau and not tail_rising:
            lines.append(f"\n### 🔴 建议: 停止训练")
            lines.append(f"\n整体 mAP 和 Tail AP 均已趋平，继续训练 ROI 较低。")
        elif tail_rising:
            lines.append(f"\n### 🟢 建议: 继续训练")
            lines.append(f"\nTail 类仍在提升（+{tail_delta:.4f}），模型仍在学习长尾类别。")
        else:
            lines.append(f"\n### 🟡 建议: 谨慎继续")
            lines.append(f"\n整体趋平但 Tail 仍有微弱提升，可再跑 1-2 个 chunk 观察。")
    else:
        lines.append("数据点不足，需至少 2 次评估才能给出建议。")

    lines.append("")

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    return str(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint 路径')
    parser.add_argument('--label', type=str, required=True, help='Checkpoint 标签 (如 ep75)')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--history-dir', type=str, default=None)
    args = parser.parse_args()

    history_dir = Path(args.history_dir) if args.history_dir else Path(__file__).parent
    history_path = history_dir / 'eval_history.json'

    # 运行评估
    print(f"\n🚀 评估 {args.label}...")
    eval_output = history_dir / f'eval_{args.label}.json'

    cmd = [
        sys.executable, str(EVAL_SCRIPT),
        '--weights', args.checkpoint,
        '--fold', str(args.fold),
        '--label', args.label,
        '--output', str(eval_output),
    ]
    subprocess.run(cmd, check=True)

    # 加载评估结果
    with open(eval_output) as f:
        current = json.load(f)

    # 添加分组指标到 test/val
    from eval_checkpoint import compute_group_metrics
    for split in ['val', 'test']:
        if split in current:
            current[split].update(compute_group_metrics(current[split]))

    # 加载历史
    history = load_history(history_path)

    # 添加当前结果 (含 label)
    entry = {
        'label': args.label,
        'checkpoint': str(args.checkpoint),
        'timestamp': datetime.now().isoformat(),
        'val': current.get('val', {}),
        'test': current.get('test', {}),
    }
    history.append(entry)
    save_history(history_path, history)

    # 判断
    is_plateau, plateau_n = check_plateau(history[:-1], entry)
    tail_rising, tail_delta = check_tail_rising(history[:-1], entry)

    last_delta = 0
    if len(history) >= 2:
        last_delta = entry['test']['mAP50-95'] - history[-2]['test']['mAP50-95']

    print(f"\n{'='*50}")
    print(f"📊 {args.label} 评估完成")
    print(f"   Test mAP50-95: {entry['test']['mAP50-95']:.4f} (Δ {last_delta:+.4f})")
    print(f"   Tail AP:       {entry['test'].get('Tail_AP', 0):.4f} (Δ {tail_delta:+.4f})")
    print(f"   平台期计数:     {plateau_n}/{PLATEAU_CONSECUTIVE}")
    print(f"{'='*50}")

    # 生成报告
    report_path = history_dir / '续训趋势报告.md'
    generate_report(history, report_path)
    print(f"\n📄 趋势报告已更新: {report_path}")

    # 决策
    decision = 'CONTINUE'
    if is_plateau and not tail_rising:
        decision = 'STOP'
    elif tail_rising:
        decision = 'CONTINUE'
    elif is_plateau:
        decision = 'CAUTIOUS'

    print(f"\n🎯 决策: {decision}")
    return decision


if __name__ == '__main__':
    main()
