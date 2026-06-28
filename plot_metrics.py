#!/usr/bin/env python3
"""
训练指标可视化脚本
用法:
    python plot_metrics.py                          # 自动找最新的 results.csv
    python plot_metrics.py runs/xxx/results.csv     # 指定文件
    python plot_metrics.py --watch 60               # 每60秒自动刷新
"""

import sys
import time
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 无 GUI 后端
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

plt.rcParams['axes.unicode_minus'] = False


def find_latest_csv() -> Path | None:
    """自动查找最新的 results.csv"""
    runs_dir = Path(__file__).parent / 'runs'
    if not runs_dir.exists():
        return None
    csvs = sorted(runs_dir.rglob('results.csv'), key=lambda p: p.stat().st_mtime, reverse=True)
    return csvs[0] if csvs else None


def load_data(csv_path: Path) -> pd.DataFrame:
    """加载并清洗 results.csv"""
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    return df


def plot_all(df: pd.DataFrame, title: str = '', save_path: str = 'training_metrics.png'):
    """绘制完整的训练指标面板"""
    epochs = df['epoch'].values
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    fig.suptitle(title or 'YOLO11-OBB Training Metrics', fontsize=16, fontweight='bold', y=0.98)

    # ── 配色 ──
    c_train = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    c_val = ['#17becf', '#e377c2', '#8c564b', '#bcbd22']
    c_metrics = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # ── Row 1: Train Loss ──
    ax = axes[0, 0]
    train_losses = ['train/box_loss', 'train/cls_loss', 'train/dfl_loss', 'train/angle_loss']
    for col, color in zip(train_losses, c_train):
        if col in df.columns:
            ax.plot(epochs, df[col], color=color, alpha=0.7, linewidth=0.8, label=col.split('/')[1])
    ax.set_title('Train Loss')
    ax.set_ylabel('Loss')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(epochs) * 1.02)

    # ── Row 1, Col 2: Val Loss ──
    ax = axes[0, 1]
    val_losses = ['val/box_loss', 'val/cls_loss', 'val/dfl_loss', 'val/angle_loss']
    for col, color in zip(val_losses, c_val):
        if col in df.columns:
            ax.plot(epochs, df[col], color=color, alpha=0.7, linewidth=0.8, label=col.split('/')[1])
    ax.set_title('Val Loss')
    ax.set_ylabel('Loss')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(epochs) * 1.02)

    # ── Row 1, Col 3: DFL Loss ──
    ax = axes[0, 2]
    if 'train/dfl_loss' in df.columns:
        ax.plot(epochs, df['train/dfl_loss'], color='#1f77b4', alpha=0.6, linewidth=0.8, label='train')
    if 'val/dfl_loss' in df.columns:
        ax.plot(epochs, df['val/dfl_loss'], color='#ff7f0e', alpha=0.6, linewidth=0.8, label='val')
    ax.set_title('DFL Loss')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(epochs) * 1.02)

    # ── Row 2: mAP & Precision/Recall ──
    ax = axes[1, 0]
    map_cols = ['metrics/mAP50(B)', 'metrics/mAP50-95(B)']
    for col, color in zip(map_cols, c_metrics[:2]):
        if col in df.columns:
            label = 'mAP50' if 'mAP50(B)' in col else 'mAP50-95'
            ax.plot(epochs, df[col], color=color, linewidth=1.2, label=label)
            max_idx = df[col].idxmax()
            max_val = df[col].max()
            ax.annotate(f'{max_val:.4f}', (epochs[max_idx], max_val),
                        textcoords="offset points", xytext=(0, 10), fontsize=8,
                        color=color, ha='center',
                        arrowprops=dict(arrowstyle='->', color=color, alpha=0.5))
    ax.set_title('Val mAP')
    ax.set_ylabel('mAP')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(epochs) * 1.02)

    ax = axes[1, 1]
    pr_cols = ['metrics/precision(B)', 'metrics/recall(B)']
    for col, color in zip(pr_cols, c_metrics[2:]):
        if col in df.columns:
            ax.plot(epochs, df[col], color=color, linewidth=1.0, label=col.split('/')[1].replace('(B)', ''))
    ax.set_title('Precision & Recall')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(epochs) * 1.02)

    ax = axes[1, 2]
    if 'train/box_loss' in df.columns and 'val/box_loss' in df.columns:
        ax.plot(epochs, df['train/box_loss'], color='#1f77b4', alpha=0.6, linewidth=0.8, label='train box')
        ax.plot(epochs, df['val/box_loss'], color='#ff7f0e', alpha=0.6, linewidth=0.8, label='val box')
    ax.set_title('Box Loss (IoU)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(epochs) * 1.02)

    # ── Row 3: LR & Time ──
    ax = axes[2, 0]
    lr_cols = [c for c in df.columns if c.startswith('lr/')]
    colors_lr = plt.cm.viridis(np.linspace(0, 1, max(len(lr_cols), 1)))
    for col, color in zip(lr_cols, colors_lr):
        ax.plot(epochs, df[col], color=color, linewidth=0.8, label=col, alpha=0.7)
    ax.set_title('Learning Rate Schedule')
    ax.set_ylabel('LR')
    ax.set_xlabel('Epoch')
    ax.legend(fontsize=6, loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(epochs) * 1.02)

    ax = axes[2, 1]
    if 'time' in df.columns:
        time_hours = df['time'] / 3600.0
        ax.plot(epochs, time_hours, color='#555555', linewidth=1.0)
        ax.set_title('Cumulative Training Time')
        ax.set_ylabel('Time (hours)')
        ax.set_xlabel('Epoch')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max(epochs) * 1.02)

    ax = axes[2, 2]
    ax.axis('off')
    summary_lines = [
        f"Experiment: {Path(save_path).stem}",
        f"Total Epochs: {len(df)}",
        f"",
        f"-- Best Metrics --",
    ]
    for col in ['metrics/mAP50(B)', 'metrics/mAP50-95(B)', 'metrics/precision(B)', 'metrics/recall(B)']:
        if col in df.columns:
            name = col.split('/')[1].replace('(B)', '')
            best = df[col].max()
            best_ep = df.loc[df[col].idxmax(), 'epoch']
            summary_lines.append(f"{name}: {best:.4f} @ ep {int(best_ep)}")

    summary_lines += [
        f"",
        f"-- Final Loss --",
    ]
    for col in ['train/box_loss', 'train/cls_loss', 'train/dfl_loss', 'train/angle_loss']:
        if col in df.columns:
            name = col.split('/')[1]
            summary_lines.append(f"{name}: {df[col].iloc[-1]:.4f}")

    summary_lines += [
        f"",
        f"-- Training Info --",
        f"Data points: {len(df)}",
    ]
    if 'time' in df.columns:
        summary_lines.append(f"Total time: {df['time'].iloc[-1]/3600:.1f}h")

    ax.text(0.05, 0.95, '\n'.join(summary_lines), transform=ax.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Chart saved: {save_path}")
    return fig


def main():
    watch_mode = False
    watch_interval = 60
    csv_path = None

    # 解析参数
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == '--watch':
            watch_mode = True
            if i + 1 < len(args) and args[i + 1].isdigit():
                watch_interval = int(args[i + 1])
                i += 1
        elif args[i].endswith('.csv'):
            csv_path = Path(args[i])
        i += 1

    if csv_path is None:
        csv_path = find_latest_csv()

    if csv_path is None or not csv_path.exists():
        print("❌ results.csv not found, please specify path")
        sys.exit(1)

    print(f"📂 Data source: {csv_path}")

    if watch_mode:
        print(f"👀 Watch mode: refresh every {watch_interval}s (Ctrl+C to exit)")
        try:
            while True:
                df = load_data(csv_path)
                plot_all(df, title=f'Training Progress: Epoch {len(df)}',
                         save_path=str(csv_path.parent / 'training_metrics.png'))
                print(f"   Updated: {time.strftime('%H:%M:%S')}, Epochs: {len(df)}")
                time.sleep(watch_interval)
        except KeyboardInterrupt:
            print("\n👋 Watch mode stopped")
    else:
        df = load_data(csv_path)
        save_path = str(csv_path.parent / 'training_metrics.png')
        plot_all(df, title=f'Training Metrics (Epoch 1-{len(df)})', save_path=save_path)


if __name__ == '__main__':
    main()
