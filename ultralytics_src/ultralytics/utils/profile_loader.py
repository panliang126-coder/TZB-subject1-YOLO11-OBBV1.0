#!/usr/bin/env python3
"""
Ultralytics 训练数据加载 Profiling 模块
==========================================
测量数据加载链路每个阶段的耗时，定位 CPU/GPU 瓶颈。

用法:
    from ultralytics.utils.profile_loader import DataLoadProfiler, enable_profiling
    enable_profiling(trainer)  # 在 train() 之前调用

配置开关 (args.yaml):
    profile_loader: true   # 开启 profiling
    profile_workers: true  # 开启 worker 级别统计
"""

from __future__ import annotations

import csv
import time
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ultralytics.utils import LOGGER, RANK

# ============================================================
# 全局 Profiler 实例 (单例)
# ============================================================

_PROFILER = None  # 全局 profiler 实例


def get_profiler() -> "DataLoadProfiler | None":
    """获取全局 profiler 实例."""
    return _PROFILER


class DataLoadProfiler:
    """训练数据加载全链路 Profiler.

    统计维度:
    - per-batch: 每个 stage 的耗时 (data_load, mosaic, affine, hsv, flip, format, forward, loss, backward, optimizer)
    - per-epoch: 聚合统计 (mean, std, min, max, total)
    - GPU idle/busy: 测量 GPU 等待 vs 计算的时间比例

    输出:
    - 每个 epoch 末尾打印 profiling 报告
    - 训练结束后保存 profiling_report.csv
    """

    # 统计的阶段定义: (key, display_name, is_gpu)
    STAGES = [
        # CPU 阶段 (数据加载 + 增强)
        ("data_load", "Data Loading", False),
        ("mosaic", "Mosaic", False),
        ("affine", "RandomAffine", False),
        ("copypaste", "CopyPaste", False),
        ("mixup", "MixUp", False),
        ("hsv", "HSV", False),
        ("flip", "Flip", False),
        ("format", "Format", False),
        ("collate", "Collate+Transfer", False),
        # GPU 阶段
        ("forward", "Forward", True),
        ("loss", "Loss", True),
        ("backward", "Backward", True),
        ("optimizer", "Optimizer", True),
        # 等待
        ("ddp_wait", "DDP Wait", True),
    ]

    def __init__(self, enabled: bool = True, save_dir: str | Path = ""):
        """初始化 Profiler.

        Args:
            enabled: 是否启用 profiling (关闭时所有操作退化为 no-op)
            save_dir: 保存 profiling 报告的目录
        """
        global _PROFILER
        _PROFILER = self

        self.enabled = enabled
        self.save_dir = Path(save_dir) if save_dir else None

        # per-batch 计时 (在 DataLoader worker 中填充, 主进程聚合)
        self.batch_timings: dict[str, list[float]] = defaultdict(list)

        # epoch 级别聚合
        self.epoch_stats: dict[int, dict[str, dict[str, float]]] = {}

        # GPU 事件 (使用 CUDA Event 精确计时)
        self.gpu_events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = defaultdict(list)

        # 上一次 batch 结束时间 (用于测量 batch 间空闲)
        self._last_batch_end: float | None = None
        self._batch_idle_times: list[float] = []

        # worker 统计 (由 DataLoader worker 进程汇报)
        self.worker_stats: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

        # epoch 计数器
        self.current_epoch: int = 0
        self.current_batch: int = 0

        # 结果存储
        self.epoch_report_lines: list[str] = []

    # ============================================================
    # Batch 级别计时
    # ============================================================

    def start_batch(self):
        """记录 batch 开始时间 (在主进程训练循环中调用)."""
        if not self.enabled:
            return
        # 测量 batch 间 idle
        now = time.perf_counter()
        if self._last_batch_end is not None:
            idle = now - self._last_batch_end
            self._batch_idle_times.append(idle)
        self._batch_start = now

    def end_batch(self):
        """记录 batch 结束时间."""
        if not self.enabled:
            return
        self._last_batch_end = time.perf_counter()

    def record_stage(self, name: str, duration_ms: float):
        """记录某个阶段的耗时 (毫秒).

        Args:
            name: 阶段名称 (需在 STAGES 中定义)
            duration_ms: 耗时 (毫秒)
        """
        if not self.enabled:
            return
        self.batch_timings[name].append(duration_ms)

    def record_gpu_stage(self, name: str, start_event: torch.cuda.Event, end_event: torch.cuda.Event):
        """记录 GPU 阶段耗时 (使用 CUDA Event 精确测量).

        Args:
            name: 阶段名称
            start_event: CUDA start event (已 record)
            end_event: CUDA end event (已 record)
        """
        if not self.enabled:
            return
        self.gpu_events[name].append((start_event, end_event))

    # ============================================================
    # 上下文管理器 (用于自动计时)
    # ============================================================

    class StageTimer:
        """阶段计时器上下文管理器."""

        def __init__(self, profiler: "DataLoadProfiler", stage_name: str, use_cuda: bool = False):
            self.profiler = profiler
            self.stage_name = stage_name
            self.use_cuda = use_cuda
            self.start_time: float = 0.0
            self.start_event: torch.cuda.Event | None = None
            self.end_event: torch.cuda.Event | None = None

        def __enter__(self):
            if not self.profiler.enabled:
                return self
            if self.use_cuda:
                self.start_event = torch.cuda.Event(enable_timing=True)
                self.end_event = torch.cuda.Event(enable_timing=True)
                self.start_event.record()
            else:
                self.start_time = time.perf_counter()
            return self

        def __exit__(self, *args):
            if not self.profiler.enabled:
                return
            if self.use_cuda and self.start_event is not None and self.end_event is not None:
                self.end_event.record()
                self.profiler.record_gpu_stage(self.stage_name, self.start_event, self.end_event)
            else:
                duration_ms = (time.perf_counter() - self.start_time) * 1000.0
                self.profiler.record_stage(self.stage_name, duration_ms)

    def timer(self, stage_name: str, use_cuda: bool = False) -> "DataLoadProfiler.StageTimer":
        """创建一个阶段计时器.

        Args:
            stage_name: 阶段名称
            use_cuda: 是否使用 CUDA Event 计时

        Returns:
            StageTimer 上下文管理器

        Usage:
            with profiler.timer("mosaic"):
                mosaic(img, labels)
        """
        return self.StageTimer(self, stage_name, use_cuda)

    # ============================================================
    # Epoch 级别聚合和输出
    # ============================================================

    def end_epoch(self, epoch: int):
        """Epoch 结束, 聚合统计并打印报告.

        Args:
            epoch: 当前 epoch (0-indexed)
        """
        if not self.enabled:
            return

        self.current_epoch = epoch

        # 计算 GPU events 的实际耗时
        self._resolve_gpu_events()

        # 构建 epoch 统计
        stats: dict[str, dict[str, float]] = {}
        for stage_key, display_name, is_gpu in self.STAGES:
            values = self.batch_timings.get(stage_key, [])
            if not values:
                continue
            arr = np.array(values)
            stats[display_name] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "total": float(np.sum(arr)),
                "count": len(arr),
            }

        self.epoch_stats[epoch] = stats

        # 打印报告
        self._print_epoch_report(epoch, stats)

        # 保存到 CSV
        self._save_epoch_csv(epoch, stats)

        # 清空 per-batch 数据
        self.batch_timings.clear()
        self.gpu_events.clear()
        self._last_batch_end = None
        self._batch_idle_times.clear()

    def _resolve_gpu_events(self):
        """解析所有 GPU event 的时间差."""
        for stage_name, event_pairs in self.gpu_events.items():
            for start_ev, end_ev in event_pairs:
                # synchronize 后取时间
                elapsed_ms = start_ev.elapsed_time(end_ev)
                self.batch_timings[stage_name].append(elapsed_ms)
        self.gpu_events.clear()

    def _print_epoch_report(self, epoch: int, stats: dict[str, dict[str, float]]):
        """打印 epoch profiling 报告."""
        if RANK not in (-1, 0):
            return

        lines = []
        lines.append("")
        lines.append("=" * 70)
        lines.append(f"  📊 Epoch {epoch + 1} Profiling Report")
        lines.append("=" * 70)

        # CPU 阶段
        cpu_total = 0.0
        lines.append(f"  {'─' * 60}")
        lines.append(f"  {'Stage':<25s} {'Mean(ms)':>10s} {'Std(ms)':>10s} {'Min/Max(ms)':>18s}")
        lines.append(f"  {'─' * 60}")

        cpu_stages = ["Data Loading", "Mosaic", "RandomAffine", "CopyPaste", "MixUp",
                       "HSV", "Flip", "Format", "Collate+Transfer"]
        for name in cpu_stages:
            if name in stats:
                s = stats[name]
                lines.append(f"  {name:<25s} {s['mean']:>10.2f} {s['std']:>10.2f} "
                             f"{s['min']:>8.1f}/{s['max']:<8.1f}")
                cpu_total += s["mean"]

        # GPU 阶段
        gpu_total = 0.0
        lines.append(f"  {'─' * 60}")
        lines.append(f"  {'─ GPU Stages ─':─^60}")
        lines.append(f"  {'─' * 60}")

        gpu_stages = ["Forward", "Loss", "Backward", "Optimizer", "DDP Wait"]
        for name in gpu_stages:
            if name in stats:
                s = stats[name]
                lines.append(f"  {name:<25s} {s['mean']:>10.2f} {s['std']:>10.2f} "
                             f"{s['min']:>8.1f}/{s['max']:<8.1f}")
                gpu_total += s["mean"]

        # 汇总
        total = cpu_total + gpu_total
        lines.append(f"  {'─' * 60}")
        if total > 0:
            cpu_pct = (cpu_total / total) * 100
            gpu_pct = (gpu_total / total) * 100
            lines.append(f"  Total CPU: {cpu_total:>8.1f} ms/batch ({cpu_pct:.0f}%)")
            lines.append(f"  Total GPU: {gpu_total:>8.1f} ms/batch ({gpu_pct:.0f}%)")
            lines.append(f"  Total:     {total:>8.1f} ms/batch")

            # GPU compute ratio: GPU时间 / (GPU时间 + batch间idle)
            if self._batch_idle_times:
                avg_idle = np.mean(self._batch_idle_times) * 1000  # 转为 ms
                lines.append(f"  Avg Batch Idle: {avg_idle:>6.1f} ms/batch")
                if gpu_total > 0:
                    gpu_ratio = gpu_total / (gpu_total + avg_idle) * 100
                    lines.append(f"  GPU Busy Ratio: {gpu_ratio:>6.1f}%")
                    if gpu_ratio < 80:
                        lines.append(f"  ⚠️ GPU Utilization < 80% - CPU 数据加载可能为瓶颈!")

        # 诊断建议
        lines.append(f"  {'─' * 60}")
        lines.extend(self._diagnose(stats))

        lines.append("=" * 70)

        report = "\n".join(lines)
        self.epoch_report_lines.append(report)
        print(report)

    def _diagnose(self, stats: dict[str, dict[str, float]]) -> list[str]:
        """根据 profiling 数据自动诊断瓶颈."""
        hints = []

        # 检查 Mosaic 是否过重
        if "Mosaic" in stats and stats["Mosaic"]["mean"] > 50:
            hints.append(f"  💡 Mosaic 耗时 {stats['Mosaic']['mean']:.0f}ms/batch (较重)")
            hints.append(f"     建议: 降低 mosaic 概率 或 增大 close_mosaic")

        # 检查 Data Loading
        if "Data Loading" in stats and stats["Data Loading"]["mean"] > 20:
            hints.append(f"  💡 Data Loading 耗时 {stats['Data Loading']['mean']:.0f}ms/batch")
            hints.append(f"     建议: cache=disk 或 cache=ram")

        # 检查 CPU vs GPU 占比
        cpu_time = sum(s["mean"] for s in [stats.get(n, {"mean": 0}) for n in
                        ["Data Loading", "Mosaic", "RandomAffine"]])
        gpu_time = sum(s["mean"] for s in [stats.get(n, {"mean": 0}) for n in
                        ["Forward", "Loss", "Backward", "Optimizer"]])
        if cpu_time > gpu_time * 2:
            hints.append(f"  ⚠️ CPU 耗时 ({cpu_time:.0f}ms) >> GPU 耗时 ({gpu_time:.0f}ms)")
            hints.append(f"     数据预处理是主要瓶颈")

        return hints

    def _save_epoch_csv(self, epoch: int, stats: dict[str, dict[str, float]]):
        """保存 epoch profiling 数据到 CSV."""
        if not self.save_dir:
            return
        csv_path = self.save_dir / "profiling_report.csv"
        write_header = not csv_path.exists()
        with open(csv_path, "a", newline="") as f:
            fields = ["epoch"] + [name for _, name, _ in self.STAGES]
            writer = csv.DictWriter(f, fieldnames=fields)
            if write_header:
                writer.writeheader()
            row = {"epoch": epoch + 1}
            for _, name, _ in self.STAGES:
                row[name] = f"{stats[name]['mean']:.2f}" if name in stats else "0"
            writer.writerow(row)

    def save_final_report(self):
        """保存最终 optimization 建议到 Markdown 文件."""
        if not self.enabled or not self.save_dir:
            return

        report_path = self.save_dir / "Optimization_Report.md"
        lines = []
        lines.append("# YOLO11-OBB 训练优化报告\n")
        lines.append(f"## 训练配置\n")
        lines.append(f"- Epochs profiled: {len(self.epoch_stats)}")
        lines.append(f"- GPU count: {torch.cuda.device_count()}")
        lines.append(f"- Workers: see args.yaml\n")

        lines.append("## Per-Epoch Profiling 数据\n")
        for epoch, stats in sorted(self.epoch_stats.items()):
            lines.append(f"### Epoch {epoch + 1}\n")
            for _, name, _ in self.STAGES:
                if name in stats:
                    s = stats[name]
                    lines.append(f"- **{name}**: {s['mean']:.1f} ± {s['std']:.1f} ms")
            lines.append("")

        lines.append("## 瓶颈分析\n")

        # 聚合所有 epoch 的数据
        all_stats: dict[str, list[float]] = defaultdict(list)
        for stats in self.epoch_stats.values():
            for name, s in stats.items():
                all_stats[name].append(s["mean"])

        # 按平均耗时排序
        avg_stats = {name: np.mean(vals) for name, vals in all_stats.items()}
        sorted_stages = sorted(avg_stats.items(), key=lambda x: x[1], reverse=True)

        lines.append("| Stage | Avg Time (ms) | % of Total |")
        lines.append("|-------|--------------|------------|")
        total = sum(avg_stats.values())
        for name, avg in sorted_stages:
            pct = (avg / total * 100) if total > 0 else 0
            lines.append(f"| {name} | {avg:.1f} | {pct:.1f}% |")

        lines.append(f"\n**Total per batch**: {total:.1f} ms\n")

        lines.append("## 优化建议\n")
        lines.append("1. **persistent_workers=True** - 消除 worker 重建开销\n")
        lines.append("2. **预解码缓存** - 使用 cache=disk 避免重复 JPEG 解码\n")
        lines.append("3. **mosaic 调优** - 如果 Mosaic 占比 > 30%，考虑降低概率或增大 close_mosaic\n")
        lines.append("4. **Worker 数量** - 当前 workers 自动封顶为 cpu_count/gpu_count\n")

        with open(report_path, "w") as f:
            f.write("\n".join(lines))
        print(f"\n📋 Optimization Report saved: {report_path}")


# ============================================================
# 训练循环 Hook (注入到 BaseTrainer)
# ============================================================

def enable_profiling(trainer, save_dir: str | Path | None = None):
    """启用 profiling 并注入到 trainer 的训练循环.

    修改 BaseTrainer._do_train 以记录每个 batch 的耗时。

    Args:
        trainer: BaseTrainer 实例
        save_dir: 报告保存目录 (默认使用 trainer.save_dir)
    """
    global _PROFILER

    save_path = save_dir or trainer.save_dir
    profiler = DataLoadProfiler(enabled=True, save_dir=save_path)

    # 保存原始的 _do_train 方法
    original_do_train = trainer._do_train

    def profiled_do_train(world_size):
        """带 profiling 的训练循环."""
        # 调用原始 _do_train (需要注入 batch 级别计时)
        # 这里使用 Monkey-patch 方式: 替换 preprocess_batch
        original_preprocess = trainer.preprocess_batch

        def profiled_preprocess(batch):
            """带计时的 preprocess_batch."""
            with profiler.timer("collate", use_cuda=False):
                result = original_preprocess(batch)
            return result

        trainer.preprocess_batch = profiled_preprocess

        # 运行原始训练
        try:
            original_do_train(world_size)
        finally:
            # 恢复
            trainer.preprocess_batch = original_preprocess

    trainer._do_train = profiled_do_train
    return profiler


# ============================================================
# 轻量级 DataLoader benchmark (自动找最佳 workers)
# ============================================================

def benchmark_workers(
    dataset,
    batch_size: int = 32,
    max_workers: int = 32,
    num_batches: int = 50,
    rank: int = 0,
) -> dict[int, float]:
    """自动测试不同 worker 数量的吞吐量, 找到最佳值.

    Args:
        dataset: PyTorch Dataset
        batch_size: 每个 batch 的大小
        max_workers: 最大测试的 worker 数
        num_batches: 每个配置测试的 batch 数
        rank: DDP rank

    Returns:
        dict: {workers: throughput_batches_per_second}
    """
    from ultralytics.data.build import build_dataloader

    results = {}
    worker_options = sorted(set([4, 8, 12, 16, 24, 32, max_workers]))

    print("\n" + "=" * 60)
    print("  🔍 Benchmarking DataLoader Workers")
    print("=" * 60)

    for nw in worker_options:
        if nw > max_workers:
            break

        dataloader = build_dataloader(
            dataset, batch=batch_size, workers=nw, shuffle=True, rank=rank
        )

        # 预热
        iterator = iter(dataloader)
        for _ in range(5):
            try:
                next(iterator)
            except StopIteration:
                iterator = iter(dataloader)

        # 计时
        start = time.perf_counter()
        for i in range(num_batches):
            try:
                next(iterator)
            except StopIteration:
                iterator = iter(dataloader)
                next(iterator)
        elapsed = time.perf_counter() - start

        throughput = num_batches / elapsed
        results[nw] = throughput
        print(f"  workers={nw:>3d}: {throughput:>6.2f} batches/s  ({elapsed/num_batches*1000:.0f} ms/batch)")

        # 清理
        del dataloader

    # 找最佳
    best_nw = max(results, key=results.get)
    best_tp = results[best_nw]
    print(f"\n  ✅ Best workers: {best_nw} ({best_tp:.2f} batches/s)")
    print("=" * 60 + "\n")

    return results


# ============================================================
# Mosaic vs No-Mosaic 对比
# ============================================================

def benchmark_mosaic(
    dataset,
    batch_size: int = 32,
    workers: int = 16,
    num_batches: int = 30,
) -> dict[str, float]:
    """对比 Mosaic 开启/关闭的吞吐量差异.

    Returns:
        dict: {"mosaic_ms": avg_ms, "no_mosaic_ms": avg_ms, "overhead_ms": diff_ms, "overhead_pct": pct}
    """
    from ultralytics.data.build import build_dataloader

    results = {}

    for mode in ["mosaic", "no_mosaic"]:
        # 临时设置 mosaic 概率
        original_transforms = dataset.transforms
        if mode == "no_mosaic":
            # 临时禁用 mosaic
            for t in dataset.transforms.transforms:
                if hasattr(t, "transforms"):
                    for t2 in t.transforms:
                        if t2.__class__.__name__ == "Mosaic":
                            t2.p = 0.0

        dataloader = build_dataloader(
            dataset, batch=batch_size, workers=workers, shuffle=True
        )
        iterator = iter(dataloader)

        # 预热
        for _ in range(5):
            next(iterator)

        start = time.perf_counter()
        for _ in range(num_batches):
            next(iterator)
        elapsed = time.perf_counter() - start

        avg_ms = elapsed / num_batches * 1000
        results[f"{mode}_ms"] = avg_ms
        print(f"  {mode}: {avg_ms:.1f} ms/batch")

        # 恢复
        if mode == "no_mosaic":
            for t in original_transforms.transforms:
                if hasattr(t, "transforms"):
                    for t2 in t.transforms:
                        if t2.__class__.__name__ == "Mosaic":
                            t2.p = 1.0

        del dataloader

    overhead = results["mosaic_ms"] - results["no_mosaic_ms"]
    results["overhead_ms"] = overhead
    results["overhead_pct"] = overhead / results["no_mosaic_ms"] * 100

    print(f"\n  📊 Mosaic overhead: +{overhead:.0f} ms/batch ({results['overhead_pct']:.0f}%)")

    if overhead > 100:
        print(f"  ⚠️ Mosaic 是主要瓶颈! 建议: close_mosaic 提前, 或 mosaic=0.5")

    return results


# ============================================================
# CPU 使用率监控
# ============================================================

def check_cpu_saturation() -> dict[str, Any]:
    """检查 CPU 是否过载.

    Returns:
        dict: {"cpu_percent": float, "ram_percent": float, "recommendation": str}
    """
    try:
        import psutil
    except ImportError:
        return {"error": "psutil not installed (pip install psutil)"}

    cpu_percent = psutil.cpu_percent(interval=1)
    ram = psutil.virtual_memory()
    swap = psutil.swap_memory()
    ctx_switches = psutil.cpu_stats().ctx_switches

    result = {
        "cpu_percent": cpu_percent,
        "ram_percent": ram.percent,
        "ram_available_gb": ram.available / (1024**3),
        "swap_percent": swap.percent,
        "ctx_switches": ctx_switches,
    }

    recommendation = ""
    if cpu_percent > 95:
        recommendation = "CPU 饱和! 建议: workers ↓, cache=ram"
    elif cpu_percent > 80:
        recommendation = "CPU 高负载, 观察是否有 page fault"
    else:
        recommendation = "CPU 正常"

    result["recommendation"] = recommendation
    return result
