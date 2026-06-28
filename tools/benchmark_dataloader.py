#!/usr/bin/env python3
"""
DataLoader Profiling 和优化工具集
=================================
用法:
    # Benchmark workers
    python tools/benchmark_dataloader.py --config enhanced --benchmark-workers

    # Benchmark mosaic
    python tools/benchmark_dataloader.py --config enhanced --benchmark-mosaic

    # Full profiling
    python tools/benchmark_dataloader.py --config enhanced --full

    # CPU health check
    python tools/benchmark_dataloader.py --check-cpu
"""

import argparse
import sys
import time
import os
from pathlib import Path

# 添加本地 ultralytics 到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'ultralytics_src'))

import numpy as np
import torch
import yaml

from ultralytics import YOLO
from ultralytics.data.build import build_dataloader, build_yolo_dataset
from ultralytics.utils import RANK, LOGGER
from ultralytics.data.augment import (
    Mosaic, MixUp, RandomPerspective, RandomHSV, RandomFlip,
    Format, Compose, Albumentations,
)


# ============================================================
# DataLoader Benchmark
# ============================================================

def benchmark_workers_full(dataset, batch_size=32, max_workers=32, num_batches=50):
    """自动测试不同 worker 数量的吞吐量, 返回最佳值."""
    results = {}
    worker_options = [4, 8, 12, 16, 24, 32]
    worker_options = [w for w in worker_options if w <= max_workers]

    print("\n" + "=" * 70)
    print("  🔍 DataLoader Worker Benchmark")
    print("=" * 70)
    print(f"  Dataset: {len(dataset)} images, batch_size={batch_size}")
    print(f"  Test batches: {num_batches}")
    print(f"  {'─' * 60}")

    best_tp = 0
    best_nw = 0

    for nw in worker_options:
        try:
            dataloader = build_dataloader(
                dataset, batch=batch_size, workers=nw, shuffle=True, rank=-1,
                persistent=True,
            )
            iterator = iter(dataloader)

            # 预热 5 个 batch
            for _ in range(5):
                try:
                    next(iterator)
                except StopIteration:
                    iterator = iter(dataloader)

            # 计时
            torch.cuda.synchronize()
            start = time.perf_counter()
            for i in range(num_batches):
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(dataloader)
                    batch = next(iterator)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            throughput = num_batches / elapsed
            avg_ms = elapsed / num_batches * 1000
            results[nw] = throughput

            marker = " ⭐ BEST" if throughput > best_tp else ""
            if throughput > best_tp:
                best_tp = throughput
                best_nw = nw

            print(f"  workers={nw:>3d}: {throughput:>6.2f} batches/s  ({avg_ms:.0f} ms/batch){marker}")

        except Exception as e:
            print(f"  workers={nw:>3d}: FAILED ({e})")
            results[nw] = 0

        finally:
            if 'dataloader' in dir():
                del dataloader

    print(f"  {'─' * 60}")
    print(f"  ✅ Best: workers={best_nw} ({best_tp:.2f} batches/s)")
    print("=" * 70 + "\n")
    return results, best_nw


def benchmark_mosaic_full(dataset, batch_size=32, workers=16, num_batches=30):
    """对比 Mosaic 开启 vs 关闭的吞吐量."""
    print("\n" + "=" * 70)
    print("  🔍 Mosaic Overhead Benchmark")
    print("=" * 70)
    print(f"  Dataset: {len(dataset)} images, batch_size={batch_size}, workers={workers}")
    print(f"  Test batches: {num_batches}")
    print(f"  {'─' * 60}")

    results = {}

    for mode_name, mosaic_p in [("Mosaic ON", 1.0), ("No Mosaic", 0.0)]:
        # 临时修改 mosaic 概率
        _set_mosaic_probability(dataset, mosaic_p)

        dataloader = build_dataloader(
            dataset, batch=batch_size, workers=workers, shuffle=True, rank=-1,
            persistent=True,
        )
        iterator = iter(dataloader)

        # 预热
        for _ in range(5):
            try:
                next(iterator)
            except StopIteration:
                iterator = iter(dataloader)

        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_batches):
            try:
                next(iterator)
            except StopIteration:
                iterator = iter(dataloader)
                next(iterator)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        avg_ms = elapsed / num_batches * 1000
        results[mode_name] = avg_ms
        print(f"  {mode_name:<15s}: {avg_ms:>8.1f} ms/batch")

        del dataloader

    # 恢复
    _set_mosaic_probability(dataset, 1.0)

    overhead_ms = results["Mosaic ON"] - results["No Mosaic"]
    overhead_pct = overhead_ms / results["No Mosaic"] * 100

    print(f"  {'─' * 60}")
    print(f"  Mosaic overhead: +{overhead_ms:.0f} ms/batch ({overhead_pct:.0f}%)")
    if overhead_pct > 30:
        print(f"  ⚠️  Mosaic 是主要瓶颈! 建议: close_mosaic 提前, 或 mosaic=0.5")
    print("=" * 70 + "\n")
    return results


def _set_mosaic_probability(dataset, p):
    """临时设置 mosaic 概率 (遍历 Compose 链)."""
    def _set_recursive(transform):
        if isinstance(transform, Compose):
            for t in transform.transforms:
                _set_recursive(t)
        elif hasattr(transform, "transforms"):
            for t in transform.transforms:
                _set_recursive(t)
        if isinstance(transform, Mosaic):
            transform.p = p
    _set_recursive(dataset.transforms)


# ============================================================
# Augment Profiling
# ============================================================

def profile_augment_stages(dataset, num_samples=100):
    """测量每个增强阶段的耗时分布.

    通过临时包装 Compose.__call__ 来统计每个 transform 的 CPU 时间.
    """
    print("\n" + "=" * 70)
    print("  🔍 Augmentation Stage Profiling")
    print("=" * 70)
    print(f"  Samples: {num_samples}")
    print(f"  {'─' * 60}")

    # 收集 per-transform 耗时
    stage_times: dict[str, list[float]] = {}

    # 保存原始的 Compose.__call__
    orig_compose_call = dataset.transforms.__class__.__call__

    def profiled_compose_call(self, data):
        """带计时的 Compose.__call__, 对每个子 transform 进行计时."""
        if not isinstance(data, dict) or "img" not in data:
            return orig_compose_call(self, data)

        for i, t in enumerate(self.transforms):
            t0 = time.perf_counter()
            data = t(data)
            elapsed = (time.perf_counter() - t0) * 1000  # ms
            name = t.__class__.__name__
            stage_times.setdefault(name, []).append(elapsed)
        return data

    # Monkey-patch
    dataset.transforms.__class__.__call__ = profiled_compose_call

    try:
        # 采样
        for idx in range(min(num_samples, len(dataset))):
            _ = dataset[idx]

        # 输出统计
        if not stage_times:
            print("  (no data collected — transforms may use different call path)")
            return {}

        print(f"  {'Stage':<25s} {'Mean(ms)':>10s} {'Std(ms)':>10s} {'%':>8s}")
        print(f"  {'─' * 60}")
        total = sum(np.mean(v) for v in stage_times.values())
        for name in sorted(stage_times.keys(), key=lambda n: np.mean(stage_times[n]), reverse=True):
            vals = stage_times[name]
            mean_v = np.mean(vals)
            pct = mean_v / total * 100 if total > 0 else 0
            print(f"  {name:<25s} {mean_v:>10.1f} {np.std(vals):>10.1f} {pct:>7.1f}%")

        print(f"  {'─' * 60}")
        print(f"  Total per sample: {total:.1f} ms")
        print("=" * 70 + "\n")

    finally:
        # 恢复
        dataset.transforms.__class__.__call__ = orig_compose_call

    return stage_times


# ============================================================
# DDP Barrier / Sync Profiling
# ============================================================

def profile_ddp_sync(dataset, batch_size=32, workers=16, num_batches=100):
    """测量 per-GPU batch 完成时间差异 (检测 DDP 负载不均衡).

    在单 GPU 模式下运行多次, 统计每个 batch 的耗时分布。
    如果 std/mean > 0.2, 说明 batch 负载差异大, DDP 下可能导致同步等待。
    """
    print("\n" + "=" * 70)
    print("  🔍 DDP Load Balance Profiling")
    print("=" * 70)
    print(f"  Dataset: {len(dataset)} images, batch_size={batch_size}")
    print(f"  Test batches: {num_batches}")
    print(f"  {'─' * 60}")

    # 构建 DataLoader (非 DDP)
    dataloader = build_dataloader(
        dataset, batch=batch_size, workers=workers, shuffle=True, rank=-1,
        persistent=True,
    )
    iterator = iter(dataloader)

    batch_times = []
    instance_counts = []

    torch.cuda.synchronize()
    for _ in range(num_batches):
        t0 = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            batch = next(iterator)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        batch_times.append(elapsed * 1000)  # ms
        if "cls" in batch:
            instance_counts.append(len(batch["cls"]))

    batch_times = np.array(batch_times)
    mean_t = np.mean(batch_times)
    std_t = np.std(batch_times)
    cv = std_t / mean_t if mean_t > 0 else 0  # 变异系数

    print(f"  Mean batch time: {mean_t:.1f} ms")
    print(f"  Std batch time:  {std_t:.1f} ms")
    print(f"  CV (std/mean):   {cv:.3f}")
    print(f"  Min/Max:         {np.min(batch_times):.0f} / {np.max(batch_times):.0f} ms")
    print(f"  Imbalance ratio: {np.max(batch_times)/np.min(batch_times):.1f}x")
    print(f"  {'─' * 60}")

    if cv > 0.3:
        print(f"  ⚠️  变异系数 > 0.3, batch 负载严重不均衡!")
        print(f"     DDP 下某张 GPU 可能等待其他 GPU 完成")
        print(f"     原因: 图像尺寸不同 (multi_scale) 或 目标数量差异大")
        print(f"     建议: 关闭 multi_scale 或 固定 image size")
    elif cv > 0.15:
        print(f"  ⚡ 变异系数 > 0.15, batch 负载略有波动")
    else:
        print(f"  ✅ 变异系数 < 0.15, batch 负载均衡")

    print("=" * 70 + "\n")
    return {"mean_ms": mean_t, "std_ms": std_t, "cv": cv, "imbalance_ratio": float(np.max(batch_times)/np.min(batch_times))}


# ============================================================
# CPU Health Check
# ============================================================

def check_cpu_health():
    """检查 CPU 和内存使用情况."""
    print("\n" + "=" * 70)
    print("  🔍 CPU / System Health Check")
    print("=" * 70)

    try:
        import psutil
    except ImportError:
        print("  ❌ psutil not installed. Run: pip install psutil")
        print("=" * 70 + "\n")
        return

    cpu_pct = psutil.cpu_percent(interval=1)
    per_cpu = psutil.cpu_percent(interval=0.1, percpu=True)
    ram = psutil.virtual_memory()
    swap = psutil.swap_memory()

    print(f"  CPU Usage:     {cpu_pct:.1f}%")
    print(f"  CPU Cores:     {psutil.cpu_count()} logical")
    print(f"  Per-CPU:       min={min(per_cpu):.0f}% max={max(per_cpu):.0f}% avg={np.mean(per_cpu):.0f}%")
    print(f"  RAM Used:      {ram.used/(1024**3):.1f}G / {ram.total/(1024**3):.1f}G ({ram.percent:.0f}%)")
    print(f"  RAM Available: {ram.available/(1024**3):.1f}G")
    if swap.total > 0:
        print(f"  Swap:          {swap.used/(1024**3):.1f}G / {swap.total/(1024**3):.1f}G ({swap.percent:.0f}%)")

    # 检查 page faults
    try:
        import resource
        page_faults = resource.getrusage(resource.RUSAGE_SELF).ru_majflt
        print(f"  Page Faults:   {page_faults} (major)")
    except Exception:
        pass

    print(f"  {'─' * 60}")

    # 诊断
    issues = []
    if cpu_pct > 95:
        issues.append("CPU 饱和 (>95%): 建议减少 workers")
    elif cpu_pct > 80:
        issues.append("CPU 高负载 (>80%): 考虑 cache=ram 减轻解码开销")

    if ram.percent > 90:
        issues.append(f"RAM 不足 ({ram.percent}%): 考虑 cache=disk 代替 cache=ram")
    elif ram.percent > 70:
        issues.append(f"RAM 使用较高 ({ram.percent}%): 留意 OOM 风险")

    if any(swap.total > 0 and swap.percent > 5 for _ in [1]):
        if swap.percent > 5:
            issues.append(f"Swap 使用 ({swap.percent}%): 可能影响性能")

    if issues:
        for iss in issues:
            print(f"  ⚠️  {iss}")
    else:
        print(f"  ✅ System resources healthy")

    print("=" * 70 + "\n")


# ============================================================
# Full Profiling (所有检查)
# ============================================================

def run_full_profiling(config_yaml, fold=0, num_batches=30):
    """运行全部 profiling 检查."""
    print("\n" + "=" * 70)
    print("  🚀 Full DataLoader Profiling Suite")
    print("=" * 70)

    # 加载配置 (补全默认值)
    from ultralytics.cfg import get_cfg
    from ultralytics.utils import DEFAULT_CFG_DICT

    cfg_dict = DEFAULT_CFG_DICT.copy()
    with open(config_yaml) as f:
        yaml_cfg = yaml.safe_load(f)
        cfg_dict.update({k: v for k, v in yaml_cfg.items() if v is not None})
    cfg = get_cfg(cfg_dict)

    batch_size = cfg.batch
    workers = cfg.workers

    # 构建数据集
    data_yaml_path = str(PROJECT_ROOT / f'dataset_yolo/fold_{fold}/data.yaml')
    with open(data_yaml_path) as f:
        data_dict = yaml.safe_load(f)
    dataset = build_yolo_dataset(
        cfg=cfg,
        img_path=str(PROJECT_ROOT / f'dataset_yolo/fold_{fold}/train/images'),
        batch=batch_size,
        data=data_dict,
        mode='train',
        rect=False,
        stride=32,
    )
    print(f"  Dataset loaded: {len(dataset)} train images")

    # 1. Worker benchmark
    worker_results, best_nw = benchmark_workers_full(dataset, batch_size, workers, num_batches)

    # 2. Mosaic benchmark
    mosaic_results = benchmark_mosaic_full(dataset, batch_size, best_nw, num_batches)

    # 3. DDP load balance
    ddp_results = profile_ddp_sync(dataset, batch_size, best_nw, num_batches)

    # 4. Augment profiling
    augment_results = profile_augment_stages(dataset, min(num_batches * 2, 200))

    # 5. CPU check
    check_cpu_health()

    # 生成报告
    print("\n" + "=" * 70)
    print("  📋 Summary & Recommendations")
    print("=" * 70)
    print(f"  Recommended workers: {best_nw}")
    print(f"  Mosaic overhead: {mosaic_results.get('overhead_pct', 0):.0f}%")
    print(f"  Batch time CV: {ddp_results.get('cv', 0):.3f}")
    print("=" * 70 + "\n")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DataLoader Profiling Suite")
    parser.add_argument("--config", type=str, default="enhanced", help="Config: baseline or enhanced")
    parser.add_argument("--fold", type=int, default=0, help="Fold index")
    parser.add_argument("--full", action="store_true", help="Run all profiling checks")
    parser.add_argument("--benchmark-workers", action="store_true", help="Benchmark optimal workers")
    parser.add_argument("--benchmark-mosaic", action="store_true", help="Benchmark mosaic overhead")
    parser.add_argument("--profile-ddp", action="store_true", help="Profile DDP load balance")
    parser.add_argument("--profile-augment", action="store_true", help="Profile augmentation stages")
    parser.add_argument("--check-cpu", action="store_true", help="Check CPU/memory health")
    parser.add_argument("--num-batches", type=int, default=30, help="Batches to test per config")
    args_parsed = parser.parse_args()

    config_path = PROJECT_ROOT / f"configs/{args_parsed.config}.yaml"
    if not config_path.exists():
        print(f"❌ Config not found: {config_path}")
        sys.exit(1)

    if args_parsed.full:
        run_full_profiling(config_path, args_parsed.fold, args_parsed.num_batches)
    else:
        from ultralytics.cfg import get_cfg
        from ultralytics.utils import DEFAULT_CFG_DICT

        cfg_dict = DEFAULT_CFG_DICT.copy()
        with open(config_path) as f:
            yaml_cfg = yaml.safe_load(f)
            cfg_dict.update({k: v for k, v in yaml_cfg.items() if v is not None})
        cfg_dict["fraction"] = 0.01  # 只用1%数据来做benchmark
        cfg = get_cfg(cfg_dict)
        batch_size = cfg.batch
        workers = cfg.workers

        # 加载 data.yaml (classes, path 等)
        data_yaml_path = str(PROJECT_ROOT / f'dataset_yolo/fold_{args_parsed.fold}/data.yaml')
        with open(data_yaml_path) as f:
            data_dict = yaml.safe_load(f)

        dataset = build_yolo_dataset(
            cfg=cfg,
            img_path=str(PROJECT_ROOT / f'dataset_yolo/fold_{args_parsed.fold}/train/images'),
            batch=batch_size,
            data=data_dict,
            mode='train',
            rect=False,
            stride=32,
        )
        print(f"Dataset loaded: {len(dataset)} images (fraction=0.01 for quick benchmark)")

        if args_parsed.benchmark_workers:
            benchmark_workers_full(dataset, batch_size, workers, args_parsed.num_batches)
        if args_parsed.benchmark_mosaic:
            benchmark_mosaic_full(dataset, batch_size, workers, args_parsed.num_batches)
        if args_parsed.profile_ddp:
            profile_ddp_sync(dataset, batch_size, workers, args_parsed.num_batches)
        if args_parsed.profile_augment:
            profile_augment_stages(dataset, min(args_parsed.num_batches * 2, 100))
        if args_parsed.check_cpu:
            check_cpu_health()
