"""
高精度测试计时工具 - 毫秒级统计

提供多种计时方式和统计分析功能
"""
from amgpn.config import require as _cfg_require, resolve as _cfg_resolve, format_value as _cfg_format

import time
import torch
from contextlib import contextmanager
from collections import defaultdict
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import matplotlib.pyplot as plt


class Timer:
    "\n    高精度计时器\n    \n    使用方法：\n        timer = Timer()\n        \n        # 方法<configured>：手动计时\n        timer.start('model_forward')\n        # ... 代码\n        timer.stop('model_forward')\n        \n        # 方法<configured>：上下文管理器\n        with timer.time('model_forward'):\n            # ... 代码\n            pass\n        \n        # 获取统计\n        stats = timer.get_stats()\n    "

    def __init__(self, sync_cuda=True):
        """
        Args:
            sync_cuda: 是否同步CUDA（GPU计时必须开启）
        """
        self.sync_cuda = sync_cuda
        self.timings = defaultdict(list)  # {name: [time1, time2, ...]}
        self.start_times = {}  # {name: start_time}

    def start(self, name: str):
        """开始计时"""
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_times[name] = time.perf_counter()

    def stop(self, name: str) -> float:
        """
        停止计时

        Returns:
            elapsed_time: 经过的时间（秒）
        """
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()

        if name not in self.start_times:
            raise ValueError(f"Timer '{name}' was not started")

        elapsed = time.perf_counter() - self.start_times[name]
        self.timings[name].append(elapsed)
        del self.start_times[name]

        return elapsed

    @contextmanager
    def time(self, name: str):
        """
        上下文管理器计时

        用法：
            with timer.time('operation'):
                # 代码
                pass
        """
        self.start(name)
        try:
            yield
        finally:
            self.stop(name)

    def get_stats(self, unit='ms') -> Dict:
        """
        获取统计信息

        Args:
            unit: 'ms' (毫秒) 或 's' (秒)

        Returns:
            stats: {name: {'mean', 'std', 'min', 'max', 'total', 'count'}}
        """
        multiplier = _cfg_require('timing_utils.py.Timer.get_stats.size_or_budget') if unit == 'ms' else 1

        stats = {}
        for name, times in self.timings.items():
            times_array = np.array(times) * multiplier
            stats[name] = {
                'mean': np.mean(times_array),
                'std': np.std(times_array),
                'min': np.min(times_array),
                'max': np.max(times_array),
                'median': np.median(times_array),
                'total': np.sum(times_array),
                'count': len(times_array),
                'unit': unit
            }

        return stats

    def print_stats(self, unit='ms', sort_by='mean'):
        """
        打印统计信息

        Args:
            unit: 'ms' 或 's'
            sort_by: 排序依据 ('mean', 'total', 'count')
        """
        stats = self.get_stats(unit)

        if not stats:
            print("No timing data available")
            return

        # 排序
        sorted_items = sorted(
            stats.items(),
            key=lambda x: x[1][sort_by],
            reverse=True
        )

        # 打印表头
        print("\n" + "="*90)
        print(f"{'Operation':<30} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12} {'Count':<8}")
        print("="*90)

        # 打印每一行
        for name, stat in sorted_items:
            print(
                f"{name:<30} {stat['mean']:>10.2f}{unit:<2} {stat['std']:>10.2f}{unit:<2} {stat['min']:>10.2f}{unit:<2} {stat['max']:>10.2f}{unit:<2} {stat['count']:>8}"
            )

        print("="*90)

        # 打印总览
        total_time = sum(s['total'] for s in stats.values())
        print(f"\nTotal time: {total_time:.2f} {unit}")
        print(f"Number of operations: {len(stats)}")

    def to_dataframe(self, unit='ms') -> pd.DataFrame:
        """转换为DataFrame"""
        stats = self.get_stats(unit)
        return pd.DataFrame(stats).T

    def reset(self):
        """重置所有计时数据"""
        self.timings.clear()
        self.start_times.clear()

    def get_raw_times(self, name: str, unit='ms') -> List[float]:
        """
        获取原始时间数据

        Args:
            name: 操作名称
            unit: 'ms' 或 's'

        Returns:
            times: 时间列表
        """
        if name not in self.timings:
            return []

        multiplier = _cfg_require('timing_utils.py.Timer.get_raw_times.size_or_budget') if unit == 'ms' else 1
        return [t * multiplier for t in self.timings[name]]


class TaskTimer:
    """
    任务级计时器 - 专门用于Few-Shot测试

    自动统计每个任务的各个阶段用时
    """

    def __init__(self):
        self.timer = Timer(sync_cuda=True)
        self.task_times = []  # 每个任务的总时间
        self.current_task_start = None

    def start_task(self, task_id: int):
        """开始一个新任务"""
        self.current_task_start = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def end_task(self, task_id: int):
        """结束当前任务"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        if self.current_task_start is not None:
            elapsed = time.perf_counter() - self.current_task_start
            self.task_times.append(elapsed * _cfg_require('timing_utils.py.TaskTimer.end_task.size_or_budget'))  # 转换为毫秒
            self.current_task_start = None

    def time_operation(self, name: str):
        """返回上下文管理器用于计时操作"""
        return self.timer.time(name)

    def get_task_stats(self) -> Dict:
        """获取任务级统计"""
        if not self.task_times:
            return {}

        times = np.array(self.task_times)
        return {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times),
            'median': np.median(times),
            'total': np.sum(times),
            'count': len(times),
            'unit': 'ms'
        }

    def get_operation_stats(self, unit='ms') -> Dict:
        """获取操作级统计"""
        return self.timer.get_stats(unit)

    def print_summary(self):
        """打印完整统计摘要"""
        print("\n" + "="*90)
        print("TIMING SUMMARY")
        print("="*90)

        # 任务级统计
        task_stats = self.get_task_stats()
        if task_stats:
            print(f"\n【任务级统计】({task_stats['count']} tasks)")
            print(f"  平均每任务: {task_stats['mean']:.2f} ms")
            print(f"  标准差: {task_stats['std']:.2f} ms")
            print(f"  最快任务: {task_stats['min']:.2f} ms")
            print(f"  最慢任务: {task_stats['max']:.2f} ms")
            print(f"  总时间: {task_stats['total']/_cfg_require('timing_utils.py.TaskTimer.print_summary.size_or_budget'):.2f} s")

        # 操作级统计
        print(f"\n【操作级统计】")
        self.timer.print_stats(unit='ms')


def plot_timing_analysis(timer: Timer, save_path: Optional[str] = None):
    """
    可视化计时分析

    Args:
        timer: Timer对象
        save_path: 保存路径（可选）
    """
    stats = timer.get_stats('ms')

    if not stats:
        print("No timing data to plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # <configured>. 平均时间柱状图
    ax1 = axes[0, 0]
    names = list(stats.keys())
    means = [stats[n]['mean'] for n in names]
    stds = [stats[n]['std'] for n in names]

    x_pos = np.arange(len(names))
    bars = ax1.bar(x_pos, means, yerr=stds, capsize=5, alpha=_cfg_require('timing_utils.py.plot_timing_analysis.alpha'), color='steelblue')
    ax1.set_xlabel('Operation', fontsize=11)
    ax1.set_ylabel('Time (ms)', fontsize=11)
    ax1.set_title('Average Time per Operation', fontsize=13, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(names, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=_cfg_require('timing_utils.py.plot_timing_analysis.alpha__2'))

    # 添加数值标签
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.1f}', ha='center', va='bottom', fontsize=9)

    # <configured>. 时间占比饼图
    ax2 = axes[0, 1]
    totals = [stats[n]['total'] for n in names]
    colors = plt.cm.Set3(np.linspace(0, 1, len(names)))

    wedges, texts, autotexts = ax2.pie(
        totals, labels=names, autopct='%1.1f%%',
        colors=colors, startangle=90
    )
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)
    ax2.set_title('Time Distribution', fontsize=13, fontweight='bold')

    # <configured>. 箱线图（显示分布）
    ax3 = axes[1, 0]
    data_to_plot = [timer.get_raw_times(name, 'ms') for name in names]
    bp = ax3.boxplot(data_to_plot, labels=names, patch_artist=True)

    # 美化箱线图
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax3.set_xlabel('Operation', fontsize=11)
    ax3.set_ylabel('Time (ms)', fontsize=11)
    ax3.set_title('Time Distribution (Boxplot)', fontsize=13, fontweight='bold')
    ax3.set_xticklabels(names, rotation=45, ha='right')
    ax3.grid(axis='y', alpha=_cfg_require('timing_utils.py.plot_timing_analysis.alpha__3'))

    # <configured>. 时间序列（显示每次执行的时间）
    ax4 = axes[1, 1]
    for i, name in enumerate(names):
        times = timer.get_raw_times(name, 'ms')
        if times:
            ax4.plot(times, label=name, marker='o', markersize=3, alpha=_cfg_require('timing_utils.py.plot_timing_analysis.alpha__5'))

    ax4.set_xlabel('Iteration', fontsize=11)
    ax4.set_ylabel('Time (ms)', fontsize=11)
    ax4.set_title('Time Series', fontsize=13, fontweight='bold')
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(True, alpha=_cfg_require('timing_utils.py.plot_timing_analysis.alpha__4'))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=_cfg_require('timing_utils.py.plot_timing_analysis.size_or_budget'), bbox_inches='tight')
        print(f"\n图表已保存: {save_path}")

    plt.show()


# ============================================================================
# 使用示例
# ============================================================================

def example_usage():
    """使用示例"""

    print('示例<configured>：基本计时')
    timer = Timer()

    # 方法<configured>：手动计时
    timer.start('operation1')
    time.sleep(0.01)  # 模拟操作
    timer.stop('operation1')

    # 方法<configured>：上下文管理器
    with timer.time('operation2'):
        time.sleep(0.02)

    # 打印统计
    timer.print_stats('ms')

    print('\n示例<configured>：Few-Shot测试计时')
    task_timer = TaskTimer()

    # 模拟<configured>个测试任务
    for i in range(5):  # 简化为<configured>个任务演示
        task_timer.start_task(i)

        # 数据加载
        with task_timer.time_operation('data_loading'):
            time.sleep(0.005)

        # 前向传播
        with task_timer.time_operation('forward_pass'):
            time.sleep(0.01)

        # 损失计算
        with task_timer.time_operation('loss_computation'):
            time.sleep(0.003)

        task_timer.end_task(i)

    # 打印摘要
    task_timer.print_summary()

    print('\n示例<configured>：生成可视化图表')


if __name__ == "__main__":
    print("高精度计时工具已加载")
    print("\n主要类:")
    print("  1. Timer - 基础计时器")
    print("  2. TaskTimer - 任务级计时器")
    print("\n主要函数:")
    print("  - plot_timing_analysis() - 可视化分析")
    print("\n运行示例:")
    print("  python timing_utils.py")

    # 运行示例
