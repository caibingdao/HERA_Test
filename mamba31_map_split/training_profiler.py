import contextlib
import contextvars
import math
import statistics
import time
from collections import defaultdict

import torch


_ACTIVE_PROFILER = contextvars.ContextVar(
    "ariadne_active_training_profiler",
    default=None,
)


class TrainingProfiler:
    """Opt-in timing for one learner update at a time."""

    def __init__(
        self,
        enabled=False,
        warmup_updates=10,
        active_updates=50,
        log_interval=10,
        device=None,
    ):
        self.enabled = bool(enabled)
        self.warmup_updates = int(warmup_updates)
        self.active_updates = int(active_updates)
        self.log_interval = int(log_interval)
        self.device = torch.device(device or "cpu")
        self.update_count = 0
        self.measured_updates = 0
        self.samples = defaultdict(list)
        self._current_cpu = defaultdict(float)
        self._current_cuda = defaultdict(list)
        self._token = None

        if self.warmup_updates < 0:
            raise ValueError("warmup_updates must be non-negative")
        if self.active_updates <= 0:
            raise ValueError("active_updates must be positive")
        if self.log_interval <= 0:
            raise ValueError("log_interval must be positive")

    @property
    def uses_cuda_events(self):
        return (
            self.enabled
            and self.device.type == "cuda"
            and torch.cuda.is_available()
        )

    @property
    def collecting(self):
        return (
            self.enabled
            and self.update_count
            < self.warmup_updates + self.active_updates
        )

    @contextlib.contextmanager
    def profile_update(self):
        if not self.collecting:
            yield
            return

        self.begin_update()
        try:
            yield
        finally:
            self.end_update()

    def begin_update(self):
        if not self.collecting:
            return False
        self._current_cpu = defaultdict(float)
        self._current_cuda = defaultdict(list)
        if self.uses_cuda_events:
            torch.cuda.reset_peak_memory_stats(self.device)
        self._token = _ACTIVE_PROFILER.set(self)
        return True

    def end_update(self):
        if self._token is not None:
            _ACTIVE_PROFILER.reset(self._token)
            self._token = None
            self._finish_update()

    @contextlib.contextmanager
    def section(self, name, cuda=True):
        if not self.collecting:
            yield
            return

        cpu_start = time.perf_counter()
        cuda_start = None
        cuda_end = None
        if cuda and self.uses_cuda_events:
            cuda_start = torch.cuda.Event(enable_timing=True)
            cuda_end = torch.cuda.Event(enable_timing=True)
            cuda_start.record()
        try:
            yield
        finally:
            if cuda_start is not None:
                cuda_end.record()
                self._current_cuda[name].append((cuda_start, cuda_end))
            else:
                self._current_cpu[name] += (
                    time.perf_counter() - cpu_start
                ) * 1000.0

    def _finish_update(self):
        is_measured = self.update_count >= self.warmup_updates
        if self.uses_cuda_events:
            torch.cuda.synchronize(self.device)
            cuda_values = {
                name: sum(
                    start.elapsed_time(end)
                    for start, end in event_pairs
                )
                for name, event_pairs in self._current_cuda.items()
            }
        else:
            cuda_values = {}

        if is_measured:
            merged = dict(self._current_cpu)
            merged.update(cuda_values)
            if self.uses_cuda_events:
                merged["peak_allocated_mib"] = (
                    torch.cuda.max_memory_allocated(self.device) / 2 ** 20
                )
                merged["peak_reserved_mib"] = (
                    torch.cuda.max_memory_reserved(self.device) / 2 ** 20
                )
            for name, value in merged.items():
                self.samples[name].append(float(value))
            self.measured_updates += 1
        self.update_count += 1

    @property
    def should_log(self):
        return (
            self.enabled
            and self.measured_updates > 0
            and self.measured_updates % self.log_interval == 0
        )

    @staticmethod
    def _percentile(values, percentile):
        if not values:
            return math.nan
        ordered = sorted(values)
        position = (len(ordered) - 1) * percentile / 100.0
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1 - weight) + ordered[upper] * weight

    def summary(self):
        return {
            name: {
                "mean": statistics.fmean(values),
                "p50": self._percentile(values, 50),
                "p95": self._percentile(values, 95),
                "count": len(values),
            }
            for name, values in self.samples.items()
        }

    def format_summary(self):
        summary = self.summary()
        return " | ".join(
            f"{name}: mean={summary[name]['mean']:.2f}, "
            f"p50={summary[name]['p50']:.2f}, "
            f"p95={summary[name]['p95']:.2f}"
            for name in sorted(summary)
        )


@contextlib.contextmanager
def profile_section(name, cuda=True):
    profiler = _ACTIVE_PROFILER.get()
    if profiler is None:
        yield
        return
    with profiler.section(name, cuda=cuda):
        yield
