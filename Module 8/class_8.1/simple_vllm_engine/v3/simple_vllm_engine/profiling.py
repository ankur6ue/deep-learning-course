from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass

import torch


@dataclass
class TimingStat:
    total_s: float = 0.0
    count: int = 0


class SimpleProfiler:
    """Simple synchronized wall-clock profiler for the teaching engine.

    When enabled on CUDA, each section synchronizes before reading the clock.
    That adds overhead, so this profiler is for attribution and teaching rather
    than for final throughput measurements.
    """

    def __init__(self, device: str, enabled: bool = False) -> None:
        self.device = device
        self.enabled = enabled
        self.stats: dict[str, TimingStat] = {}

    def _sync(self) -> None:
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()

    @contextmanager
    def section(self, name: str):
        if not self.enabled:
            yield
            return
        self._sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            dt = time.perf_counter() - t0
            stat = self.stats.setdefault(name, TimingStat())
            stat.total_s += dt
            stat.count += 1

    def summary(self) -> list[tuple[str, TimingStat]]:
        return sorted(self.stats.items(), key=lambda item: item[1].total_s, reverse=True)
