"""Public, framework-neutral contract for evaluation adapters.

This module contains typing aids only; adapters do not need to inherit from a
base class.  The driver validates the same contract at runtime.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Protocol, TypedDict


TerminationReason = Literal[
    "target_reached",
    "max_steps",
    "max_path_length",
    "no_valid_action",
    "environment_complete",
    "environment_failure",
    "collision",
]


class AdapterMetadata(TypedDict):
    """Stable identity of one independently trained model."""

    model_name: str
    model_seed: int


class EvaluationConfig(TypedDict):
    """Common budget and policy settings supplied for every episode."""

    target_coverage: float
    max_steps: int
    max_path_length: Optional[float]
    deterministic: bool


class EpisodeMetrics(TypedDict):
    """Model/environment metrics required after one episode."""

    start_x: float
    start_y: float
    start_resample_count: int
    environment_success: bool
    termination_reason: TerminationReason
    path_length: float
    steps: int
    final_coverage: float
    elapsed_time_seconds: float
    mean_inference_time_ms: float
    max_inference_time_ms: float


class CoveragePoint(TypedDict):
    """Coverage sampled after a decision (including the step-zero row)."""

    step: int
    path_length: float
    coverage: float


class EpisodeEvaluation(TypedDict):
    episode: EpisodeMetrics
    coverage_curve: List[CoveragePoint]


class EvaluationAdapter(Protocol):
    """Structural interface consumed by ``evaluate_on_split.py``."""

    def metadata(self) -> AdapterMetadata:
        ...

    def evaluate_episode(
        self,
        *,
        map_path: Path,
        map_id: str,
        eval_seed: int,
        start_seed: int,
        episode_id: int,
        config: EvaluationConfig,
    ) -> EpisodeEvaluation:
        ...


class AdapterFactory(Protocol):
    """Signature of the callable named by ``--adapter module:factory``."""

    def __call__(
        self,
        *,
        checkpoint: Path,
        deterministic: bool,
        adapter_args: Dict[str, Any],
    ) -> EvaluationAdapter:
        ...
