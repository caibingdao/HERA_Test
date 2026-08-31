#!/usr/bin/env python3
"""Run the standard five-seed evaluation with stochastic policy sampling.

This is a deliberately thin, independent entry point around
``evaluate_parallel.py``.  It preserves that evaluator's map split, per-map
seed reset, start-position derivation, worker layout, result schema and merge
validation.  The only experimental change is action selection: stochastic
``torch.multinomial`` is forced instead of greedy ``argmax``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiment_utils import write_json  # noqa: E402
from scripts import evaluate_parallel  # noqa: E402


class StochasticParallelEvaluationError(RuntimeError):
    """Raised when stochastic-only evaluation invariants are violated."""


def build_parser():
    """Return the standard parallel parser with stochastic policy by default."""

    parser = evaluate_parallel.build_parser()
    parser.description = (
        "Evaluate five independent seeds in parallel using stochastic "
        "torch.multinomial action sampling.  All other settings and seed "
        "semantics match scripts/evaluate_parallel.py."
    )
    parser.set_defaults(deterministic=False)
    for action in parser._actions:
        if action.dest == "deterministic":
            action.help = (
                "Must remain disabled; this independent evaluator only "
                "supports stochastic probability sampling."
            )
    return parser


def _annotate_metadata(output_dir: Path) -> None:
    metadata_path = output_dir / "evaluation_metadata.json"
    with metadata_path.open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    if metadata.get("deterministic") is not False:
        raise StochasticParallelEvaluationError(
            "merged metadata does not record stochastic evaluation"
        )
    metadata.update(
        {
            "evaluation_mode": "five_seed_stochastic_probability_sampling",
            "action_selection": "torch.multinomial(policy_probabilities)",
            "seed_reset_scope": "each_map_episode",
            "base_evaluator": "scripts/evaluate_parallel.py",
            "single_variable_change_from_greedy_baseline": (
                "action_selection"
            ),
        }
    )
    write_json(metadata_path, metadata)


def main(argv: Sequence[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(raw_args)
    if args.deterministic:
        raise StochasticParallelEvaluationError(
            "--deterministic is not allowed: this program fixes action "
            "selection to stochastic torch.multinomial sampling"
        )

    # Pass an explicit final flag so the base evaluator cannot silently revert
    # to its greedy default if its defaults change later.
    return_code = evaluate_parallel.main([*raw_args, "--no-deterministic"])
    _annotate_metadata(args.output_dir)
    print("Stochastic action selection: torch.multinomial")
    print("Seed reset scope: each map episode (same as greedy baseline)")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
