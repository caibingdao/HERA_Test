import csv
import io
import json
import math
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from experiment_utils import (
    coverage_success,
    load_split_manifest,
    trajectory_path_length,
)
from scripts import aggregate_results
from scripts import evaluate_on_split
from scripts import split_maps


class FakeOptimizer:
    def step(self):
        raise AssertionError("evaluation must never call optimizer.step()")


class FakeEvaluationAdapter:
    def __init__(self):
        self.optimizer = FakeOptimizer()
        self.closed = False

    def metadata(self):
        return {
            "model_name": "fake-policy",
            "model_seed": 7,
        }

    def evaluate_episode(
        self,
        *,
        map_path,
        map_id,
        eval_seed,
        start_seed,
        episode_id,
        config,
    ):
        del map_path, map_id, eval_seed, start_seed, episode_id
        target = float(config["target_coverage"])
        return {
            "episode": {
                "start_x": 0.0,
                "start_y": 0.0,
                "start_resample_count": 0,
                "environment_success": True,
                "termination_reason": "target_reached",
                "path_length": 5.0,
                "steps": 2,
                "final_coverage": target,
                "path_length_to_target": 5.0,
                "steps_to_target": 2,
                "elapsed_time_seconds": 0.01,
                "mean_inference_time_ms": 1.0,
                "max_inference_time_ms": 1.5,
            },
            "coverage_curve": [
                {
                    "step": 0,
                    "path_length": 0.0,
                    "coverage": 0.10,
                },
                {
                    "step": 1,
                    "path_length": 3.0,
                    "coverage": 0.70,
                },
                {
                    "step": 2,
                    "path_length": 5.0,
                    "coverage": target,
                },
            ],
        }

    def close(self):
        self.closed = True


class MapSplitExperimentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._temporary_directory = tempfile.TemporaryDirectory(
            prefix="ariadne_map_split_test_"
        )
        cls.root = Path(cls._temporary_directory.name)
        cls.map_root = cls.root / "maps"
        cls.map_root.mkdir()

        cls.base_masks = []
        for index in range(16):
            rng = np.random.default_rng(10_000 + index)
            mask = rng.random((13, 17)) > 0.70
            mask[0, 0] = True
            mask[0, -1] = True
            mask[-1, 0] = True
            mask[-1, -1] = True
            cls.base_masks.append(mask)
            cls._write_map(
                cls.map_root / f"base_{index:02d}.png",
                mask,
            )

        variant_dir = cls.map_root / "variants"
        variant_dir.mkdir()
        cls._write_map(
            variant_dir / "base_00_rot90.png",
            np.rot90(cls.base_masks[0]),
        )

        cls.records = split_maps.scan_map_records(cls.map_root)
        cls.manifest_path = cls.root / "split.json"
        cls.manifest = split_maps.build_split_manifest(
            cls.map_root,
            train_ratio=0.75,
            seed=42,
            output=cls.manifest_path,
            group_by_base_map=True,
            stratify=True,
            records=cls.records,
        )
        split_maps.write_manifest(cls.manifest_path, cls.manifest)

    @classmethod
    def tearDownClass(cls):
        cls._temporary_directory.cleanup()

    @staticmethod
    def _write_map(path, free_mask):
        image = np.asarray(free_mask, dtype=np.uint8) * 255
        Image.fromarray(image, mode="L").save(path)

    def _build_manifest(self, seed, output):
        return split_maps.build_split_manifest(
            self.map_root,
            train_ratio=0.75,
            seed=seed,
            output=output,
            group_by_base_map=True,
            stratify=True,
            records=self.records,
        )

    def _evaluation_args(
        self,
        *,
        checkpoint,
        split_file,
        output_dir,
        adapter="fake_test_adapter:create_adapter",
    ):
        # Build the public run_evaluation namespace directly so this otherwise
        # dependency-light suite also runs on Python 3.8, whose argparse lacks
        # BooleanOptionalAction.
        return types.SimpleNamespace(
            checkpoint=Path(checkpoint),
            split_file=Path(split_file),
            split_name="test",
            eval_seeds=[100],
            episodes_per_map=1,
            target_coverage=0.99,
            max_steps=4,
            max_path_length=None,
            output_dir=Path(output_dir),
            deterministic=True,
            adapter=adapter,
            adapter_arg=[],
            fail_fast=True,
        )

    @staticmethod
    def _compatible_evaluation_metadata(
        checkpoint_sha256,
        **overrides,
    ):
        metadata = {
            "split_manifest_sha256": "f" * 64,
            "split_name": "test",
            "num_maps": 1,
            "num_episodes": 2,
            "eval_seeds": [100, 101],
            "episodes_per_map": 2,
            "target_coverage": 0.99,
            "deterministic": True,
            "max_steps": 4,
            "max_path_length": None,
            "checkpoint_sha256": checkpoint_sha256,
        }
        metadata.update(overrides)
        return metadata

    @staticmethod
    def _write_aggregation_fixture(
        directory,
        *,
        model_name,
        checkpoint,
        model_seed,
        episode_grid,
        metadata,
        curve_grid=None,
        exception_grid=(),
        start_seed_offset=0,
        start_coordinate_offset=0.0,
    ):
        directory.mkdir()
        episode_fields = [
            "model_name",
            "checkpoint",
            "map_id",
            "model_seed",
            "eval_seed",
            "start_position_seed",
            "episode_id",
            "start_x",
            "start_y",
            "success",
            "termination_reason",
            "path_length",
            "steps",
            "final_coverage",
        ]
        with (directory / "episodes.csv").open(
            "w",
            encoding="utf-8",
            newline="",
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=episode_fields)
            writer.writeheader()
            for map_id, eval_seed, episode_id in episode_grid:
                grid_identity = (map_id, eval_seed, episode_id)
                is_exception = grid_identity in exception_grid
                writer.writerow(
                    {
                        "model_name": model_name,
                        "checkpoint": checkpoint,
                        "map_id": map_id,
                        "model_seed": model_seed,
                        "eval_seed": eval_seed,
                        "start_position_seed": (
                            ""
                            if start_seed_offset is None
                            else 1_000 + episode_id + start_seed_offset
                        ),
                        "episode_id": episode_id,
                        "start_x": (
                            float(episode_id) + start_coordinate_offset
                        ),
                        "start_y": (
                            float(-episode_id) + start_coordinate_offset
                        ),
                        "success": not is_exception,
                        "termination_reason": (
                            "exception" if is_exception else "target_reached"
                        ),
                        "path_length": 5.0,
                        "steps": 2,
                        "final_coverage": 0.99,
                    }
                )

        curve_fields = [
            "model_name",
            "model_seed",
            "map_id",
            "eval_seed",
            "episode_id",
            "step",
            "path_length",
            "coverage",
        ]
        with (directory / "coverage_curve.csv").open(
            "w",
            encoding="utf-8",
            newline="",
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=curve_fields)
            writer.writeheader()
            selected_grid = episode_grid if curve_grid is None else curve_grid
            for map_id, eval_seed, episode_id in selected_grid:
                writer.writerow(
                    {
                        "model_name": model_name,
                        "model_seed": model_seed,
                        "map_id": map_id,
                        "eval_seed": eval_seed,
                        "episode_id": episode_id,
                        "step": 0,
                        "path_length": 0.0,
                        "coverage": 0.1,
                    }
                )

        if metadata is not None:
            with (directory / "evaluation_metadata.json").open(
                "w",
                encoding="utf-8",
            ) as stream:
                json.dump(metadata, stream)

    def test_same_seed_manifest_is_byte_identical(self):
        output_a = self.root / "same_seed_a.json"
        output_b = self.root / "same_seed_b.json"
        split_maps.write_manifest(output_a, self._build_manifest(42, output_a))
        split_maps.write_manifest(output_b, self._build_manifest(42, output_b))

        self.assertEqual(output_a.read_bytes(), output_b.read_bytes())

    def test_different_seed_changes_split(self):
        output_a = self.root / "different_seed_a.json"
        output_b = self.root / "different_seed_b.json"
        manifest_a = self._build_manifest(1, output_a)
        manifest_b = self._build_manifest(2, output_b)

        self.assertNotEqual(
            set(manifest_a["train_maps"]),
            set(manifest_b["train_maps"]),
        )

    def test_train_and_test_are_disjoint_and_complete(self):
        train_maps = set(self.manifest["train_maps"])
        test_maps = set(self.manifest["test_maps"])
        expected = {record.relative_path for record in self.records}

        self.assertFalse(train_maps & test_maps)
        self.assertEqual(train_maps | test_maps, expected)
        self.assertEqual(
            len(self.manifest["train_maps"]),
            len(train_maps),
        )
        self.assertEqual(
            len(self.manifest["test_maps"]),
            len(test_maps),
        )

    def test_grouped_variants_do_not_leak(self):
        groups = self.manifest["map_groups"]
        base = "base_00.png"
        rotated = "variants/base_00_rot90.png"
        self.assertEqual(groups[base], groups[rotated])

        train_maps = set(self.manifest["train_maps"])
        self.assertEqual(base in train_maps, rotated in train_maps)
        train_groups = {
            groups[path] for path in self.manifest["train_maps"]
        }
        test_groups = {
            groups[path] for path in self.manifest["test_maps"]
        }
        self.assertFalse(train_groups & test_groups)

    def test_generic_evaluator_never_calls_optimizer_step(self):
        checkpoint = self.root / "fake_checkpoint.pth"
        checkpoint.write_bytes(b"fake checkpoint for adapter smoke test")
        output_dir = self.root / "fake_evaluation"

        adapters = []

        def create_adapter(*, checkpoint, deterministic, adapter_args):
            self.assertTrue(checkpoint.is_file())
            self.assertTrue(deterministic)
            self.assertEqual(adapter_args, {})
            adapter = FakeEvaluationAdapter()
            adapters.append(adapter)
            return adapter

        fake_module = types.ModuleType("fake_test_adapter")
        fake_module.create_adapter = create_adapter
        args = self._evaluation_args(
            checkpoint=checkpoint,
            split_file=self.manifest_path,
            output_dir=output_dir,
        )

        with mock.patch.dict(
            sys.modules,
            {"fake_test_adapter": fake_module},
        ), mock.patch.object(
            FakeOptimizer,
            "step",
            autospec=True,
            side_effect=AssertionError(
                "evaluation must never call optimizer.step()"
            ),
        ) as optimizer_step:
            outputs = evaluate_on_split.run_evaluation(args)

        optimizer_step.assert_not_called()
        self.assertEqual(len(adapters), 1)
        self.assertTrue(adapters[0].closed)
        self.assertTrue(outputs["episodes"].is_file())
        self.assertTrue(outputs["coverage_curve"].is_file())

        with outputs["episodes"].open(
            "r",
            encoding="utf-8",
            newline="",
        ) as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), len(self.manifest["test_maps"]))
        self.assertTrue(all(row["success"] == "True" for row in rows))
        self.assertTrue(
            all(float(row["final_coverage"]) == 0.99 for row in rows)
        )

    def test_trajectory_path_length(self):
        positions = [
            (0.0, 0.0),
            (3.0, 4.0),
            (3.0, 8.0),
        ]
        self.assertAlmostEqual(trajectory_path_length(positions), 9.0)
        self.assertEqual(trajectory_path_length([]), 0.0)
        self.assertEqual(trajectory_path_length([(2.0, 3.0)]), 0.0)

    def test_coverage_success_boundary(self):
        self.assertTrue(coverage_success(0.99, 0.99))
        self.assertTrue(coverage_success(1.0, 0.99))
        self.assertFalse(coverage_success(0.989999, 0.99))

    def test_single_sample_variance_is_nan(self):
        self.assertTrue(
            math.isnan(aggregate_results.sample_variance([3.0]))
        )
        self.assertAlmostEqual(
            aggregate_results.sample_variance([1.0, 3.0]),
            2.0,
        )

    def test_coverage_threshold_step_statistics_use_first_crossing(self):
        checkpoint_sha256 = "a" * 64
        episodes = [
            {
                "model_name": "threshold-policy",
                "model_seed": 0,
                "checkpoint": "/models/threshold.pth",
                "checkpoint_sha256": checkpoint_sha256,
                "map_id": "map.png",
                "eval_seed": 100 + episode_id,
                "episode_id": episode_id,
            }
            for episode_id in range(2)
        ]
        coverage_by_episode = (
            ((0, 0.10), (2, 0.65), (4, 0.82), (8, 0.96)),
            ((0, 0.10), (1, 0.60), (3, 0.75), (6, 0.91), (8, 0.94)),
        )
        coverage_rows = []
        for episode_id, points in enumerate(coverage_by_episode):
            for step, coverage in points:
                coverage_rows.append(
                    {
                        "_source_id": "threshold-fixture",
                        "model_name": "threshold-policy",
                        "model_seed": 0,
                        "checkpoint": "/models/threshold.pth",
                        "checkpoint_sha256": checkpoint_sha256,
                        "map_id": "map.png",
                        "eval_seed": 100 + episode_id,
                        "episode_id": episode_id,
                        "step": step,
                        "coverage": coverage,
                    }
                )

        rows = aggregate_results.build_coverage_threshold_summary(
            episodes,
            coverage_rows,
        )
        by_threshold = {
            row["coverage_threshold"]: row for row in rows
        }
        self.assertEqual(set(by_threshold), {0.60, 0.70, 0.80, 0.90, 0.95})
        self.assertAlmostEqual(
            by_threshold[0.60]["mean_steps_to_threshold"],
            1.5,
        )
        self.assertAlmostEqual(
            by_threshold[0.60]["variance_steps_to_threshold"],
            0.5,
        )
        self.assertAlmostEqual(
            by_threshold[0.80]["mean_steps_to_threshold"],
            5.0,
        )
        self.assertAlmostEqual(
            by_threshold[0.80]["variance_steps_to_threshold"],
            2.0,
        )
        self.assertEqual(by_threshold[0.95]["num_reached"], 1)
        self.assertEqual(by_threshold[0.95]["reach_rate"], 0.5)
        self.assertEqual(
            by_threshold[0.95]["mean_steps_to_threshold"],
            8.0,
        )
        self.assertTrue(
            math.isnan(
                by_threshold[0.95]["variance_steps_to_threshold"]
            )
        )

    def test_all_failure_aggregation_does_not_crash(self):
        evaluation_dir = self.root / "all_failures"
        evaluation_dir.mkdir()
        episodes_path = evaluation_dir / "episodes.csv"
        curve_path = evaluation_dir / "coverage_curve.csv"

        episode_fields = [
            "model_name",
            "checkpoint",
            "map_id",
            "model_seed",
            "eval_seed",
            "episode_id",
            "success",
            "termination_reason",
            "path_length",
            "steps",
            "final_coverage",
            "path_length_to_target",
            "steps_to_target",
            "mean_inference_time_ms",
        ]
        with episodes_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=episode_fields)
            writer.writeheader()
            model_runs = (
                ("policy-A", "A.pth", 0.40),
                ("policy-B", "B.pth", 0.50),
            )
            for run_index, (
                model_name,
                checkpoint,
                coverage,
            ) in enumerate(model_runs):
                writer.writerow(
                    {
                        "model_name": model_name,
                        "checkpoint": checkpoint,
                        "map_id": "held_out.png",
                        "model_seed": 0,
                        "eval_seed": 100,
                        "episode_id": 0,
                        "success": False,
                        "termination_reason": "max_steps",
                        "path_length": 5.0 + run_index,
                        "steps": 2,
                        "final_coverage": coverage,
                        "path_length_to_target": "NaN",
                        "steps_to_target": "NaN",
                        "mean_inference_time_ms": 1.0,
                    }
                )

        curve_fields = [
            "model_name",
            "model_seed",
            "map_id",
            "eval_seed",
            "episode_id",
            "step",
            "path_length",
            "coverage",
        ]
        with curve_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=curve_fields)
            writer.writeheader()
            for run_index, (
                model_name,
                _checkpoint,
                coverage,
            ) in enumerate(model_runs):
                writer.writerow(
                    {
                        "model_name": model_name,
                        "model_seed": 0,
                        "map_id": "held_out.png",
                        "eval_seed": 100,
                        "episode_id": 0,
                        "step": 0,
                        "path_length": 0.0,
                        "coverage": 0.10,
                    }
                )
                writer.writerow(
                    {
                        "model_name": model_name,
                        "model_seed": 0,
                        "map_id": "held_out.png",
                        "eval_seed": 100,
                        "episode_id": 0,
                        "step": 2,
                        "path_length": 5.0 + run_index,
                        "coverage": coverage,
                    }
                )

        output_dir = self.root / "all_failure_summary"
        with mock.patch.object(
            sys,
            "stderr",
            new_callable=io.StringIO,
        ) as stderr:
            outputs = aggregate_results.aggregate_results(
                [evaluation_dir],
                output_dir,
                bootstrap_seed=11,
                bootstrap_samples=20,
                curve_points=5,
                figure_dpi=40,
            )
        self.assertIn(
            "Warning: no evaluation_metadata.json was found",
            stderr.getvalue(),
        )

        with outputs["overall_summary_json"].open(
            "r",
            encoding="utf-8",
        ) as stream:
            summary = json.load(stream)
        self.assertEqual(summary["num_successes"], 0)
        self.assertEqual(summary["success_rate"], 0.0)
        self.assertEqual(summary["num_model_seeds"], 1)
        self.assertEqual(summary["num_model_runs"], 2)
        self.assertEqual(summary["num_model_names"], 2)
        self.assertIsNone(summary["mean_path_length_success"])
        self.assertIsNone(summary["variance_path_length_success"])

        with outputs["model_summary"].open(
            "r",
            encoding="utf-8",
            newline="",
        ) as stream:
            model_summaries = list(csv.DictReader(stream))
        self.assertEqual(len(model_summaries), 2)
        self.assertEqual(
            {row["model_name"] for row in model_summaries},
            {"policy-A", "policy-B"},
        )
        self.assertEqual(
            {row["model_seed"] for row in model_summaries},
            {"0"},
        )

        with outputs["map_summary"].open(
            "r",
            encoding="utf-8",
            newline="",
        ) as stream:
            map_summaries = list(csv.DictReader(stream))
        self.assertEqual(len(map_summaries), 2)

        with outputs["coverage_curve"].open(
            "r",
            encoding="utf-8",
            newline="",
        ) as stream:
            combined_curves = list(csv.DictReader(stream))
        self.assertEqual(
            {row["checkpoint"] for row in combined_curves},
            {"A.pth", "B.pth"},
        )
        self.assertTrue(outputs["path_length_boxplot"].is_file())
        self.assertTrue(outputs["coverage_threshold_summary"].is_file())
        self.assertTrue(outputs["coverage_threshold_steps_bar"].is_file())

    def test_aggregation_metadata_compatibility_is_enforced(self):
        grid = [
            ("held_out.png", 100, 0),
            ("held_out.png", 101, 1),
        ]
        metadata_a = self._compatible_evaluation_metadata("a" * 64)
        metadata_b = self._compatible_evaluation_metadata("b" * 64)
        evaluation_a = self.root / "metadata_compatible_a"
        evaluation_b = self.root / "metadata_compatible_b"
        self._write_aggregation_fixture(
            evaluation_a,
            model_name="shared-policy",
            checkpoint="/models/A.pth",
            model_seed=0,
            episode_grid=grid,
            metadata=metadata_a,
        )
        self._write_aggregation_fixture(
            evaluation_b,
            model_name="shared-policy",
            checkpoint="/models/B.pth",
            model_seed=0,
            episode_grid=grid,
            metadata=metadata_b,
        )

        episodes, curves, _ = aggregate_results.load_evaluation_inputs(
            [evaluation_a, evaluation_b]
        )
        self.assertEqual(
            {row["checkpoint_sha256"] for row in episodes},
            {"a" * 64, "b" * 64},
        )
        self.assertEqual(
            {row["checkpoint_sha256"] for row in curves},
            {"a" * 64, "b" * 64},
        )
        self.assertEqual(
            {row["checkpoint"] for row in curves},
            {"/models/A.pth", "/models/B.pth"},
        )
        model_summaries = aggregate_results.build_model_summary(
            episodes,
            tracker=aggregate_results.VarianceTracker(),
            bootstrap_samples=20,
            bootstrap_seed=17,
            confidence_level=0.95,
        )
        self.assertEqual(len(model_summaries), 2)
        self.assertEqual(
            {row["checkpoint_sha256"] for row in model_summaries},
            {"a" * 64, "b" * 64},
        )

        mismatched = self.root / "metadata_mismatched"
        self._write_aggregation_fixture(
            mismatched,
            model_name="policy-C",
            checkpoint="/models/C.pth",
            model_seed=0,
            episode_grid=grid,
            metadata=self._compatible_evaluation_metadata(
                "c" * 64,
                target_coverage=0.90,
            ),
        )
        with self.assertRaisesRegex(
            aggregate_results.AggregationError,
            "target_coverage",
        ):
            aggregate_results.load_evaluation_inputs(
                [evaluation_a, mismatched]
            )

        legacy = self.root / "metadata_partially_missing"
        self._write_aggregation_fixture(
            legacy,
            model_name="legacy-policy",
            checkpoint="/models/legacy.pth",
            model_seed=0,
            episode_grid=grid,
            metadata=None,
        )
        with self.assertRaisesRegex(
            aggregate_results.AggregationError,
            "Every evaluation input must include evaluation_metadata.json",
        ):
            aggregate_results.load_evaluation_inputs(
                [evaluation_a, legacy]
            )

        wrong_split = self.root / "metadata_wrong_split"
        self._write_aggregation_fixture(
            wrong_split,
            model_name="policy-D",
            checkpoint="/models/D.pth",
            model_seed=0,
            episode_grid=grid,
            metadata=self._compatible_evaluation_metadata(
                "d" * 64,
                split_name="train",
            ),
        )
        with self.assertRaisesRegex(
            aggregate_results.AggregationError,
            "split_name.*must be 'test'",
        ):
            aggregate_results.load_evaluation_inputs([wrong_split])

        invalid_count = self.root / "metadata_invalid_count"
        self._write_aggregation_fixture(
            invalid_count,
            model_name="policy-E",
            checkpoint="/models/E.pth",
            model_seed=0,
            episode_grid=grid,
            metadata=self._compatible_evaluation_metadata(
                "e" * 64,
                num_episodes=1,
            ),
        )
        with self.assertRaisesRegex(
            aggregate_results.AggregationError,
            "num_episodes=.*does not equal num_maps",
        ):
            aggregate_results.load_evaluation_inputs([invalid_count])

        wrong_schedule = self.root / "metadata_wrong_schedule"
        self._write_aggregation_fixture(
            wrong_schedule,
            model_name="policy-F",
            checkpoint="/models/F.pth",
            model_seed=0,
            episode_grid=[
                ("held_out.png", 101, 0),
                ("held_out.png", 100, 1),
            ],
            metadata=self._compatible_evaluation_metadata("6" * 64),
        )
        with self.assertRaisesRegex(
            aggregate_results.AggregationError,
            "does not realize the metadata evaluation schedule",
        ):
            aggregate_results.load_evaluation_inputs([wrong_schedule])

    def test_episode_grid_and_curve_references_are_validated(self):
        full_grid = [
            ("held_out.png", 100, 0),
            ("held_out.png", 101, 1),
        ]
        short_grid = [("held_out.png", 100, 0)]
        grid_a = self.root / "grid_model_a"
        grid_b = self.root / "grid_model_b"
        self._write_aggregation_fixture(
            grid_a,
            model_name="grid-policy-A",
            checkpoint="/models/grid-A.pth",
            model_seed=0,
            episode_grid=full_grid,
            metadata=self._compatible_evaluation_metadata("1" * 64),
        )
        self._write_aggregation_fixture(
            grid_b,
            model_name="grid-policy-B",
            checkpoint="/models/grid-B.pth",
            model_seed=0,
            episode_grid=short_grid,
            metadata=self._compatible_evaluation_metadata("2" * 64),
        )
        with self.assertRaisesRegex(
            aggregate_results.AggregationError,
            "Episode grids differ across model runs",
        ):
            aggregate_results.load_evaluation_inputs([grid_a, grid_b])

        duplicate = self.root / "duplicate_episode_identity"
        self._write_aggregation_fixture(
            duplicate,
            model_name="duplicate-policy",
            checkpoint="/models/duplicate.pth",
            model_seed=0,
            episode_grid=[full_grid[0], full_grid[0]],
            metadata=self._compatible_evaluation_metadata("3" * 64),
        )
        with self.assertRaisesRegex(
            aggregate_results.AggregationError,
            "duplicate episode identity",
        ):
            aggregate_results.load_evaluation_inputs([duplicate])

        orphan_curve = self.root / "orphan_coverage_curve"
        self._write_aggregation_fixture(
            orphan_curve,
            model_name="orphan-policy",
            checkpoint="/models/orphan.pth",
            model_seed=0,
            episode_grid=full_grid,
            curve_grid=[
                full_grid[0],
                ("unknown.png", 101, 1),
            ],
            metadata=self._compatible_evaluation_metadata("4" * 64),
        )
        with self.assertRaisesRegex(
            aggregate_results.AggregationError,
            "coverage curve row references no episode identity",
        ):
            aggregate_results.load_evaluation_inputs([orphan_curve])

        missing_curve = self.root / "missing_coverage_curve_episode"
        self._write_aggregation_fixture(
            missing_curve,
            model_name="missing-curve-policy",
            checkpoint="/models/missing-curve.pth",
            model_seed=0,
            episode_grid=full_grid,
            curve_grid=[full_grid[0]],
            metadata=self._compatible_evaluation_metadata("5" * 64),
        )
        with self.assertRaisesRegex(
            aggregate_results.AggregationError,
            "coverage curve is missing for non-exception episode",
        ):
            aggregate_results.load_evaluation_inputs([missing_curve])

        exception_without_curve = self.root / "exception_without_curve"
        self._write_aggregation_fixture(
            exception_without_curve,
            model_name="exception-policy",
            checkpoint="/models/exception.pth",
            model_seed=0,
            episode_grid=full_grid,
            curve_grid=[full_grid[0]],
            exception_grid=[full_grid[1]],
            metadata=self._compatible_evaluation_metadata("7" * 64),
        )
        episodes, curves, _ = aggregate_results.load_evaluation_inputs(
            [exception_without_curve]
        )
        self.assertEqual(len(episodes), 2)
        self.assertEqual(len(curves), 1)

        start_seed_a = self.root / "start_seed_model_a"
        start_seed_b = self.root / "start_seed_model_b"
        self._write_aggregation_fixture(
            start_seed_a,
            model_name="start-policy-A",
            checkpoint="/models/start-A.pth",
            model_seed=0,
            episode_grid=full_grid,
            metadata=self._compatible_evaluation_metadata("8" * 64),
        )
        self._write_aggregation_fixture(
            start_seed_b,
            model_name="start-policy-B",
            checkpoint="/models/start-B.pth",
            model_seed=0,
            episode_grid=full_grid,
            metadata=self._compatible_evaluation_metadata("9" * 64),
            start_seed_offset=1,
        )
        with self.assertRaisesRegex(
            aggregate_results.AggregationError,
            "start_position_seed differs across model runs",
        ):
            aggregate_results.load_evaluation_inputs(
                [start_seed_a, start_seed_b]
            )

        start_position_b = self.root / "start_position_model_b"
        self._write_aggregation_fixture(
            start_position_b,
            model_name="start-policy-B",
            checkpoint="/models/start-B.pth",
            model_seed=0,
            episode_grid=full_grid,
            metadata=self._compatible_evaluation_metadata("a" * 64),
            start_coordinate_offset=0.5,
        )
        with self.assertRaisesRegex(
            aggregate_results.AggregationError,
            "start_x/start_y differ across model runs",
        ):
            aggregate_results.load_evaluation_inputs(
                [start_seed_a, start_position_b]
            )

        missing_start_seed = self.root / "missing_start_seed"
        self._write_aggregation_fixture(
            missing_start_seed,
            model_name="missing-start-policy",
            checkpoint="/models/missing-start.pth",
            model_seed=0,
            episode_grid=full_grid,
            metadata=self._compatible_evaluation_metadata("0" * 64),
            start_seed_offset=None,
        )
        with self.assertRaisesRegex(
            aggregate_results.AggregationError,
            "missing required start_position_seed",
        ):
            aggregate_results.load_evaluation_inputs([missing_start_seed])

    def test_missing_checkpoint_map_and_split_have_clear_errors(self):
        missing_checkpoint = self.root / "missing_checkpoint.pth"
        with self.assertRaisesRegex(
            FileNotFoundError,
            "checkpoint does not exist",
        ):
            evaluate_on_split.run_evaluation(
                self._evaluation_args(
                    checkpoint=missing_checkpoint,
                    split_file=self.manifest_path,
                    output_dir=self.root / "missing_checkpoint_output",
                )
            )

        checkpoint = self.root / "present_checkpoint.pth"
        checkpoint.write_bytes(b"present")
        missing_split = self.root / "missing_split.json"
        with self.assertRaisesRegex(
            FileNotFoundError,
            "split manifest does not exist",
        ):
            evaluate_on_split.run_evaluation(
                self._evaluation_args(
                    checkpoint=checkpoint,
                    split_file=missing_split,
                    output_dir=self.root / "missing_split_output",
                )
            )

        missing_map_root = self.root / "missing_map_dataset"
        missing_map_root.mkdir()
        self._write_map(
            missing_map_root / "train.png",
            self.base_masks[1],
        )
        missing_map_manifest = self.root / "missing_map.json"
        split_maps.write_manifest(
            missing_map_manifest,
            {
                "dataset_root": str(missing_map_root),
                "train_maps": ["train.png"],
                "test_maps": ["missing.png"],
            },
        )
        with self.assertRaisesRegex(
            FileNotFoundError,
            "manifest map does not exist",
        ):
            load_split_manifest(
                missing_map_manifest,
                "test",
                require_complete_dataset=False,
            )


if __name__ == "__main__":
    unittest.main()
