"""Record the executed trajectory of a seed-4777 exploration episode."""
import argparse
import csv
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile


WORKSPACE = Path(__file__).resolve().parents[2]
PROJECT = WORKSPACE / "ARiADNE_MAMBA_Test/mamba31_map_split"
MANIFEST = PROJECT / "experiment/map_splits/map_split_seed_42.json"
TOKEN_RESULTS = WORKSPACE / "ARiADNE_MAMBA_3.1_TOKEN21/results/data"
REFERENCES = {
    "hera": WORKSPACE / "HERA/results/data/recovery_streak2/recovery_on/single_seed_4777",
    "ariadne": TOKEN_RESULTS / "ariadne/single_seed_greedy_4777",
    "dsvp": TOKEN_RESULTS / "dsvp/default/single_seed_4777",
    "nbvp": WORKSPACE / "ARiADNE_NBVP_OFFICIAL_2D/results/data/single_seed_4777",
}
CNN_CHECKPOINT = WORKSPACE / (
    "training_runs/cnn_pure_recovery_off_10000_seed4777_20261118/model/checkpoint.pth"
)
SEED = 4777
CONFIG = dict(target_coverage=0.99, max_steps=128, max_path_length=None, deterministic=True)


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def load_reference(model, map_id):
    source = REFERENCES.get(model)
    if source is None:
        return None, None
    metadata = json.loads((source / "evaluation_metadata.json").read_text())
    row = next(row for row in read_csv(source / "episodes.csv") if row["map_id"] == map_id)
    return metadata, row


def configure_hera(checkpoint, manifest_path):
    from run_full_test_eval import _configure_recovery_from_checkpoint, _install_token21_compatibility

    _configure_recovery_from_checkpoint(checkpoint)
    os.environ["ARIADNE_MAP_SPLIT_FILE"] = str(manifest_path)
    os.environ["ARIADNE_MAP_SPLIT_NAME"] = "train"
    os.environ["ARIADNE_USE_RECOVERY"] = "1"
    _install_token21_compatibility()


def replay(model, map_id, output, replay_prefix):
    sys.path[:0] = [str(PROJECT), str(WORKSPACE)]
    metadata, reference = load_reference(model, map_id)
    checkpoint = Path(metadata["checkpoint"]) if metadata else CNN_CHECKPOINT
    factory = metadata["adapter"] if metadata else "evaluation_adapters.cnn_pure:create_adapter"
    adapter_args = metadata["adapter_args"].copy() if metadata else {"device": "cpu"}
    manifest = json.loads(MANIFEST.read_text())
    root = (MANIFEST.parent / manifest["dataset_root"]).resolve()
    map_ids = manifest["test_maps"]
    index = map_ids.index(map_id)

    with tempfile.TemporaryDirectory(prefix="trajectory_replay_") as directory:
        temporary = Path(directory)
        if model == "hera":
            compatible = temporary / "test_maps.json"
            compatible.write_text(json.dumps({"train_maps": [str(root / map_id)]}))
            configure_hera(checkpoint, compatible)
        if model == "nbvp":
            library = temporary / "visibility.so"
            subprocess.run(
                ["g++", "-O3", "-std=c++17", "-fPIC", "-shared",
                 str(WORKSPACE / "ARiADNE_NBVP_OFFICIAL_2D/visibility.cpp"), "-o", str(library)],
                check=True,
            )
            os.environ["RH_NBVP_GEOMETRY_LIB"] = str(library)

        from experiment_utils import seed_everything, sha256_file
        import numpy as np

        digest = sha256_file(checkpoint)
        if metadata and (
            sha256_file(MANIFEST) != metadata["split_manifest_sha256"]
            or digest != metadata["checkpoint_sha256"]
            or any(metadata[k] != v for k, v in CONFIG.items())
            or metadata["global_seed"] != SEED
        ):
            raise ValueError(f"{model} checkpoint or protocol differs from the published run")

        module_name, factory_name = factory.split(":")
        module = importlib.import_module(module_name)
        original_env = module.Env
        captured = None

        class RecordingEnv(original_env):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                if Path(kwargs["map_path"]).name == map_id:
                    nonlocal captured
                    captured = self

        module.Env = RecordingEnv
        adapter = getattr(module, factory_name)(
            checkpoint=checkpoint, deterministic=True, adapter_args=adapter_args
        )
        policy_metadata = adapter.metadata()
        if model == "cnn" and (
            policy_metadata["completed_training_episodes"] != 10000
            or policy_metadata["recovery_enabled"] or policy_metadata["training_recovery_enabled"]
        ):
            raise ValueError("Expected the pure CNN checkpoint after 10,000 episodes")
        seed_everything(SEED, deterministic=True)

        if replay_prefix:
            reference_episodes = read_csv(REFERENCES[model] / "episodes.csv")
            for count, name in enumerate(map_ids[:index], start=1):
                episode = adapter.evaluate_episode(
                    map_path=root / name, map_id=name, eval_seed=SEED,
                    start_seed=SEED, episode_id=0, config=CONFIG,
                )["episode"]
                expected = reference_episodes[count - 1]
                if expected["map_id"] != name or not metrics_match(episode, expected):
                    raise ValueError(f"{model} prefix diverged at {count}/{index}: {name}")
                if count % 10 == 0:
                    print(f"{model}: replayed {count}/{index} preceding maps", flush=True)

        result = adapter.evaluate_episode(
            map_path=root / map_id, map_id=map_id, eval_seed=SEED,
            start_seed=SEED, episode_id=0, config=CONFIG,
        )
        row = result["episode"]
        curve = result["coverage_curve"]
        trajectory = np.asarray(captured.robot_trajectory, dtype=float)
        distance = np.linalg.norm(np.diff(trajectory, axis=0), axis=1).sum()
        start = [int(row["start_cell_x"]), int(row["start_cell_y"])]
        origin = [captured.belief_origin_x, captured.belief_origin_y]
        cells = (trajectory - origin) / captured.cell_size
        if (
            len(trajectory) != len(curve) or len(trajectory) != row["steps"] + 1
            or not math.isclose(distance, row["path_length"], abs_tol=1e-8)
            or not np.allclose(cells[0], start) or row["error"]
        ):
            raise ValueError(f"{model} trajectory fails alignment or path-length validation")
        if reference and any(start[i] != int(reference[key]) for i, key in enumerate(("start_cell_x", "start_cell_y"))):
            raise ValueError(f"{model} start differs from the published run")
        matches = metrics_match(row, reference) if reference else None
        if (replay_prefix or model in {"hera", "ariadne"}) and not matches:
            raise ValueError(f"{model} published metrics differ from the replay: {row}")

        output.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": model,
            "map_id": map_id,
            "map_sha256": sha256_file(root / map_id),
            "seed": SEED,
            "start_cell": start,
            "cell_size_m": captured.cell_size,
            "map_origin_m": origin,
            "config": CONFIG,
            "checkpoint": str(checkpoint.relative_to(WORKSPACE)),
            "checkpoint_sha256": digest,
            "action_selection": policy_metadata.get("action_selection", "greedy_argmax"),
            "inference_device": policy_metadata.get("device", "cpu"),
            "rng_scope": "original_test_order" if replay_prefix else "single_map",
            "matches_published_metrics": matches,
            "published_path_length_m": float(reference["path_length"]) if reference else None,
            "episode": {key: row[key] for key in (
                "steps", "path_length", "final_coverage", "termination_reason", "recovery_enabled",
            )},
        }
        staging = output / f".{model}_trajectory.csv"
        with staging.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("step", "x_m", "y_m", "column", "row", "path_length_m", "coverage"))
            for point, cell, step in zip(trajectory, cells, curve):
                writer.writerow((step["step"], *point, *cell, step["path_length"], step["coverage"]))
        staging.replace(output / f"{model}_trajectory.csv")
        (output / f"{model}.json").write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps(payload, indent=2), flush=True)


def metrics_match(episode, reference):
    return (
        int(episode["steps"]) == int(reference["steps"])
        and episode["termination_reason"] == reference["termination_reason"]
        and all(math.isclose(float(episode[key]), float(reference[key]), rel_tol=0, abs_tol=1e-8)
                for key in ("path_length", "final_coverage"))
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("hera", "ariadne", "dsvp", "nbvp", "cnn"))
    parser.add_argument("--map", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--replay-prefix", action="store_true",
                        help="Preserve the global random stream for DSVP and RH-NBVP")
    args = parser.parse_args()
    if args.replay_prefix and args.model not in {"dsvp", "nbvp"}:
        parser.error("--replay-prefix applies to DSVP and RH-NBVP")
    replay(args.model, args.map, args.output_dir.resolve(), args.replay_prefix)


if __name__ == "__main__":
    main()
