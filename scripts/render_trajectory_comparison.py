"""Draw aligned exploration trajectories from replay_exploration_trajectory.py."""
import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from replay_exploration_trajectory import MANIFEST, PROJECT, REFERENCES, read_csv


MODELS = {
    "hera": "HERA", "ariadne": "ARiADNE", "dsvp": "DSVP",
    "cnn": "CNN", "nbvp": "RH-NBVP",
}
PREFIX_MODELS = {"dsvp", "nbvp"}


def load_trajectory(key, source):
    path = source / f"{key}.json"
    if not path.is_file():
        return None
    metadata = json.loads(path.read_text())
    # Classical planning samples from the original split's continuous stream.
    if key in PREFIX_MODELS and metadata["rng_scope"] != "original_test_order":
        return None
    if key != "cnn" and metadata["matches_published_metrics"] is not True:
        raise ValueError(f"{key} did not reproduce its published metrics")
    rows = read_csv(source / f"{key}_trajectory.csv")
    episode = metadata["episode"]
    endpoints = np.array([(float(r["column"]), float(r["row"])) for r in rows])
    if (
        episode["termination_reason"] != "target_reached"
        or len(rows) != episode["steps"] + 1
        or not np.allclose(endpoints[0], metadata["start_cell"])
        or abs(float(rows[-1]["coverage"]) - episode["final_coverage"]) > 1e-8
        or abs(float(rows[-1]["path_length_m"]) - episode["path_length"]) > 1e-8
    ):
        raise ValueError(f"{key} recorded episode differs from its trajectory")
    first_75 = next(i for i, row in enumerate(rows) if float(row["coverage"]) >= 0.75)
    return metadata, endpoints, rows, float(rows[first_75]["path_length_m"])


def validate_curve(key, map_id, source):
    with (REFERENCES[key] / "coverage_curve.csv").open(newline="") as stream:
        original = [row for row in csv.DictReader(stream) if row["map_id"] == map_id]
    replay = read_csv(source / f"{key}_trajectory.csv")
    expected = [(float(row["path_length"]), float(row["coverage"])) for row in original]
    observed = [(float(row["path_length_m"]), float(row["coverage"])) for row in replay]
    if len(expected) != len(observed) or not np.allclose(expected, observed, atol=1e-8, rtol=0):
        raise ValueError(f"{key} does not reproduce the published step-by-step coverage curve")


def gif_artists(ax, belief):
    """Match the trajectory panel in Env.plot_env."""
    image = ax.imshow(belief, cmap="gray", vmin=1, vmax=255, origin="upper",
                      interpolation="nearest")
    ax.axis("off")
    line, = ax.plot([], [], "b", linewidth=2, zorder=1)
    marker, = ax.plot([], [], "mo", markersize=4, zorder=5)
    return image, line, marker


def update_artists(artists, belief, points):
    image, line, marker = artists
    image.set_data(belief)
    line.set_data(*points.T)
    marker.set_data(*points[-1:].T)


def gif_title(coverage, distance):
    return f"Explored ratio: {coverage:.4g}  Travel distance: {distance:.4g}"


def render_run(key, trajectory, path, staging):
    from env import Env

    metadata, points, rows, _ = trajectory
    environment = Env(0, plot=False, map_path=path, start_seed=metadata["seed"])
    if list(environment.start_cell) != metadata["start_cell"]:
        raise ValueError(f"{key} GIF rendering has a different starting cell")
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.91)
    artists = gif_artists(ax, environment.robot_belief)
    title = fig.suptitle("")
    frames = []
    for step, row in enumerate(rows):
        if step:
            environment.step(np.array([float(row["x_m"]), float(row["y_m"])]))
        coverage = float(environment.get_exploration_rate())
        distance = float(environment.travel_dist)
        if not np.allclose([coverage, distance],
                           [float(row["coverage"]), float(row["path_length_m"])],
                           rtol=0, atol=1e-8):
            raise ValueError(f"{key} GIF rendering diverged from the recorded rollout at step {step}")
        update_artists(artists, environment.robot_belief, points[:step + 1])
        title.set_text(gif_title(coverage, distance))
        fig.canvas.draw()
        frame = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB")
        frames.append(frame.quantize(colors=128))
    frame.save(staging / f"{key}.png")
    frames[0].save(staging / f"{key}.gif", save_all=True, append_images=frames[1:],
                   duration=500, loop=0, disposal=2)
    plt.close(fig)
    return environment.robot_belief


def render(source, destination, required):
    data = {key: load_trajectory(key, source) for key in MODELS}
    ready = {key: value for key, value in data.items() if value is not None}
    if not ready or (required and len(ready) != len(MODELS)):
        raise ValueError("Missing verified trajectories")
    reference = next(iter(ready.values()))[0]
    identity_fields = ("map_id", "map_sha256", "seed", "start_cell", "cell_size_m", "config")
    for key, trajectory in ready.items():
        metadata = trajectory[0]
        if any(metadata[key] != reference[key] for key in identity_fields):
            raise ValueError("The trajectories have different map or test conditions")
        if key in REFERENCES:
            validate_curve(key, metadata["map_id"], source)
    manifest = json.loads(MANIFEST.read_text())
    path = (MANIFEST.parent / manifest["dataset_root"] / reference["map_id"]).resolve()
    if sha256(path.read_bytes()).hexdigest() != reference["map_sha256"]:
        raise ValueError("Map raster changed after the trajectory replay")
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    if str(PROJECT) not in sys.path:
        sys.path.insert(0, str(PROJECT))

    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".render-", dir=destination) as directory:
        staging = Path(directory)
        beliefs = {key: render_run(key, value, path, staging) for key, value in ready.items()}
        if len(ready) == len(MODELS):
            figure, axes = plt.subplots(2, 3, figsize=(10.2, 5.4))
            figure.subplots_adjust(left=0.035, right=0.965, top=0.86, bottom=0.05,
                                   hspace=0.47, wspace=0.11)
            for (key, value), ax in zip(ready.items(), axes.flat):
                metadata, points, _, _ = value
                update_artists(gif_artists(ax, beliefs[key]), beliefs[key], points)
                episode = metadata["episode"]
                ax.set_title(f'{MODELS[key]}\n{gif_title(episode["final_coverage"], episode["path_length"])}',
                             fontsize=8)
            axes.flat[-1].axis("off")
            figure.text(0.046, 0.966, f'{reference["map_id"]}   ·   Seed {reference["seed"]}',
                        va="top", fontsize=10, color="#263445")
            for extension in ("png", "pdf"):
                figure.savefig(staging / f"trajectory_comparison.{extension}", dpi=320)
            plt.close(figure)

        with (staging / "episode_metrics.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("model", "map_id", "seed", "steps", "final_coverage",
                             "path_length_m", "path_to_75_m", "path_from_75_to_99_m"))
            for key, (metadata, _, _, distance_to_75) in ready.items():
                episode = metadata["episode"]
                path_length = episode["path_length"]
                writer.writerow((MODELS[key], metadata["map_id"], metadata["seed"], episode["steps"],
                                 episode["final_coverage"], path_length, distance_to_75,
                                 path_length - distance_to_75))
        trajectories = destination / "trajectories"
        trajectories.mkdir(exist_ok=True)
        for key in ready:
            for filename in (f"{key}.json", f"{key}_trajectory.csv"):
                temporary = staging / filename
                shutil.copy2(source / filename, temporary)
                temporary.replace(trajectories / filename)
        for file in staging.iterdir():
            file.replace(destination / file.name)
    print(f"Exported {len(ready)}/{len(MODELS)} trajectories to {destination}", flush=True)
    return len(ready)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--partial", action="store_true")
    parser.add_argument("--wait-for", nargs="*", default=[], help="User services producing replay CSVs")
    args = parser.parse_args()
    count = render(args.source, args.destination, required=not (args.partial or args.wait_for))
    while args.wait_for and count < len(MODELS):
        for service in args.wait_for:
            model = service.split("-")[-2]
            if load_trajectory(model, args.source) is not None:
                continue
            result = subprocess.run(["systemctl", "--user", "is-active", "--quiet", service])
            if result.returncode and load_trajectory(model, args.source) is None:
                raise RuntimeError(f"{service} stopped before publishing a verified trajectory")
        time.sleep(30)
        ready = sum(load_trajectory(model, args.source) is not None for model in MODELS)
        if ready > count:
            count = render(args.source, args.destination, required=False)


if __name__ == "__main__":
    main()
