#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import os
import shutil
from collections import Counter


def _episode_id_from_path(path):
    stem = os.path.basename(path)
    return int(stem.split("_")[1])


def _copy_matches(pattern, output_directory):
    copied = []
    for source in sorted(glob.glob(pattern)):
        destination = os.path.join(output_directory, os.path.basename(source))
        shutil.copy2(source, destination)
        copied.append(os.path.basename(destination))
    return copied


def archive_recovery_episodes(run_directory, output_directory):
    event_paths = sorted(glob.glob(os.path.join(
        run_directory,
        "recovery_events",
        "episode_*_recovery.json",
    )))
    os.makedirs(output_directory, exist_ok=True)
    index = []
    for event_path in event_paths:
        with open(event_path, encoding="utf-8") as event_file:
            events = json.load(event_file)
        if not events:
            continue

        episode_id = int(
            events[0].get("episode_id", _episode_id_from_path(event_path))
        )
        episode_directory = os.path.join(
            output_directory,
            f"episode_{episode_id:06d}",
        )
        os.makedirs(episode_directory, exist_ok=True)
        event_name = os.path.basename(event_path)
        shutil.copy2(event_path, os.path.join(episode_directory, event_name))
        gifs = _copy_matches(
            os.path.join(run_directory, f"{episode_id}_explored_rate_*.gif"),
            episode_directory,
        )
        final_images = _copy_matches(
            os.path.join(run_directory, f"{episode_id}_*_samples.png"),
            episode_directory,
        )
        target_type_counts = Counter(
            event.get("target_type")
            or event.get("escape_target_type")
            or "legacy_unspecified"
            for event in events
        )
        summary = {
            "episode_id": episode_id,
            "trigger_count": len(events),
            "trigger_steps": [event.get("trigger_step") for event in events],
            "target_type_counts": dict(sorted(target_type_counts.items())),
            "target_reached_count": sum(
                bool(event.get("target_reached")) for event in events
            ),
            "retrigger_within_20_count": sum(
                bool(event.get("retrigger_within_20_steps"))
                for event in events
            ),
            "retrigger_within_30_count": sum(
                bool(event.get("retrigger_within_30_steps"))
                for event in events
            ),
            "gif_available": bool(gifs),
            "files": {
                "events": event_name,
                "gifs": gifs,
                "final_images": final_images,
            },
        }
        summary_name = f"episode_{episode_id:06d}_archive_summary.json"
        with open(
            os.path.join(episode_directory, summary_name),
            "w",
            encoding="utf-8",
        ) as summary_file:
            json.dump(summary, summary_file, ensure_ascii=False, indent=2)
        index.append({
            "episode_id": episode_id,
            "trigger_count": len(events),
            "trigger_steps": summary["trigger_steps"],
            "target_type_counts": summary["target_type_counts"],
            "target_reached_count": summary["target_reached_count"],
            "retrigger_within_20_count": summary[
                "retrigger_within_20_count"
            ],
            "retrigger_within_30_count": summary[
                "retrigger_within_30_count"
            ],
            "gif_available": bool(gifs),
            "directory": os.path.relpath(
                episode_directory,
                output_directory,
            ),
        })

    index_path = os.path.join(output_directory, "index.json")
    with open(index_path, "w", encoding="utf-8") as index_file:
        json.dump(index, index_file, ensure_ascii=False, indent=2)
    csv_path = os.path.join(output_directory, "index.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "episode_id",
            "trigger_count",
            "trigger_steps",
            "target_type_counts",
            "target_reached_count",
            "retrigger_within_20_count",
            "retrigger_within_30_count",
            "gif_available",
            "directory",
        ])
        for item in index:
            writer.writerow([
                item["episode_id"],
                item["trigger_count"],
                json.dumps(item["trigger_steps"]),
                json.dumps(item["target_type_counts"], sort_keys=True),
                item["target_reached_count"],
                item["retrigger_within_20_count"],
                item["retrigger_within_30_count"],
                item["gif_available"],
                item["directory"],
            ])

    overview = {
        "run_directory": run_directory,
        "triggered_episode_count": len(index),
        "trigger_count": sum(item["trigger_count"] for item in index),
        "episode_with_gif_count": sum(
            item["gif_available"] for item in index
        ),
        "episode_without_gif_count": sum(
            not item["gif_available"] for item in index
        ),
        "index_json": os.path.basename(index_path),
        "index_csv": os.path.basename(csv_path),
    }
    overview_path = os.path.join(output_directory, "overview.json")
    with open(overview_path, "w", encoding="utf-8") as overview_file:
        json.dump(overview, overview_file, ensure_ascii=False, indent=2)
    return overview


def main():
    parser = argparse.ArgumentParser(
        description="Collect triggered recovery episode files into one archive",
    )
    parser.add_argument("run_directory")
    parser.add_argument("output_directory")
    args = parser.parse_args()
    overview = archive_recovery_episodes(
        args.run_directory,
        args.output_directory,
    )
    print(json.dumps(overview, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
