# ARiADNE + Mamba 1.1
Public code and model of <a href="https://arxiv.org/pdf/2301.11575.pdf">ARiADNE: A Reinforcement learning approach using Attention-based Deep Networks for Exploration</a>, which is accepted for the oral presentation at ICRA 2023.

**Note**: This is a new implementation of ARiADNE. 
You can find our original implementation in the [main branch](https://github.com/marmotlab/ARiADNE/tree/main).
We reimplement ARiADNE to optimize the computing time, RAM/VRAM usage, and compatibility with ROS. 
The upstream ARiADNE trained model can be tested in the
[ARiADNE ROS planner](https://github.com/marmotlab/ARiADNE-ROS-Planner);
this repository intentionally does not bundle pretrained weights.

## Current model state

The active implementation is the source code from
[`ariadne_mamba_1.1_trained`](https://github.com/caibingdao/ARiADNE_MAMBA/tree/ariadne_mamba_1.1_trained)
at commit `1e8316b9739e2b46b8aaaa4d2be69f175206f207`, merged with this
repository's reproducible map-split experiment interfaces.

Only source code was imported. No pretrained checkpoint, optimizer state,
TensorBoard event, or training-generated image was copied. `LOAD_MODEL=False`
in `parameter.py`, so both `python driver.py` and
`scripts/train_on_split.py` start from randomly initialized weights. The split
wrapper additionally rejects a non-empty output directory. The active run name
is `ariadne1_mamba_1.1`; generated checkpoints and logs are ignored by Git.

There is therefore no bundled model available for testing immediately. Train
first, then pass the resulting `final_checkpoint.pth` to the evaluation
command below.


## Run

#### Dependencies
We recommend to use conda for package management. 
Our planner is coded in Python and based on Pytorch. 
Other than Pytorch, please install following packages by:
```
pip install scikit-image matplotlib ray tensorboard
```
We tested our planner in various version of these packages so you can just install the latest one.

#### Training
Download this repo and go into the folder:
```
git clone https://github.com/caibingdao/ARiADNE_MAMBA_Test.git
cd ARiADNE_MAMBA_Test
```
Launch your conda environment if any and run:

```bash
python driver.py
```

The default training code requires around 8GB VRAM and 20G RAM. 
You can modify the hyperparameters in `parameter.py`.
This direct command trains on the legacy map sampler with no fixed
train/test split. For a controlled experiment, use
`scripts/train_on_split.py` below.

## Reproducible map-split experiments

The experiment tools in `scripts/` provide a fixed train/test split, fresh
training runs, held-out evaluation, statistics, confidence intervals, and
plots. They use the maps in this repository's `maps/` directory. The existing
`python driver.py` command keeps its original no-manifest, unlimited-training
behavior.

The current map audit found 5,663 PNG files. Four groups are exact copies:
`202.png/203.png`, `695.png/696.png`, `733.png/734.png`, and
`735.png/736.png/737.png`. The split tool keeps every file but assigns each
copy/variant group wholly to train or test. It found no
additional rotation, reflection, resolution, or start-position variants and
no explicit difficulty labels, so the current dataset uses seeded group random
splitting rather than invented strata.

### 1. Create the fixed split

```bash
python scripts/split_maps.py \
  --map-root maps \
  --train-ratio 0.8 \
  --seed 42 \
  --output experiment/map_splits/map_split_seed_42.json \
  --group-by-base-map \
  --stratify
```

The JSON stores relative map paths, content hashes, base-map group identities,
split statistics, and whether stratification was actually applied. It validates
that every map exists, every discovered map appears exactly once, train/test do
not overlap, and no base-map group crosses the boundary. Re-running with the
same inputs and seed produces byte-identical JSON.

### 2. Train only on training maps

Set `USE_MAMBA` in `parameter.py` before launching a process. When it is
`False`, the Mamba rollout-history cache is disabled automatically; the model,
reward, replay buffer, optimizers, and SAC update equations are otherwise
unchanged.

```bash
python scripts/train_on_split.py \
  --split-file experiment/map_splits/map_split_seed_42.json \
  --split-name train \
  --training-seed 0 \
  --total-env-steps 1000000 \
  --output-dir experiment/checkpoints/seed_0
```

The split wrapper always starts a fresh run and refuses a non-empty output
directory, preventing old checkpoints or TensorBoard events from being mixed
into a new experiment. It writes `checkpoint.pth`, `final_checkpoint.pth`,
`training_config.json`, and TensorBoard events below the output directory.
Checkpoint metadata includes the Git commit, manifest path and hash, split
seed, training seed, requested/completed environment steps, timestamp, model
variant, and main hyperparameters. The final checkpoint is selected by the
exact environment-step budget only; held-out test results are never used for
selection.

Python, NumPy, PyTorch, CUDA, and per-episode rollout seeds are initialized from
the training seed. Deterministic PyTorch kernels are requested in warn-only
mode. Ray workers finish asynchronously, so exact bitwise identity across
different machines or scheduling orders is not guaranteed; the training
command prints this limitation.

#### Resume an interrupted training run

Use the most recent `checkpoint.pth` from the interrupted output directory.
Keep `USE_MAMBA` in `parameter.py` consistent with the checkpoint. When
`--output-dir` is omitted, the checkpoint's parent directory is reused:

```bash
python scripts/train_on_split.py \
  --split-file experiment/map_splits/map_split_seed_42.json \
  --split-name train \
  --training-seed 0 \
  --total-env-steps 1000000 \
  --resume-checkpoint experiment/checkpoints/seed_0/checkpoint.pth
```

The total step argument remains the final target, not an additional number of
steps. The resume path restores policy, critics, target critics, optimizers,
entropy state, counters, and saved random-number-generator state. The replay
buffer is not stored in the current checkpoint format, so it is rebuilt from
new training episodes after restart; SAC updates resume after the minimum
buffer size is reached. This is a practical continuation but not a bitwise
identical uninterrupted run.

### 3. Evaluate the held-out maps

```bash
python scripts/evaluate_on_split.py \
  --checkpoint experiment/checkpoints/seed_0/final_checkpoint.pth \
  --split-file experiment/map_splits/map_split_seed_42.json \
  --split-name test \
  --eval-seeds 100 101 102 103 104 \
  --episodes-per-map 5 \
  --target-coverage 0.99 \
  --max-steps 128 \
  --output-dir experiment/results/seed_0 \
  --deterministic \
  --fail-fast
```

Add `--max-path-length METRES` when a distance budget is needed. The default
contract runs exactly one episode per evaluation seed, so
`--episodes-per-map` must equal the number of unique `--eval-seeds`. The default
ARIADNE adapter loads only the policy, calls `eval()`, performs inference under
`torch.no_grad()`, and never constructs an optimizer or performs a learning or
target-network update. Deterministic evaluation uses the argmax of the SAC
policy log-probabilities. Mamba checkpoints retain the same online history
tokens used during rollout.

Evaluation writes:

- `episodes.csv`: one row per map/episode, including model/evaluation/start
  seeds, world-coordinate starts, resampling count, termination, coverage,
  distances, steps, and timing. The built-in ARIADNE adapter additionally
  records raw `start_cell_x`/`start_cell_y`; other environments may leave
  those optional columns empty;
- `coverage_curve.csv`: coverage versus decision step and traveled distance;
- `evaluation_metadata.json`: checkpoint/manifest hashes and run settings.

For parallel evaluation, run the same schedule through independent seed
workers. Each completed worker is preserved and reused after interruption;
the final files are compatibility-checked and merged atomically:

```bash
python scripts/evaluate_parallel.py \
  --checkpoint experiment/checkpoints/seed_0/final_checkpoint.pth \
  --split-file experiment/map_splits/map_split_seed_42.json \
  --eval-seeds 100 101 102 103 104 \
  --episodes-per-map 5 \
  --target-coverage 0.99 \
  --max-steps 128 \
  --workers 5 \
  --threads-per-worker 4 \
  --output-dir experiment/results/seed_0 \
  --deterministic
```

The original maps encode an 8x8 downsampled start-marker region. Legacy
training still chooses its original fixed marker cell when no experiment seed
is supplied. Split training/evaluation chooses a reproducible valid cell within
that same marker region from the recorded start seed; invalid marker cells are
resampled and counted.

### Model-independent evaluation interface

`evaluate_on_split.py` is deliberately independent of ARIADNE, SAC, PyTorch,
and the optimizer. The external checkpoint may be a file or directory in any
format understood by the selected adapter; the generic driver only checks that
it exists and records a deterministic SHA-256 of the artifact. Model-specific
behavior is selected with:

```bash
--adapter your_package.your_adapter:create_adapter
```

The factory receives `checkpoint`, `deterministic`, and a dictionary populated
by repeatable `--adapter-arg KEY=VALUE` options. It returns an object with:

```python
class Adapter:
    def metadata(self):
        return {"model_name": "my_model", "model_seed": 0}

    def evaluate_episode(
        self, *, map_path, map_id, eval_seed, start_seed, episode_id, config
    ):
        return {
            "episode": {
                "start_x": 1.0,
                "start_y": 2.0,
                "start_resample_count": 0,
                "environment_success": False,
                "termination_reason": "target_reached",
                "path_length": 12.5,
                "steps": 2,
                "final_coverage": 0.99,
                "elapsed_time_seconds": 0.20,
                "mean_inference_time_ms": 3.0,
                "max_inference_time_ms": 4.0,
            },
            "coverage_curve": [
                {"step": 0, "path_length": 0.0, "coverage": 0.1},
                {"step": 1, "path_length": 5.0, "coverage": 0.7},
                {"step": 2, "path_length": 12.5, "coverage": 0.99},
            ],
        }

    # Optional; called even if an episode raises.
    def close(self):
        pass
```

The factory and result types are documented in
`evaluation_adapters/protocol.py`. `model_name` must be non-empty and
`model_seed` a non-negative integer. The per-episode `config` contains
`target_coverage: float`, `max_steps: int`,
`max_path_length: float | None`, and `deterministic: bool`. Termination must be
one of
`target_reached`, `max_steps`, `max_path_length`, `no_valid_action`,
`environment_complete`, `environment_failure`, or `collision`.
`exception` is reserved for rows synthesized by the generic driver when an
adapter raises; adapters must not return it directly.
The coverage curve must begin at step/path length zero, contain one row for
every consecutive decision step, be non-decreasing in distance and coverage,
and end at the episode's step, distance, and coverage.

The generic driver injects model/map/seed/checkpoint identities, enforces the
common schema, defines primary success from the requested coverage threshold,
and derives the first target-crossing step and distance from the saved curve.
It requires `target_reached` exactly when coverage-based success is true and
requires the episode to stop at that first crossing. It enforces the decision
budget; a path-length budget is checked after each discrete move, so the final
edge may overshoot it but no later decision is allowed.
Consequently the same aggregation scripts can compare ARIADNE/Mamba with
unrelated models without modifying the evaluator. The built-in implementation
is `evaluation_adapters.ariadne:create_adapter`. With `--fail-fast`, any
adapter exception stops the evaluation; without it, diagnostic exception rows
are retained, and an all-failed evaluation still exits with an error.
For an older ARIADNE checkpoint that predates experiment metadata, provide its
training identity explicitly, for example
`--adapter-arg model_seed=0 --adapter-arg model_name='"ARIADNE"'`.

A third-party adapter is responsible for seeding its framework from
`eval_seed`, using `start_seed` for reproducible start selection, honoring
`deterministic` and all budgets, entering inference/evaluation mode, and never
performing optimizer, gradient, replay-buffer, target-network, or parameter
updates. The generic driver validates the observable result but cannot inspect
an arbitrary framework's internal state.

### 4. Aggregate seeds and generate plots

Evaluate training seeds 0, 1, and 2 with the same manifest and evaluation
seeds, then run:

```bash
python scripts/aggregate_results.py \
  experiment/results/seed_0 \
  experiment/results/seed_1 \
  experiment/results/seed_2 \
  --output-dir experiment/summary \
  --bootstrap-seed 42 \
  --bootstrap-samples 10000
```

Outputs are `episodes.csv`, `map_summary.csv`, `model_summary.csv`,
`overall_summary.csv`, `overall_summary.json`, `coverage_curve.csv`,
`coverage_threshold_summary.csv`, and:

- `path_length_boxplot.png`
- `final_coverage_boxplot.png`
- `success_rate_by_map.png`
- `coverage_vs_distance.png`
- `coverage_vs_step.png`
- `coverage_threshold_steps_bar.png`

`model_summary.csv` reports each independent model run, identified by model
name, training seed, and checkpoint content hash. Before aggregation, the
driver verifies that all result directories used the same held-out manifest,
evaluation settings, `(map_id, eval_seed, episode_id)` grid, start seeds, and
recorded start coordinates. Cross-model
standard deviations and bootstrap confidence intervals resample independent
model runs, not repeated episodes from one model. Pooled episode statistics
are retained separately.

### Metric definitions

- **Path length:** `Env.travel_dist`, the sum of Euclidean distances between
  successive robot waypoints. The simulator moves directly from one graph
  waypoint to the next; evaluation cross-checks this value against the saved
  robot trajectory.
- **Coverage:** `count(robot_belief == 255) / count(ground_truth == 255)`.
  Obstacles and unknown belief cells are excluded. The formula uses all
  ground-truth free cells; all current maps have one start-connected free
  component, so this equals reachable-free-space coverage for this dataset.
- **Primary success:** `final_coverage >= target_coverage` (default 0.99).
  `coverage_success` and the original environment condition
  `environment_success = (robot.utility.sum() == 0)` are both recorded.
- **Target distance/steps:** the first path length and decision step at which
  target coverage is reached; failed episodes store NaN.
- **Coverage-threshold steps:** for 60%, 70%, 80%, 90%, and 95% coverage,
  `coverage_threshold_summary.csv` reports the mean and sample variance of the
  first crossing step among episodes that reached that threshold. It also
  reports `num_reached` and `reach_rate`, so episodes that did not reach a
  threshold remain visible instead of being assigned an artificial step.
- **Termination:** distinguishes `target_reached`, `max_steps`,
  `max_path_length`, `no_valid_action`, `environment_complete`, and
  `exception`. This simulator has no collision event.
- **Variance:** sample variance with `ddof=1`; fewer than two valid samples
  yield NaN with a diagnostic. Both all-episode and successful-episode path
  statistics are always present.

Confidence intervals for success rate, mean all-episode path length, and mean
final coverage use a fixed-seed 95% percentile bootstrap. Coverage plots show
the mean and sample-standard-deviation band and are generated from saved CSV
rows, not by rerunning evaluation.

### One-command multi-seed run

Activate the intended environment, review the settings at the top of the
script (especially `TOTAL_ENV_STEPS`), then run:

```bash
bash scripts/run_split_experiment.sh
```

The script uses `set -euo pipefail`, stops at the failing stage, trains seeds
0/1/2, locates each final checkpoint, evaluates all test maps with the same
seeds, and generates the combined summaries and figures. Paths are
project-relative and can be overridden with environment variables. A matching
configuration example is in `configs/map_split_evaluation.yaml`.
For another evaluator, set `EVAL_ADAPTER` and add any `KEY=JSON_VALUE`
entries to the `EVAL_ADAPTER_ARGS` array at the top of the shell script.

For a quick check:

```bash
python -m pytest -q tests/test_map_split_experiment.py
```


## Files
* `parameter.py` Training parameters.
* `driver.py` Driver of training program, maintain & update the global network.
* `runner.py` Wrapper of the workers.
* `worker.py` Interact with environment and collect episode experience.
* `model.py` Define attention-based network.
* `env.py` Autonomous exploration environment.
* `node_manager.py` Manage and update the informative graph.
* `quads` Quad tree for node indexing provided by [Daniel Lindsley](https://github.com/toastdriven).
* `sensor.py` Simulate the sensor model of Lidar.
* `/maps` Maps of training environments provided by <a href="https://github.com/RobustFieldAutonomyLab/DRL_robot_exploration">Chen et al.</a>.

### Demo of ARiADNE

<div>
   <img src="gifs/demo_1.gif" width="360"/><img src="gifs/demo_2.gif" width="360"/>
   <img src="gifs/demo_3.gif" width="360"/><img src="gifs/demo_4.gif" width="360"/>
</div>

### Cite
If you find our work helpful or enlightening, feel free to cite our paper:
```
@INPROCEEDINGS{cao2023ariadne,
  author={Cao, Yuhong and Hou, Tianxiang and Wang, Yizhuo and Yi, Xian and Sartoretti, Guillaume},
  booktitle={2023 IEEE International Conference on Robotics and Automation (ICRA)}, 
  title={ARiADNE: A Reinforcement learning approach using Attention-based Deep Networks for Exploration}, 
  year={2023},
  pages={10219-10225},
  doi={10.1109/ICRA48891.2023.10160565}}
```

### Authors
[Yuhong Cao](https://github.com/caoyuhong001)\
Tianxiang Hou\
[Yizhuo Wang](https://github.com/wyzh98)\
Xian Yi\
[Guillaume Sartoretti](https://github.com/gsartoretti)
