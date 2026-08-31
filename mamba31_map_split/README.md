# ARiADNE-Mamba 3.1 map-split integration

This directory combines the reproducible map-split training/evaluation
scripts from `agent/mamba-map-split-experiments` with the policy, environment,
recovery, and rollout implementation from the local ARiADNE-Mamba 3.1
checkout. The original ARiADNE and initial Mamba logs/charts from that branch
are retained under `experiment/logs/` and `experiment/visualization/`.

The fixed manifest contains 4,530 training maps and 1,133 test maps. Each
test map is evaluated once for each of seeds 100--104 (5,665 episodes).

The default launcher performs a completely fresh random-initialized run on
the 4,530-map train split. It does not read the existing Mamba 3.1 checkpoint.
The old checkpoint is only used when `RESUME_FROM_LEGACY=1` is explicitly set;
it is never modified.

From the repository root:

```bash
cd mamba31_map_split
TOTAL_ENV_STEPS=1000000 \
  ./scripts/run_mamba31_split_experiment.sh
```

The fresh launcher refuses to use a non-empty `RUN_DIR`, preventing accidental
checkpoint reuse. To intentionally resume the old 3.1 checkpoint instead:

```bash
RESUME_FROM_LEGACY=1 TOTAL_ENV_STEPS=1000000 \
  ./scripts/run_mamba31_split_experiment.sh
```

Useful overrides:

```bash
TOTAL_ENV_STEPS=2000000 EVAL_WORKERS=5 \
  ./scripts/run_mamba31_split_experiment.sh
```

The default source checkpoint is the existing local
`ARiADNE_MAMBA_3.1/model/parallel_ariadne_mamba_3_1/checkpoint.pth`. Set
`LEGACY_CHECKPOINT` to use another one. Training writes
`experiment/checkpoints/mamba31_seed_0/final_checkpoint.pth`; evaluation
writes CSV/JSON outputs under `experiment/results/mamba31_seed_0/`.

The prior three models were trained with the legacy all-map sampler. Their
results remain useful for a same-dataset comparison, but the new 3.1 run is
the only one trained exclusively on the fixed 4,530-map training split.
