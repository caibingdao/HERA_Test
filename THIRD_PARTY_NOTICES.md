# Third-party notices and license status

This branch is a research artifact snapshot. **No license file has been added to this branch**, and the upstream license terms were not found in the local checkouts. Do not assume that the code, adapters, or maps may be redistributed commercially or under an open-source license until the status below is confirmed.

## 1. HERA code in this branch

- The HERA implementation is derived from ARiADNE and includes Mamba temporal modeling, the 21-dimensional token, and the heuristic recovery controller.
- No `LICENSE`, `COPYING`, or `NOTICE` file was present in the source `HERA/` directory or in this branch.
- **Action required before public redistribution:** the authors must add a `LICENSE` for their own code and confirm that the chosen license is compatible with the upstream ARiADNE terms.

## 2. Upstream code

| Component | Upstream | Local reference commit | License status in local checkout |
|---|---|---|---|
| ARiADNE / HERA base | https://github.com/marmotlab/ARiADNE | HERA snapshot (no git history) | No LICENSE file found locally |
| CNN baseline | https://github.com/caibingdao/ARiADNE-new_version | `9e7f4c7` | No LICENSE file found locally |
| NBVP baseline | https://github.com/caibingdao/ARiADNE-new_version | `4019ec5` | No LICENSE file found locally |
| TARE local baseline | https://github.com/caibingdao/ARiADNE-new_version | `fde7253` | No LICENSE file found locally |
| DSVP baseline | https://github.com/HongbiaoZ/dsv_planner | `5ad721d` | No LICENSE file found locally |
| ARiADNE-Mamba lineage | https://github.com/caibingdao/ARiADNE_MAMBA | `16d30bf` | No LICENSE file found locally |

`paper/evaluator/evaluation_adapters/` contains adapter code written for the shared evaluator; it is not a copy of the complete upstream repositories.

## 3. Maps and data

- `paper/maps/test_maps/` contains the 1133 test maps used by the seed-4777 / split-seed-42 protocol. They are from the ARiADNE map set; the original map collection is attributed in the upstream ARiADNE project.
- The repository's `main` branch already contains the larger map set; this branch adds the exact 1133-map test subset for self-contained evaluation.
- **Action required:** confirm the redistribution terms of the ARiADNE map collection. If permission cannot be confirmed, remove `paper/maps/test_maps/*.png` and keep `paper/maps/test_maps.csv` plus a download/verification script instead.

## 4. What is not included

Training weights, per-step diagnostics, TensorBoard events, training/control logs, GIFs, and the 4530 training maps are not included. See `paper/docs/EXCLUSIONS.md`.

## 5. Practical suggestion

Before making this branch public and citable:

1. Add a `LICENSE` for HERA code.
2. Replace this file with a `THIRD_PARTY_NOTICES` that records the confirmed license of each upstream repository and dataset.
3. Add a `DATA_AVAILABILITY.md` (already present in this branch) stating where the maps/weights can be obtained and under what terms.
