# Final-project results

This directory contains the complete reported results for the Noisy-TV
MiniGrid final project by Christian Jesinghaus.

## Stored experiments

| Directory | Environment | Methods | Seeds | Steps per run | Runs |
|---|---|---|---:|---:|---:|
| `main/` | Noisy TV enabled | DQN; RND and LP-RND with beta in `{0.01, 0.05, 0.1}` | 5 | 100,000 | 35 |
| `clean/` | Noisy TV disabled | DQN; RND and LP-RND with beta `0.01` | 5 | 100,000 | 15 |

Total: 50 completed final runs.

The three-seed pilot was used as a preliminary gate and is not part of the
final 50-run comparison.

## Artifact contents

Every run directory contains:

- `config.json`: complete experiment configuration;
- `run_state.json`: completion state and progress metadata;
- `summary.json`: final run-level summary;
- `episodes.csv`: training-episode data;
- `evaluations.csv`: greedy extrinsic evaluation results;
- `diagnostics.csv`: binned exploration and intrinsic-reward diagnostics;
- `visitation.npy`: position-visitation counts;
- `checkpoint.pt`: trained model and optimizer states.

Sweep directories additionally contain run plans, serialized configurations,
aggregate summaries, confidence intervals, plots, and heatmaps.

## Experimental environment

The experiments were executed on Ubuntu using an Intel Xeon w3-2525 CPU and
32 GiB RAM. Training was CPU-only with one PyTorch thread per run.

The 35-run main sweep used two concurrent runs (`--jobs 2`). The 15 clean
controls were executed serially.

Exact software information is stored in:

- `training-requirements.txt`;
- `python-version.txt`;
- `uv-version.txt`;
- the repository-level `uv.lock`.

## Reproduction

Install the exact locked environment from the repository root:

```bash
uv sync --extra dev --frozen
uv run pytest tests/final_project -q
```
Inspect the matrices without starting training:
```bash
uv run python -m rl_exercises.final_project.sweep main --dry-run
uv run python -m rl_exercises.final_project.sweep clean --dry-run
```

Reproduce the experiments into fresh directories:
```bash

uv run python -m rl_exercises.final_project.sweep main \
  --jobs 2 \
  --output-dir results/reproduction_main

uv run python -m rl_exercises.final_project.sweep clean \
  --output-dir results/reproduction_clean
```

Result directories are intentionally never overwritten.

## Historical note on the clean controls

The committed clean controls were originally executed as 15 individual
experiment commands before the clean sweep preset was added. Their
completeness is recorded independently in each run's run_state.json and
summary.json.

The later clean preset reproduces the same method, beta, seed, step,
evaluation, and environment matrix. Adding the preset does not modify the
stored historical artifacts.

## Main conclusion

Learning-progress gating substantially attenuated the intrinsic signal produced
by the stochastic TV observations. At beta 0.01, LP-RND performed better than
standard RND in success-rate AUC. However, it did not reliably reduce
distractor-region visitation and did not outperform the extrinsic-only DQN
baseline. The results therefore support signal attenuation, but not reliable
behavioral avoidance.