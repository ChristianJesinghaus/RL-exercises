# Noisy-TV final project

This isolated module implements the experiment for:

> Can learning-progress-gated RND avoid noisy-TV distraction while keeping the
> exploration benefit of intrinsic rewards in sparse-reward MiniGrid?

It does not modify the weekly exercise implementations.

## Experimental contract

- Environment: a controlled `11 x 11` FourRooms task with fixed geometry,
  fixed start, and fixed goal. The route crosses two room boundaries.
- Agent observation: the standard partial `7 x 7 x 3` symbolic MiniGrid image,
  MiniGrid's direction field as a four-dimensional one-hot vector, and a
  16-dimensional irrelevant TV vector.
- Noisy TV: the vector is resampled while the agent is in a fixed region near
  the start. It changes neither transitions nor extrinsic reward.
- Actions: turn left, turn right, and move forward.
- Compared methods: Double DQN, Double DQN + RND, and Double DQN + LP-RND.
- LP-RND signal:

  `max(0, error_lagged(observation) - error_current(observation))`

- Global `(x, y)` position and `(x, y, direction)` are used only in metrics,
  coverage, and heatmaps. They never enter DQN, RND, or LP-RND.
- Evaluation is greedy and uses extrinsic reward only.

The compact FourRooms variant is deliberate. In the standard `19 x 19`
environment, large open areas produce severe observation aliasing for a
feed-forward DQN. The compact layout retains partial observability and two door
crossings while making walls and openings locally identifiable.

## Files

- `envs.py`: fixed FourRooms geometry and Noisy-TV observation wrapper.
- `networks.py`: Q/RND MLPs and replay buffer.
- `intrinsic.py`: RND and lagged-predictor LP-RND.
- `experiment.py`: one training run, evaluation, diagnostics, and checkpoints.
- `sweep.py`: `smoke`, `pilot`, `main`, and `clean` experiment matrices.
- `aggregate.py`: bootstrap curves, tables, intrinsic diagnostics, and heatmaps.

## Quick start

From the repository root:

```bash
python -m pytest tests/final_project -q
python -m rl_exercises.final_project.sweep smoke --output-dir results/smoke
python -m rl_exercises.final_project.sweep pilot --output-dir results/pilot
```

Do not start `main` until the three-seed pilot shows both:

1. a usable extrinsic-performance signal, and
2. persistent separation between RND and LP-RND inside versus outside the TV.

The main matrix is:

```bash
python -m rl_exercises.final_project.sweep main --output-dir results/main
```

It contains 35 runs: 5 DQN seeds plus 5 seeds for every combination of
`{RND, LP-RND}` and beta in `{0.01, 0.05, 0.1}`.

Inspect a matrix without running it:

```bash
python -m rl_exercises.final_project.sweep main --dry-run
```
The clean-control matrix disables the stochastic TV vector and reproduces the
15-run sanity check: DQN, RND, and LP-RND with five seeds, using `beta=0.01`
for the intrinsic methods.

Because `results/clean/` already contains the reported runs, use a fresh output
directory when reproducing the control:

```bash
python -m rl_exercises.final_project.sweep clean \
  --output-dir results/reproduction_clean
Run a single configuration:

```bash
python -m rl_exercises.final_project.experiment \
  --method lp_rnd \
  --seed 0 \
  --beta 0.01 \
  --total-steps 50000 \
  --output-dir results/lp_rnd_seed0
```

Aggregate an existing sweep again:

```bash
python -m rl_exercises.final_project.aggregate results/pilot
```

## Outputs

Every run writes and flushes:

- `config.json` and `run_state.json`;
- `episodes.csv`;
- `evaluations.csv`;
- `diagnostics.csv`;
- `visitation.npy`;
- `checkpoint.pt`;
- `summary.json` after successful completion.

The sweep additionally creates `aggregate/` with:

- per-run and across-seed CSV summaries;
- success, return, TV-attraction, and coverage curves;
- inside/outside-TV intrinsic-signal curves;
- aggregate visitation heatmaps.

The diagnostics are binned every 1,000 training steps by default. This avoids
the storage cost of a step-level CSV while retaining the failure-mode signal.

## Reproducibility and resources

- Environment, action sampling, replay sampling, network initialization, and TV
  noise are seeded.
- Every run defaults to CPU and one PyTorch thread.
- Sweeps run serially by default. `--jobs 2` is available, but only use it if
  memory is sufficient.
- Result directories are never overwritten. Use a fresh output directory for
  every new sweep.
