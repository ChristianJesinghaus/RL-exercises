# Pilot gate and interpretation

## Status

The packaged implementation is intended to be treated as **pilot-ready**, not
as a source of final empirical claims.

Exploratory runs in the earlier temporary development environment produced the
following indications:

- Standard RND retained a much larger normalized signal inside the TV region
  than outside (approximately a `16.2` inside/outside ratio in one run).
- LP-RND reduced that ratio to approximately `0.99` in the corresponding
  one-seed diagnostic.
- LP-RND with `beta=0.01` reached a successful greedy evaluation at 50k steps
  in one exploratory seed.
- Standard RND showed a temporary successful policy followed by collapse while
  its TV error remained high.

These numbers came from exploratory gate runs. Their raw result directories
were not retained when the temporary workspace expired. They therefore must
**not** be copied into the report as final evidence. Reproduce the mechanism
with the packaged three-seed pilot first.

## Required pilot

Run:

```bash
python -m rl_exercises.final_project.sweep pilot --output-dir results/pilot
```

This produces 9 runs:

- DQN, RND, LP-RND;
- seeds 0, 1, 2;
- 50k steps;
- `beta=0.01` for RND and LP-RND.

## Go/no-go criteria

Proceed to the main matrix only if the pilot jointly shows:

1. reward discovery and a non-trivial greedy success signal;
2. persistently higher RND error or normalized RND bonus inside the TV region;
3. a substantially flatter inside/outside LP-RND signal;
4. interpretable TV-attraction and position-coverage differences across seeds.

If no method learns:

- first extend only the pilot budget from 50k to 100k for the same three seeds;
- do not change the layout and algorithm simultaneously;
- inspect reward-discovery steps before changing DQN.

If RND is not distracted:

- verify that TV noise is active in `diagnostics.csv`;
- inspect inside/outside raw RND error;
- only then consider increasing `noise_dim` or TV-region size.

If LP-RND has no signal:

- inspect the snapshot interval;
- compare current and lagged RND error;
- test one interval change in the pilot only.

## Main matrix

The main preset contains the requested compact beta ablation:

- DQN: 5 seeds;
- RND: beta in `{0.01, 0.05, 0.1}`, 5 seeds each;
- LP-RND: beta in `{0.01, 0.05, 0.1}`, 5 seeds each;
- total: 35 runs at 100k steps.

Clean FourRooms is optional and should only be run with the selected beta after
the noisy main experiment is complete.
