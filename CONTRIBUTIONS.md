```markdown
# Contributions and Attribution

## Author

I, Christian Jesinghaus, am the sole author of the final project and am responsible
for the experimental design, execution, verification, analysis, report, and
presentation of the results.

## Original project contribution

The final-project-specific implementation is located in
`rl_exercises/final_project/`. My contribution comprises:

- designing the controlled Noisy-TV MiniGrid experiment;
- defining the fixed compact FourRooms layout and stochastic distractor region;
- ensuring that privileged position information is used only for diagnostics;
- implementing the Double-DQN experimental backbone, RND, LP-RND and the experimental specifics like the experiment matrices, the training and evaluation process
and interpretation;


The principal methodological extension is LP-RND. For observation \(o_t\), it
uses the non-negative improvement of the current RND predictor over a lagged
predictor snapshot:

`max(0, error_lagged(o_t) - error_current(o_t))`

This is intended to reward reducible prediction error while attenuating
irreducible stochastic novelty.

## Project-specific files

| Path | Purpose |
|---|---|
| `rl_exercises/final_project/envs.py` | Fixed FourRooms environment and Noisy-TV observation wrapper |
| `rl_exercises/final_project/networks.py` | Q-networks, predictor networks, and replay buffer |
| `rl_exercises/final_project/intrinsic.py` | RND and LP-RND intrinsic-reward implementations |
| `rl_exercises/final_project/experiment.py` | Training, evaluation, diagnostics, and checkpoint creation |
| `rl_exercises/final_project/sweep.py` | Smoke, pilot, main, and clean experiment matrices |
| `rl_exercises/final_project/aggregate.py` | Aggregation, confidence intervals, plots, and heatmaps |
| `tests/final_project/` | Focused tests for environment, methods, artifacts, and sweep definitions |
| `results/main/` | Complete 35-run Noisy-TV experiment |
| `results/clean/` | Complete 15-run clean-control experiment |
| `final_project/` | Proposal, final report, references, and course report style |

## External assets and conceptual sources

| Asset or concept | How it is used | Source | License or status |
|---|---|---|---|
| RL course repository | Repository structure, weekly exercises, packaging, and course infrastructure | [automl-edu/RL-exercises](https://github.com/automl-edu/RL-exercises) | Apache License 2.0 |
| MiniGrid | Environment API and symbolic partial observations; the project defines its own fixed layout and wrapper | [Farama Foundation MiniGrid](https://github.com/Farama-Foundation/Minigrid) | Apache License 2.0 |
| Double DQN | Conceptual basis of the value-learning update; no external implementation was copied | [van Hasselt, Guez, and Silver, 2016](https://ojs.aaai.org/index.php/AAAI/article/view/10295) | Academic citation; not imported source code |
| Random Network Distillation | Conceptual basis of the intrinsic reward; no external implementation was copied | [Burda et al., 2019](https://openreview.net/forum?id=H1lJJnR5Ym) | Academic citation; not imported source code |


## AI assistance disclosure

I used ChatGPT for discussing and challenging the research idea, refining the
experimental design, drafting and reviewing code, debugging,
constructing plotting and aggregation utilities, checking result
interpretations, and editing the report and poster.

I made the final methodological decisions, inspected and integrated and changed code,
ran the experiments on my own hardware, checked the stored configurations and
outputs, reviewed the generated figures and text.

## Result ownership and verification

All 50 reported final runs were executed by myself:

- 35 Noisy-TV main runs;
- 15 clean-control runs.

Each completed run includes its configuration, raw metrics, diagnostics,
visitation array, summary, completion state, and model checkpoint. 
