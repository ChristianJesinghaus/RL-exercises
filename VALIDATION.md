# Validation record

Validation date: 2026-07-29

The bundle was checked with:

- formatting: Ruff;
- static linting: Ruff;
- Python byte-compilation;
- 8 focused final-project tests;
- one complete `smoke` sweep with DQN, RND, and LP-RND;
- automatic aggregation of all three smoke runs.

Results:

```text
All checks passed.
8 passed.
3/3 smoke runs completed.
Aggregate CSV files, learning curves, intrinsic diagnostics, and all heatmaps
were generated successfully.
```

The validation environment used MiniGrid 3.1.0, Gymnasium 1.3.0, PyTorch
2.13.0, and pandas 2.3.3. The course repository targets Python 3.11; the source
uses Python-3.11-compatible language features.

The long three-seed pilot was deliberately not rerun while rebuilding this
download bundle. Run the packaged `pilot` preset before treating any
exploratory observation as a report result.
