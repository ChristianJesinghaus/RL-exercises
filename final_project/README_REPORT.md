# Final report bundle

This bundle contains the final ADRL project report, its vector figures, the
figure-generation script, and the aggregate CSV inputs used by that script.

## Build

From this directory, with a standard TeX Live installation:

```bash
python3 make_report_figures.py
latexmk -pdf report.tex
```

The LaTeX source expects the course-provided `adrl.sty` and the generated
figures in `figures/`. The report PDF included in the bundle was built from the
same source and visually checked page by page.

## Main files

- `report.tex`: final four-page report body plus references and checklist.
- `references.bib`: verified publication references.
- `make_report_figures.py`: deterministic report visualizations.
- `figures/`: vector PDFs and high-resolution PNG previews.
- `data/`: aggregate and evaluation CSVs for the noisy and clean sweeps.
