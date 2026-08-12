# Installation und Ausführung

Dieses Paket ist so aufgebaut, dass du es direkt in die Wurzel deines
`RL-exercises`-Repositories entpackst.

## 1. Dateien an die richtige Stelle kopieren

Wechsle zuerst in deinen lokalen Clone:

```bash
cd PFAD/ZU/RL-exercises
```

Entpacke `rl_final_project_bundle.zip` **in genau diesen Ordner**.

macOS/Linux:

```bash
unzip PFAD/ZU/rl_final_project_bundle.zip -d .
```

Windows PowerShell:

```powershell
Expand-Archive -Path PFAD\ZU\rl_final_project_bundle.zip -DestinationPath .
```

Danach müssen insbesondere diese Pfade existieren:

```text
RL-exercises/
├── INSTALL_AND_RUN.md
├── rl_exercises/
│   └── final_project/
│       ├── envs.py
│       ├── intrinsic.py
│       ├── experiment.py
│       ├── sweep.py
│       └── aggregate.py
├── tests/
│   └── final_project/
│       └── test_final_project.py
└── final_project/
    ├── proposal.tex
    ├── references.bib
    └── PILOT_GATE.md
```

Wichtig: Es darf **nicht** versehentlich
`RL-exercises/rl_final_project_bundle/rl_exercises/...` entstehen. Falls doch,
hast du den zusätzlichen Paketordner statt dessen Inhalt in das Repo kopiert.

## 2. Python-Umgebung einrichten

Das Kurs-Repo verlangt Python 3.11. Prüfe zunächst:

```bash
python --version
```

Empfohlener Weg mit `uv`:

```bash
python -m pip install uv
uv venv --python 3.11
```

Umgebung aktivieren:

macOS/Linux:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Dann das Repo samt Testabhängigkeiten installieren:

```bash
uv pip install -e ".[dev]"
```

Falls `gymnasium[box2d]` wegen einer lokalen SWIG-Installation scheitert, führe
zuerst Folgendes aus und wiederhole danach den vorherigen Befehl:

```bash
uv pip install swig
```

Alle folgenden Befehle werden aus der Wurzel von `RL-exercises` und mit
aktivierter virtueller Umgebung ausgeführt.

## 3. Installation prüfen

```bash
python -m pytest tests/final_project -q
```

Erwartung: `8 passed`.

Optional kannst du die Presets kontrollieren, ohne Training zu starten:

```bash
python -m rl_exercises.final_project.sweep smoke --dry-run
python -m rl_exercises.final_project.sweep pilot --dry-run
python -m rl_exercises.final_project.sweep main --dry-run
```

Die Matrizen umfassen 3, 9 beziehungsweise 35 Runs.

## 4. Smoke-Test ausführen

```bash
python -m rl_exercises.final_project.sweep smoke --output-dir results/smoke
```

Das sind drei sehr kurze Runs: DQN, RND und LP-RND. Danach müssen unter
`results/smoke/aggregate/plots/` automatisch Grafiken liegen.

Wenn `results/smoke` bereits Daten enthält, verwende einen neuen Namen, zum
Beispiel:

```bash
python -m rl_exercises.final_project.sweep smoke --output-dir results/smoke_02
```

Bestehende Resultate werden absichtlich nie überschrieben.

## 5. Drei-Seed-Pilot ausführen

Erst wenn Tests und Smoke-Test funktionieren:

```bash
python -m rl_exercises.final_project.sweep pilot --output-dir results/pilot
```

Der Pilot enthält:

- DQN, RND und LP-RND;
- Seeds `0`, `1`, `2`;
- 50.000 Schritte je Run;
- für RND und LP-RND `beta=0.01`;
- greedy Evaluation alle 5.000 Schritte;
- insgesamt 9 Runs.

Standardmäßig laufen sie seriell und mit einem CPU-Thread. Das ist die sichere
Einstellung. Falls dein Rechner genug Arbeitsspeicher und CPU-Kerne hat:

```bash
python -m rl_exercises.final_project.sweep pilot \
  --output-dir results/pilot_parallel \
  --jobs 2
```

Miss zunächst die Laufzeit des Smoke-Tests. RL-Laufzeiten hängen stark vom
Rechner ab; extrapoliere erst danach auf 9 oder 35 Runs.

## 6. Pilot auswerten

Die wichtigsten Dateien sind:

```text
results/pilot/aggregate/aggregate_summary.csv
results/pilot/aggregate/per_run_summary.csv
results/pilot/aggregate/plots/success_rate.png
results/pilot/aggregate/plots/tv_fraction.png
results/pilot/aggregate/plots/intrinsic_inside_outside.png
results/pilot/aggregate/plots/position_coverage.png
```

Falls du die Auswertung erneut erzeugen willst:

```bash
python -m rl_exercises.final_project.aggregate results/pilot
```

Der Pilot ist bestanden, wenn:

1. mindestens eine intrinsische Methode reproduzierbar Reward findet und eine
   brauchbare greedy Policy lernt;
2. Standard-RND innerhalb der TV-Zone ein dauerhaft höheres Signal zeigt;
3. LP-RND dieses Inside/Outside-Gefälle deutlich reduziert;
4. TV-Aufenthalt und extrinsische Leistung über die drei Seeds interpretierbar
   sind.

Ein einzelner Seed ist kein berichtsfähiges Ergebnis.

## 7. Hauptläufe erst nach dem Pilot

Wenn der Pilot den Mechanismus bestätigt:

```bash
python -m rl_exercises.final_project.sweep main --output-dir results/main
```

Das Hauptpreset enthält 35 Runs:

- DQN: 5 Seeds;
- RND: 3 Beta-Werte × 5 Seeds;
- LP-RND: 3 Beta-Werte × 5 Seeds;
- `beta` in `{0.01, 0.05, 0.1}`;
- 100.000 Schritte je Run.

Starte nicht vorsorglich alle 35 Runs, bevor du die Pilotplots geprüft hast.

## 8. Proposal kompilieren

Die Proposal-Dateien liegen in `final_project/`. Für das Kurslayout benötigst du
zusätzlich `adrl.sty` aus der offiziellen Kursvorlage. Kopiere diese Datei
neben `proposal.tex`:

```text
final_project/adrl.sty
```

Danach:

```bash
cd final_project
pdflatex proposal.tex
bibtex proposal
pdflatex proposal.tex
pdflatex proposal.tex
```

Alternativ genügt bei installiertem `latexmk`:

```bash
latexmk -pdf proposal.tex
```

## Konkrete nächste Schritte

1. ZIP in den Repo-Root entpacken.
2. Python-3.11-Umgebung aktivieren und Repo installieren.
3. Die 8 Tests ausführen.
4. Das Drei-Methoden-Smoke-Preset ausführen.
5. Die erzeugten Smoke-Plots kurz prüfen.
6. Den 9-Run-Pilot starten.
7. Pilotplots und CSVs gemeinsam auswerten.
8. Erst dann Konfiguration einfrieren und den 35-Run-Main-Sweep starten.
9. Optional anschließend einen kleinen Clean-FourRooms-Sanity-Check mit dem
   besten Beta durchführen.
