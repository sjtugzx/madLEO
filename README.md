# MAD-LEO

**MAD-LEO** is a maneuver-annotated, multi-source orbital dataset for low
Earth orbit satellites, spanning **1,134 mission-reported maneuver events
(1992–2026) over eleven geodetic and altimetry satellites**, each evaluated
against three independent evidence sources — public TLE catalogs,
GNSS/DORIS precise orbit products, and satellite laser ranging (SLR) —
together with **1,139 deterministic stable no-event windows** as negative
controls and a **107-hour full-constellation Starlink operational slice**
(43,361,358 predicted ephemeris states, no validated labels).

This repository contains the complete reproducible pipeline that builds the
dataset from public provider archives, together with the
technical-validation experiment suite. The released dataset is openly
available on Figshare at https://doi.org/10.6084/m9.figshare.33446503.v1
under CC BY 4.0.

## Repository layout

```text
mad-leo/
├── dataset/                  # THE Figshare release package (4.3 GB, git-ignored)
│   ├── mission_reported/     #   mission-reported subset (11 reference satellites)
│   │   ├── annotations/      #     4 label tables (events, stable windows, matched)
│   │   └── evidence/         #     per-target TLE / orbit / SLR Parquet snapshots
│   ├── operational/starlink/ #   Starlink slice + TLE records (Parquet only)
│   ├── docs/                 #   metadata.md (file inventory + table schemas)
│   └── manifest.json         #   per-file SHA-256, byte size, row count
├── experiments/              # TECHNICAL-VALIDATION ARTIFACTS (tracked in git)
│   ├── validation/           #   60 files behind the paper's Technical Validation
│   └── starlink/             #   8 audit/analysis tables for the operational slice
├── scripts/
│   ├── data_pipeline/        # END-TO-END DATASET PIPELINE
│   │   ├── acquire.py        #   stage 1 entry: provider downloads
│   │   ├── process.py        #   stages 2–4 entry: normalize/align/labels/release
│   │   ├── run_reference.sh  #   mission-reported subset driver (dry-run by default)
│   │   ├── run_operational.sh#   Starlink operational subset driver
│   │   ├── params/           #   non-secret pipeline parameters (*.env)
│   │   ├── downloaders/      #   provider clients (Space-Track, CDSE, PO.DAAC,
│   │   │                     #   CDDIS/ILRS) + acquisition orchestrators
│   │   ├── processors/       #   format parsers (POD, SLR, OGDR, SWOT POR,
│   │   │                     #   coordinate transforms TEME/ITRF/GCRS)
│   │   ├── alignment/        #   normalization, window audits, evidence
│   │   │                     #   alignment, label construction, release build
│   │   ├── pipeline/         #   four-stage orchestration (Methods-aligned)
│   │   ├── benchmarking/     #   core library: schema, config, stable windows
│   │   └── analyzers/        #   response/state estimators shared with experiments
│   └── experiments/          # TECHNICAL-VALIDATION EXPERIMENTS (code)
│       ├── run_experiments.py#   entry point for all experiments
│       └── generate_*.py     #   event response, stable windows, state
│                             #   estimation/fusion, sigma model, distributions,
│                             #   Starlink consistency, external benchmark,
│                             #   figure builders (make_tv_figures.py)
├── configs/                  # versioned target/provider manifests (JSON)
├── requirements.txt          # pinned dependencies
└── .env.example              # credential template (copy to .env)
```

The released dataset is `dataset/` only — it ships to Figshare without any
analysis tables. On the deposit, path separators are encoded as double
underscores in the file names (the deposit stores a flat list); the mapping
is documented in `docs__metadata.md` on the deposit. The statistical
artifacts behind the Technical Validation analyses live in `experiments/`
and are published with the code.
Two directories are created at runtime and are git-ignored: `data/`
(pipeline workspace: raw downloads + interim tables) and `results/`
(experiment scratch output).

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in credentials only if you plan to run acquisition
```

## Quickstart: validate the shipped dataset

These experiments consume the shipped `dataset/` snapshots directly and
need no credentials or prior pipeline run. To obtain the snapshots,
download the Figshare deposit and place the files under `dataset/`,
restoring the logical paths by replacing each `__` in a file name with a
path separator.

```bash
# Starlink shell/ephemeris distributions and TLE-ephemeris consistency
python scripts/experiments/run_experiments.py starlink-distributions

# Per-target TLE / orbit / SLR distribution summaries + tier equivalence
python scripts/experiments/run_experiments.py distribution-validation
```

Verify the shipped package integrity:

```bash
python - <<'EOF'
import json, hashlib, pathlib
root = pathlib.Path("dataset")
m = json.loads((root / "manifest.json").read_text())
ok = all(
    hashlib.sha256((root / f["path"]).read_bytes()).hexdigest() == f["sha256"]
    for f in m["files"]
)
print(f"{len(m['files'])} files, all SHA-256 verified: {ok}")
EOF
```

Release-level consistency gate (table counts, label semantics, known-issue
ledger):

```bash
python scripts/data_pipeline/process.py selfcheck
```

## Full reproduction pipeline

The four-stage pipeline mirrors the paper's Methods. Stage 1 downloads
from public provider archives (needs credentials in `.env`); stages 2–4
process, align, construct labels, and build the release package.

```bash
bash scripts/data_pipeline/run_reference.sh            # prints the plan (dry-run)
DRY_RUN=0 bash scripts/data_pipeline/run_reference.sh  # executes
DRY_RUN=0 bash scripts/data_pipeline/run_operational.sh
```

Individual stages can be run through the CLI dispatchers:

```bash
python scripts/data_pipeline/acquire.py ids|tle|orbit-bulk|orbit-windows|orbit-samples|slr|benchmark-data|starlink --execute
python scripts/data_pipeline/process.py normalize-annotations|audit-windows|annotation-alignment|export-labels|export-release
```

After the pipeline, the full experiment suite regenerates every
technical-validation table and figure:

```bash
for e in event-response stable-windows state-estimates state-fusion \
         tv-hardening distribution-validation starlink-distributions \
         external-crossvalidation kozai-comparison core-label-tables \
         provenance-map slr-crossformat-check slr-oc-audit \
         window-sensitivity attenuation-demo starlink-step-candidates; do
  python scripts/experiments/run_experiments.py $e
done
python scripts/experiments/run_experiments.py results-figures
python scripts/experiments/run_experiments.py new-analysis-figures
```
