# MAD-LEO

**MAD-LEO** is a maneuver-annotated, multi-source orbital dataset for low
Earth orbit satellites, spanning **1,134 mission-reported maneuver events
(1992–2026) over eleven geodetic and altimetry satellites**, each evaluated
against three independent evidence sources — public TLE catalogs,
GNSS/DORIS precise orbit products, and satellite laser ranging (SLR) —
together with **1,139 deterministic stable no-event windows** as negative
controls and a **107-hour full-constellation Starlink operational slice**
(43,361,358 predicted ephemeris states, no validated labels).

This repository contains the released dataset itself, the complete
reproducible pipeline that builds it from public provider archives, the
technical-validation experiment suite, and the accompanying paper.

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
│                             #   paper figure builders (make_tv_figures.py)
├── configs/                  # versioned target/provider manifests (JSON)
├── requirements.txt          # pinned dependencies
└── .env.example              # credential template (copy to .env)
```

The paper sources (LaTeX, Nature *Scientific Data* format) live under
`arxiv/MAD_LEO_paper/`; `arxiv/` is a git-ignored quarantine area, so the
manuscript is not tracked in git.

The released dataset is `dataset/` only — it ships to Figshare without any
analysis tables. The statistical artifacts behind the paper's Technical
Validation section live in `experiments/` and are published with the code.
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
need no credentials or prior pipeline run:

```bash
# Starlink shell/ephemeris distributions and TLE-ephemeris consistency
python scripts/experiments/run_experiments.py starlink-distributions

# Per-target TLE / orbit / SLR distribution summaries + tier equivalence
python scripts/experiments/run_experiments.py distribution-validation

# The paper's technical-validation figures (reads experiments/validation,
# experiments/starlink and dataset/, writes arxiv/MAD_LEO_paper/images/)
python scripts/experiments/run_experiments.py paper-tv-figures
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
         provenance-map slr-crossformat-check slr-oc-audit; do
  python scripts/experiments/run_experiments.py $e
done
python scripts/experiments/run_experiments.py results-figures
python scripts/experiments/run_experiments.py new-analysis-figures
python scripts/experiments/run_experiments.py paper-tv-figures
```

## The paper

`arxiv/MAD_LEO_paper/` contains the LaTeX sources of the accompanying data
descriptor (Nature *Scientific Data* format; git-ignored). Compile from that
directory:

```bash
cd arxiv/MAD_LEO_paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

All technical-validation figures under `arxiv/MAD_LEO_paper/images/` (except
the manually maintained Figure 1, `dataCollection.pdf`) are regenerated from
the shipped dataset by `paper-tv-figures`.

## Data sources and credentials

All credentials are read from environment variables (`.env`, git-ignored;
see `.env.example`): `SPACETRACK_ID/PASSWORD`, `CDSE_USERNAME/PASSWORD`,
`EARTHDATA_USERNAME/PASSWORD`, `CDDIS_FTPS_EMAIL`. No credentials are
stored in the repository. The private reviewer-provided TLE archive link
is redacted from `configs/` and configured via `STARLINK_TLE_ARCHIVE_URL`.

Annotation truth comes from International DORIS Service (IDS)
mission-published maneuver histories; orbit evidence from CDDIS/PO.DAAC/
CDSE; SLR from ILRS archives; TLE from public catalogs with Space-Track
as authenticated fallback.

## Claim boundaries

Starlink ephemerides are operator-published **predictions**, never
maneuver ground truth; the operational subset carries no labels by
design. Labels are `event` / `no_event` / `ignore`, where `ignore` is a
reserved value (no released window carries it). Confidence tiers (A/B/C)
encode evidence completeness, not data quality. See
`dataset/docs/metadata.md`.
