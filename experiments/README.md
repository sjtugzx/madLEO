# MAD-LEO Technical-Validation Artifacts

This folder carries the statistical artifacts of the MAD-LEO technical
validation: the experiment and audit tables behind the paper's Technical
Validation section. They are analysis outputs, not dataset content — the
released dataset (`dataset/`, distributed separately) contains only the
annotations, evidence snapshots, and the operational Starlink slice. Every
table here is regenerated from the released dataset (plus the pipeline
workspace for the acquisition-dependent steps) via
`python3 scripts/experiments/run_experiments.py <cmd>`.

- `validation/` — the reference-subset validation tables (alignment audits,
  quantitative event/stable-window responses, state-estimation and fusion
  experiments, harmonization and tier-stratified checks, distribution
  completeness, external cross-validation, SLR audits, parse/rejection
  ledgers) plus the machine-readable `PROVENANCE.json` map and the
  `gap_taxonomy_completeness.json` completeness record.
- `starlink/` — the Q4 distribution/consistency analysis tables derived
  from the operational Starlink slice (shell structure, ephemeris state
  distributions, TLE-vs-ephemeris consistency, frame sensitivity).

Main generator commands: `event-response`, `stable-windows`,
`state-estimates`, `state-fusion`, `tv-hardening`, `distribution-validation`,
`starlink-distributions`, `external-crossvalidation`, `kozai-comparison`,
`core-label-tables`, `provenance-map`, `slr-crossformat-check`,
`slr-oc-audit`, `window-sensitivity`, `attenuation-demo`,
`starlink-step-candidates`.
