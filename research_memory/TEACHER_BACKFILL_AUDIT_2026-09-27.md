# GIADA teacher research-memory recovery — 2026-09-27

## Why this was necessary

The earlier mirror had narrative findings and protocol text, but many original
teacher experiments lacked queryable run, metric, observation, analysis, and
source-provenance records. This was a registration failure, not evidence that
the original experiment outputs were lost. The recovery indexes the outputs;
it does not rerun experiments or change their original conclusions.

## Recovered scope

- 30 outcome tasks (Tasks 1–16, including suffixed diagnostics and the distinct
  temporal-granularity matrix) are listed in
  `records/teacher_source_backfill_manifest_v1.json`.
- Their 29 distinct ZIP files are preserved in `source_archives/` (Tasks 12 and
  13 share one archive). Every ZIP passed CRC; each has an SHA-256 artifact
  record. The Task 0.4 double-oracle ZIP is also archived and linked to its
  preparatory finding.
- Each of the 30 outcome tasks has a protocol, prediction, aggregate run,
  source artifact, metric/evaluation/observation, finding, analysis, and
  technical quality check in SQLite. The backfill indexes 1,154 numeric
  observations, 1,025 evaluation definitions, and 138 explicitly recoverable
  experimental arms in the mirror overall. Aggregate run records are **not**
  misrepresented as per-seed runs.
- Preparatory Tasks 0.5 and 0.6 are linked to their versioned contract
  analyses; they are **not** labeled as performance experiments.
- The Task 16 gate observation is now linked to its run, analysis, and ZIP.

## Provenance and limits

- For 22 of the 30 outcome-task associations, the original result note states
  the exact ZIP SHA-256. Seven original result notes did not state that hash;
  their associations are marked semantic rather than cryptographically
  certified. Task 2b has no separate result note. The ZIPs themselves are
  hash-identified and CRC-verified. No absent historical hash has been invented.
- Indexed numbers come from explicit Markdown table cells or JSON scalar
  paths. Source names, row labels, JSON paths, and ZIP hashes are retained.
  Rounded Markdown values are not promoted to full-precision measurements;
  full reports and arrays remain in the archived ZIPs.
- Composite protocol criteria are stored as composite predictions. They were
  not atomized into individual prediction/evaluation links when a reliable
  one-to-one mapping was not available. A query that requires that exact edge
  may therefore undercount even though the experiment's metric observations
  are queryable by protocol. This is a remaining granularity limitation, not
  missing numerical evidence.
- The backfill covers the GIADA Ca_HVA teacher task chain, not every older
  HayFlow research program or every possible nested value in a report. It
  never claims independent seed-level samples where the original report only
  gave aggregates.
- Only the local SQLite mirror and this repository were modified. Airtable
  was not read, written, verified, or synchronized.

## Repeatable checks

Run from the repository root:

```powershell
python -m research_memory.audit_teacher_sources
python -m research_memory.audit_teacher_coverage
python -m research_memory verify
python -m unittest discover -s tests -p test_research_memory.py -q
```

At the time of this audit: 30/30 outcome tasks had all core relational links;
the mirror had 3,797 records across 78 tables, 241 relationships, zero pending
writes, zero foreign-key errors, and passed logical round-trip verification.
All 20 research-memory tests passed. These checks validate registration and
integrity, not the scientific validity of every original experimental claim.
