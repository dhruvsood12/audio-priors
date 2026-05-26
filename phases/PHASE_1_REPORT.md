# Phase 1 Report: data ingestion and validation

## What changed

Added:
- `src/audio_priors/schemas.py` with a `ProcessedTrack` Pandera schema for
  the canonical frame and three minimal raw schemas
  (`RawMaharshipandya`, `RawRodolfofigueroa`, `RawParadisejoy`). Audio
  features bounded to `[0, 1]`, popularity to `[0, 100]`, tempo strictly
  inside `(40, 250)`, and duration_ms strictly inside `(30000, 600000)`
  per brief Section 6.1.
- `src/audio_priors/data.py` with one loader per source slug, a
  normalized-key deduplicator that keeps the row with the most non-null
  fields among duplicates, a range validator (drops out-of-range
  popularity / tempo / duration_ms, coerces invalid `[0, 1]` audio
  features to NaN), and `build_processed_dataset` end-to-end orchestration.
  `sha256_file` and `write_manifest` cover provenance.
- `scripts/download_data.py`: a Typer CLI that pulls each Kaggle dataset
  to `data/raw/<subdir>/`, computes SHA256, records row count, writes
  `data/raw/MANIFEST.json`, and optionally invokes
  `build_processed_dataset` to produce `data/processed/tracks.parquet`.
  Flags: `--skip-download` (use already-downloaded CSVs) and
  `--skip-build` (manifest only).
- `scripts/make_demo_data.py`: a Typer CLI that generates a synthetic
  2,000-row CSV in the maharshipandya raw schema. Writes to the path the
  real maharshipandya download lands at so the same pipeline runs without
  Kaggle credentials.
- `tests/test_schemas.py` (4 tests) and `tests/test_data.py` (11 tests).
- `DATA.md` with the Spotify Web API deprecation note, the combined
  corpus stats, and per-source attribution, license, fields used,
  fields dropped, and known gaps.

Modified:
- `pyproject.toml`: per-file ignore `B008` for `scripts/*` so
  `typer.Option(...)` as a default-arg value (the standard Typer idiom)
  is not flagged.
- `CLAUDE.md`: Phase 1 box checked, current focus updated.

## Tests

```
$ pytest tests/ -q
...............                                                          [100%]
15 passed
```

Breakdown:
- `test_schemas.py`: 4 tests covering valid acceptance, popularity > 100
  rejection, danceability > 1 rejection, and the strict tempo lower
  bound (`tempo == 40` must be rejected because the brief uses an open
  interval).
- `test_data.py`: 11 tests covering the list-shaped artist parser,
  whitespace and case dedup behavior, dedup row preference (keep the
  row with more non-null fields), the four range gates (popularity,
  tempo, duration_ms, and the `[0, 1]` features coerced to NaN), blank
  track / artist drop, unknown-slug `KeyError`, deterministic SHA256,
  manifest round-trip, and source-loader registration.

## Metrics

Not applicable for Phase 1. Corpus build statistics from
`build_processed_dataset` running against the three Kaggle sources:

| Quantity | Value |
|---|---|
| Pre-dedup rows | 1,320,025 |
| Rows dropped (blank required) | 4 |
| Duplicate rows collapsed | 121,042 |
| Tempo out of range | 3,219 |
| Duration_ms out of range | 30,259 |
| **Final rows in tracks.parquet** | **1,165,501** |

`data/raw/MANIFEST.json` records each source slug, CSV name, SHA256,
row count, and byte size.

## Known issues

- `rodolfofigueroa/spotify-12m-songs` carries no `popularity` and no
  `genre`. Its 1.2M rows enter the parquet with NaN in both columns and
  are excluded from popularity-target model fitting (Phase 3). They
  remain available for cold-start retrieval (Phase 5), which does not
  require labels. Documented in `DATA.md` and surfaced via the `source`
  column.
- Kaggle metadata reports the three source licenses as `ODbL-1.0`,
  `unknown`, and `other`. The raw CSVs are gitignored and not
  redistributed by this repo. Anyone consuming the processed parquet
  downstream is responsible for honoring the strictest upstream license.

## Acceptance criteria

- [x] Processed parquet has 300K+ rows after dedup
      (`rows_final = 1,165,501`).
- [x] `pytest tests/test_data.py tests/test_schemas.py` passes with
      8+ tests (15 tests, all green).
- [x] `data/raw/MANIFEST.json` records every source dataset with
      checksum (three entries, each with `sha256`, `rows`, `bytes`).
- [x] `DATA.md` updated with per-source attribution and license.

## Next phase entry condition

Met. Phase 2 (`feat/phase-2-eda`: EDA notebook and label design, target
~2 hours, brief Section 6 Phase 2) opens after Dhruv reviews and merges
this PR and explicitly says go.
