# Data sources

Three Kaggle datasets feed the audio-priors corpus. None are called via the
live Spotify Web API; all are archived snapshots downloaded with the Kaggle
CLI. The harmonization pipeline lives in `src/audio_priors/data.py` and the
provenance manifest is regenerated on every run at `data/raw/MANIFEST.json`.

## Spotify Web API status

Spotify deprecated the public `/audio-features` and `/audio-analysis`
endpoints for newly registered applications in November 2024. This project
does not call the live API. Every audio feature in the corpus comes from
the three Kaggle dumps below, which were pulled in 2020 to 2022 against
the then-public API.

## Combined corpus

|  | Count |
|---|---|
| Rows pre-dedup | 1,320,025 |
| Rows dropped (blank track or artist name) | 4 |
| Duplicate rows collapsed | 121,042 |
| Rows dropped (`tempo` outside `(40, 250)`) | 3,219 |
| Rows dropped (`duration_ms` outside `(30000, 600000)`) | 30,259 |
| **Rows in `data/processed/tracks.parquet`** | **1,165,501** |

Dedup key is the lowercased, whitespace-stripped pair `(track_name,
artist_name)`. Among duplicates, the row with the most non-null fields wins,
so popularity-bearing rows from maharshipandya beat the same track's
unlabeled twin from rodolfofigueroa.

## Source 1: maharshipandya/-spotify-tracks-dataset

- Slug: `maharshipandya/-spotify-tracks-dataset`
- Kaggle URL: https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset
- License (per Kaggle metadata): `ODbL-1.0` (Open Database License 1.0)
- Rows: 114,000
- Sources: Spotify
- Fields used: `track_name`, `artists` (renamed to `artist_name`),
  `track_genre` (renamed to `genre`), `popularity`, `danceability`,
  `energy`, `valence`, `acousticness`, `instrumentalness`, `liveness`,
  `speechiness`, `loudness`, `tempo`, `duration_ms`.
- Fields dropped: `track_id`, `album_name`, `explicit`, `key`, `mode`,
  `time_signature`, the unnamed pandas index column.
- Known gaps: none.

## Source 2: rodolfofigueroa/spotify-12m-songs

- Slug: `rodolfofigueroa/spotify-12m-songs`
- Kaggle URL: https://www.kaggle.com/datasets/rodolfofigueroa/spotify-12m-songs
- License (per Kaggle metadata): `unknown`
- Rows: 1,204,025
- Sources: MusicBrainz, Spotify
- Fields used: `name` (renamed to `track_name`), `artists` (parsed from the
  string repr of a Python list, first element becomes `artist_name`),
  and the same audio-feature columns as Source 1.
- Fields dropped: `id`, `album`, `album_id`, `artist_ids`, `track_number`,
  `disc_number`, `explicit`, `key`, `mode`, `time_signature`, `year`,
  `release_date`.
- Known gaps: **no `popularity` and no `genre`.** Rows from this source
  enter the parquet with NaN in both columns and are excluded from any
  popularity-target model fitting. They remain available for cold-start
  retrieval (Phase 5), which does not require popularity labels.

## Source 3: paradisejoy/top-hits-spotify-from-20002019

- Slug: `paradisejoy/top-hits-spotify-from-20002019`
- Kaggle URL: https://www.kaggle.com/datasets/paradisejoy/top-hits-spotify-from-20002019
- License (per Kaggle metadata): `other`
- Rows: 2,000
- Sources: Spotify (collected with Spotipy, top playlists 2000 to 2019)
- Fields used: `song` (renamed to `track_name`), `artist` (renamed to
  `artist_name`), `genre`, `popularity`, plus the same audio features.
- Fields dropped: `year`, `explicit`, `key`, `mode`.
- Known gaps: small sample, chart-skewed (top-hits playlists only). Useful
  as a sanity-check overlap set with the larger maharshipandya frame.

## Preprocessing decisions

1. **Single canonical schema.** Every source is mapped onto the column set
   in `src/audio_priors/schemas.py::PROCESSED_COLUMNS` so downstream code
   ignores source provenance except for diagnostics. A `source` column
   preserves the slug each row came from.
2. **Dedup on normalized identifier pair.** A track that appears in
   both maharshipandya and rodolfofigueroa collapses to the
   maharshipandya row (it has popularity). A track that appears only in
   rodolfofigueroa stays with NaN popularity.
3. **Range gates as drop vs coerce.** Out-of-range `popularity`, `tempo`,
   or `duration_ms` drops the row (a track is unusable without a valid
   one of these). An out-of-range `[0, 1]` audio feature is set to NaN
   instead, on the read that those are usually data-entry artifacts and
   the rest of the row is still informative.
4. **Strict bounds on tempo and duration_ms.** Per brief Section 6.1
   these intervals are open (`(40, 250)` and `(30000, 600000)`). The
   Pandera schema in `ProcessedTrack` uses `gt` / `lt` rather than `ge`
   / `le` so a track at exactly the boundary is rejected.

## Attribution

When citing this corpus, list all three Kaggle slugs and credit Spotify
as the upstream feature source. The license soup means redistribution of
the raw data should respect each source's individual terms; the
processed parquet is not redistributed publicly by this project and is
gitignored under `data/processed/`.
