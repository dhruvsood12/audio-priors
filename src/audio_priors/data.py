"""Column harmonization, deduplication, validation, and processed-frame export.

Each source dataset has its own column names and shape. ``load_*`` functions
map a raw frame onto the canonical schema declared in
:mod:`audio_priors.schemas`. ``build_processed_dataset`` orchestrates the
full pipeline: read each source CSV, harmonize, concatenate, deduplicate,
validate ranges, and write the result to a parquet file.
"""

from __future__ import annotations

import ast
import hashlib
import json
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .schemas import AUDIO_01_FEATURES, PROCESSED_COLUMNS, ProcessedTrack


def _normalize_artist_list(value: Any) -> str:
    """rodolfofigueroa stores ``artists`` as a string repr of a Python list.

    Return the first artist name or an empty string if parsing fails or the
    value is missing.
    """

    if not isinstance(value, str) or not value.strip():
        return ""
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value.strip()
    if isinstance(parsed, list) and parsed:
        return str(parsed[0]).strip()
    return str(parsed).strip()


def _select_canonical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` reindexed to the canonical column order, filling gaps."""

    out = df.reindex(columns=list(PROCESSED_COLUMNS))
    return out


def load_maharshipandya(path: str | Path) -> pd.DataFrame:
    """Harmonize maharshipandya/-spotify-tracks-dataset into canonical shape."""

    df = pd.read_csv(path)
    df = df.rename(
        columns={
            "artists": "artist_name",
            "track_genre": "genre",
        }
    )
    df["source"] = "maharshipandya"
    return _select_canonical_columns(df)


def load_rodolfofigueroa(path: str | Path) -> pd.DataFrame:
    """Harmonize rodolfofigueroa/spotify-12m-songs into canonical shape.

    Parses the list-shaped ``artists`` column to its first entry, renames
    ``name`` to ``track_name``, and leaves ``popularity`` / ``genre`` NaN
    because the source carries neither.
    """

    df = pd.read_csv(path)
    df = df.rename(columns={"name": "track_name"})
    df["artist_name"] = df["artists"].apply(_normalize_artist_list)
    df["source"] = "rodolfofigueroa"
    return _select_canonical_columns(df)


def load_paradisejoy(path: str | Path) -> pd.DataFrame:
    """Harmonize paradisejoy/top-hits-spotify-from-20002019 into canonical."""

    df = pd.read_csv(path)
    df = df.rename(columns={"song": "track_name", "artist": "artist_name"})
    df["source"] = "paradisejoy"
    return _select_canonical_columns(df)


SOURCE_LOADERS = {
    "maharshipandya/-spotify-tracks-dataset": load_maharshipandya,
    "rodolfofigueroa/spotify-12m-songs": load_rodolfofigueroa,
    "paradisejoy/top-hits-spotify-from-20002019": load_paradisejoy,
}


def normalized_key(series: pd.Series) -> pd.Series:
    """Lowercase + strip a string column for case-insensitive dedup."""

    return series.fillna("").astype(str).str.strip().str.lower()


def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop duplicates on the normalized (track_name, artist_name) key.

    Sorts so the row with the most non-null fields wins among duplicates,
    keeping the row with the most signal.
    """

    work = df.copy()
    work["_track_key"] = normalized_key(work["track_name"])
    work["_artist_key"] = normalized_key(work["artist_name"])
    work["_non_null"] = work.notna().sum(axis=1)
    work = work.sort_values("_non_null", ascending=False, kind="stable")
    out = work.drop_duplicates(subset=["_track_key", "_artist_key"], keep="first")
    out = out.drop(columns=["_track_key", "_artist_key", "_non_null"])
    out = out.reset_index(drop=True)
    return out, len(df) - len(out)


def validate_ranges(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Drop rows whose required numeric fields fall outside the brief's ranges.

    Audio features outside [0, 1] are coerced to NaN rather than dropped.
    Popularity outside [0, 100], tempo outside (40, 250), and duration_ms
    outside (30000, 600000) drop the row.
    """

    out = df.copy()
    notes: dict[str, int] = {}

    if "popularity" in out.columns:
        bad = (out["popularity"] < 0) | (out["popularity"] > 100)
        n = int(bad.fillna(False).sum())
        if n:
            out = out.loc[~bad.fillna(False)].reset_index(drop=True)
            notes["popularity_out_of_range_dropped"] = n

    for col in AUDIO_01_FEATURES:
        if col not in out.columns:
            continue
        bad = (out[col] < 0) | (out[col] > 1)
        n = int(bad.fillna(False).sum())
        if n:
            out.loc[bad.fillna(False), col] = pd.NA
            notes[f"{col}_invalid_set_to_na"] = n

    if "tempo" in out.columns:
        bad = (out["tempo"] <= 40) | (out["tempo"] >= 250)
        n = int(bad.fillna(False).sum())
        if n:
            out = out.loc[~bad.fillna(False)].reset_index(drop=True)
            notes["tempo_out_of_range_dropped"] = n

    if "duration_ms" in out.columns:
        bad = (out["duration_ms"] <= 30000) | (out["duration_ms"] >= 600000)
        n = int(bad.fillna(False).sum())
        if n:
            out = out.loc[~bad.fillna(False)].reset_index(drop=True)
            notes["duration_ms_out_of_range_dropped"] = n

    return out, notes


def drop_missing_required(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop rows where ``track_name`` or ``artist_name`` is blank."""

    work = df.copy()
    track_blank = normalized_key(work["track_name"]) == ""
    artist_blank = normalized_key(work["artist_name"]) == ""
    keep = ~(track_blank | artist_blank)
    out = work.loc[keep].reset_index(drop=True)
    return out, int((~keep).sum())


def harmonize_sources(loaders: Iterable[tuple[str, Path]]) -> pd.DataFrame:
    """Apply each ``(slug, path)`` loader and concatenate the results.

    Unknown slugs raise ``KeyError``; loaders are looked up in
    :data:`SOURCE_LOADERS`.
    """

    frames: list[pd.DataFrame] = []
    for slug, path in loaders:
        if slug not in SOURCE_LOADERS:
            raise KeyError(f"no loader registered for slug: {slug}")
        frames.append(SOURCE_LOADERS[slug](path))
    if not frames:
        raise ValueError("no source frames to harmonize")
    return pd.concat(frames, ignore_index=True)


def build_processed_dataset(
    sources: Iterable[tuple[str, Path]],
    output_path: str | Path,
) -> dict[str, Any]:
    """End-to-end: harmonize, dedup, validate, write parquet, return stats."""

    combined = harmonize_sources(sources)
    pre_rows = len(combined)

    combined, n_missing = drop_missing_required(combined)
    dedupped, n_dup = deduplicate(combined)
    validated, range_notes = validate_ranges(dedupped)

    validated = ProcessedTrack.validate(validated)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    validated.to_parquet(out_path, index=False)

    return {
        "rows_pre_dedup": pre_rows,
        "rows_dropped_missing_required": n_missing,
        "rows_duplicates": n_dup,
        "rows_final": len(validated),
        "range_validation": range_notes,
        "output_path": str(out_path),
    }


def sha256_file(path: str | Path) -> str:
    """Hex SHA256 of a file. Reads in 1 MiB chunks for large CSVs."""

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(
    entries: Iterable[dict[str, Any]],
    manifest_path: str | Path,
) -> Path:
    """Write a MANIFEST.json with provenance for each source dataset."""

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sources": list(entries),
    }
    p = Path(manifest_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=2) + "\n")
    return p
