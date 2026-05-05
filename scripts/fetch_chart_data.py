#!/usr/bin/env python3
"""
Fetch the sashankpillai/spotify-top-200-charts-20202021 dataset from Kaggle and
reshape it into the existing data/raw/spotify_tracks.csv schema plus a
chart_weeks column derived from "Number of Times Charted".

Requires Kaggle credentials at ~/.kaggle/kaggle.json (or the newer
~/.kaggle/access_token format).
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATASET = "sashankpillai/spotify-top-200-charts-20202021"
EXTERNAL_DIR = ROOT / "data" / "external"
EXTERNAL_CSV = EXTERNAL_DIR / "spotify_dataset.csv"
RAW_CSV = ROOT / "data" / "raw" / "spotify_tracks.csv"


def download_if_missing() -> None:
    if EXTERNAL_CSV.is_file():
        print(f"Already present: {EXTERNAL_CSV}")
        return
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {DATASET} -> {EXTERNAL_DIR}")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", DATASET, "-p", str(EXTERNAL_DIR), "--unzip"],
        check=True,
    )


def first_genre(raw: object) -> str:
    """Sashankpillai's Genre column is a string-repr of a list; take the first label."""
    if not isinstance(raw, str) or not raw.strip():
        return "unknown"
    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return raw.strip()
    if isinstance(parsed, list):
        return str(parsed[0]).strip() if parsed else "unknown"
    return str(parsed).strip()


def reshape_to_canonical(src: pd.DataFrame) -> pd.DataFrame:
    """Convert sashankpillai schema -> repo's expected schema + chart_weeks."""
    out = pd.DataFrame()
    out["track_name"] = src["Song Name"].astype(str)
    out["artist_name"] = src["Artist"].astype(str)
    out["genre"] = src["Genre"].apply(first_genre)
    out["popularity"] = pd.to_numeric(src["Popularity"], errors="coerce")
    out["chart_weeks"] = pd.to_numeric(src["Number of Times Charted"], errors="coerce")
    out["danceability"] = pd.to_numeric(src["Danceability"], errors="coerce")
    out["energy"] = pd.to_numeric(src["Energy"], errors="coerce")
    out["valence"] = pd.to_numeric(src["Valence"], errors="coerce")
    out["tempo"] = pd.to_numeric(src["Tempo"], errors="coerce")
    out["loudness"] = pd.to_numeric(src["Loudness"], errors="coerce")
    out["speechiness"] = pd.to_numeric(src["Speechiness"], errors="coerce")
    out["acousticness"] = pd.to_numeric(src["Acousticness"], errors="coerce")
    out["liveness"] = pd.to_numeric(src["Liveness"], errors="coerce")
    out["duration_ms"] = pd.to_numeric(src["Duration (ms)"], errors="coerce")
    # Sashankpillai does not include instrumentalness; fill with NaN so the
    # cleaner can drop or impute as configured.
    out["instrumentalness"] = pd.NA
    return out


def main() -> int:
    download_if_missing()
    src = pd.read_csv(EXTERNAL_CSV)
    out = reshape_to_canonical(src)
    RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(RAW_CSV, index=False)
    print(f"Wrote {RAW_CSV} {out.shape}")
    print("chart_weeks distribution:")
    print(out["chart_weeks"].describe().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
