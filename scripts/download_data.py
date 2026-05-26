"""Download Kaggle source datasets and write data/raw/MANIFEST.json.

Optionally invokes the harmonization pipeline to produce
``data/processed/tracks.parquet``. Kaggle credentials must be installed at
``~/.kaggle/kaggle.json`` (or ``~/.kaggle/access_token``) before the download
step.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import typer

from audio_priors import data

app = typer.Typer(add_completion=False)

REPO_ROOT = Path(__file__).resolve().parents[1]

# Each entry: kaggle slug, local subdirectory under data/raw/, expected CSV name.
SOURCES: list[tuple[str, str, str]] = [
    ("maharshipandya/-spotify-tracks-dataset", "spotify-tracks-dataset", "dataset.csv"),
    ("rodolfofigueroa/spotify-12m-songs", "spotify-12m-songs", "tracks_features.csv"),
    (
        "paradisejoy/top-hits-spotify-from-20002019",
        "top-hits-spotify-from-20002019",
        "songs_normalize.csv",
    ),
]


@app.command()
def main(
    raw_dir: Path = typer.Option(
        REPO_ROOT / "data" / "raw",
        help="Directory under which each source dataset lives.",
    ),
    manifest_path: Path = typer.Option(
        REPO_ROOT / "data" / "raw" / "MANIFEST.json",
        help="Where to write the per-source manifest with checksums.",
    ),
    processed_path: Path = typer.Option(
        REPO_ROOT / "data" / "processed" / "tracks.parquet",
        help="Where to write the harmonized, deduplicated, validated parquet.",
    ),
    skip_download: bool = typer.Option(
        False,
        "--skip-download",
        help="Use already-downloaded CSVs and only refresh the manifest.",
    ),
    skip_build: bool = typer.Option(
        False,
        "--skip-build",
        help="Stop after writing the manifest. Do not build the parquet.",
    ),
) -> None:
    """Download each Kaggle dataset and build the processed parquet."""

    entries: list[dict[str, object]] = []
    sources_for_build: list[tuple[str, Path]] = []

    for slug, subdir, csv_name in SOURCES:
        target_dir = raw_dir / subdir
        target_dir.mkdir(parents=True, exist_ok=True)
        csv_path = target_dir / csv_name

        if not skip_download and not csv_path.is_file():
            typer.echo(f"-> kaggle datasets download -d {slug}")
            subprocess.run(
                [
                    "kaggle",
                    "datasets",
                    "download",
                    "-d",
                    slug,
                    "-p",
                    str(target_dir),
                    "--unzip",
                ],
                check=True,
            )

        if not csv_path.is_file():
            typer.echo(f"!  skipping {slug}: {csv_path} not present", err=True)
            continue

        checksum = data.sha256_file(csv_path)
        with open(csv_path, encoding="utf-8") as fh:
            row_count = sum(1 for _ in fh) - 1

        entries.append(
            {
                "slug": slug,
                "subdir": subdir,
                "csv": csv_name,
                "sha256": checksum,
                "rows": row_count,
                "bytes": csv_path.stat().st_size,
            }
        )
        sources_for_build.append((slug, csv_path))
        typer.echo(f"   {slug}: {row_count:,} rows, sha256 {checksum[:12]}")

    data.write_manifest(entries, manifest_path)
    typer.echo(f"wrote {manifest_path} ({len(entries)} sources)")

    if skip_build:
        return

    stats = data.build_processed_dataset(sources_for_build, processed_path)
    typer.echo(
        f"processed: {stats['rows_final']:,} rows after dedup, written to {stats['output_path']}"
    )


if __name__ == "__main__":
    app()
