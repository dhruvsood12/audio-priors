"""Unit tests for the data harmonization layer."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from audio_priors.data import (
    SOURCE_LOADERS,
    _normalize_artist_list,
    deduplicate,
    drop_missing_required,
    harmonize_sources,
    sha256_file,
    validate_ranges,
    write_manifest,
)

AUDIO_BASE = {
    "danceability": 0.5,
    "energy": 0.5,
    "valence": 0.5,
    "acousticness": 0.5,
    "instrumentalness": 0.5,
    "liveness": 0.5,
    "speechiness": 0.5,
}


def test_normalize_artist_list_parses_python_list_repr() -> None:
    assert _normalize_artist_list("['Rage Against The Machine']") == "Rage Against The Machine"
    assert _normalize_artist_list("['A', 'B']") == "A"


def test_normalize_artist_list_handles_blank_and_none() -> None:
    assert _normalize_artist_list("") == ""
    assert _normalize_artist_list(None) == ""
    assert _normalize_artist_list("   ") == ""


def test_deduplicate_keeps_row_with_more_signal() -> None:
    """Casing and surrounding whitespace must not defeat deduplication."""
    df = pd.DataFrame(
        {
            "source": ["a", "b"],
            "track_name": [" Song One ", "song one"],
            "artist_name": ["Foo", "FOO"],
            "popularity": [50.0, None],
            "genre": [None, None],
            **{k: [v, v] for k, v in AUDIO_BASE.items()},
            "loudness": [-5.0, -5.0],
            "tempo": [120.0, 120.0],
            "duration_ms": [200_000, 200_000],
        }
    )
    out, n_dup = deduplicate(df)
    assert n_dup == 1
    assert len(out) == 1
    # The row with the popularity value (more non-null fields) should win.
    assert out.iloc[0]["popularity"] == 50.0


def test_validate_ranges_drops_out_of_range_popularity() -> None:
    df = pd.DataFrame(
        {
            "popularity": [50.0, 150.0, -5.0, 80.0],
            "tempo": [100.0, 100.0, 100.0, 100.0],
            "duration_ms": [200_000, 200_000, 200_000, 200_000],
            **{k: [v, v, v, v] for k, v in AUDIO_BASE.items()},
        }
    )
    out, notes = validate_ranges(df)
    assert len(out) == 2
    assert notes["popularity_out_of_range_dropped"] == 2


def test_validate_ranges_sets_invalid_audio_feature_to_na() -> None:
    df = pd.DataFrame(
        {
            "popularity": [50.0, 50.0],
            "tempo": [100.0, 100.0],
            "duration_ms": [200_000, 200_000],
            "danceability": [0.5, 1.5],
            **{k: [0.5, 0.5] for k in AUDIO_BASE if k != "danceability"},
        }
    )
    out, notes = validate_ranges(df)
    assert pd.isna(out.loc[1, "danceability"])
    assert notes["danceability_invalid_set_to_na"] == 1


def test_validate_ranges_drops_out_of_range_tempo_and_duration() -> None:
    df = pd.DataFrame(
        {
            "popularity": [50.0, 50.0, 50.0, 50.0],
            "tempo": [120.0, 30.0, 260.0, 120.0],
            "duration_ms": [200_000, 200_000, 200_000, 10_000],
            **{k: [v, v, v, v] for k, v in AUDIO_BASE.items()},
        }
    )
    out, notes = validate_ranges(df)
    assert len(out) == 1
    assert notes["tempo_out_of_range_dropped"] == 2
    assert notes["duration_ms_out_of_range_dropped"] == 1


def test_drop_missing_required_drops_blank_track_or_artist() -> None:
    df = pd.DataFrame(
        {
            "track_name": ["Song", "", "  "],
            "artist_name": ["Foo", "Bar", "Baz"],
        }
    )
    out, n = drop_missing_required(df)
    assert len(out) == 1
    assert n == 2


def test_harmonize_sources_unknown_slug_raises(tmp_path: Path) -> None:
    csv = tmp_path / "fake.csv"
    csv.write_text("a,b\n1,2\n")
    with pytest.raises(KeyError, match="nonexistent"):
        harmonize_sources([("nonexistent/slug", csv)])


def test_sha256_file_is_deterministic(tmp_path: Path) -> None:
    p = tmp_path / "x.txt"
    p.write_text("hello world")
    assert sha256_file(p) == sha256_file(p)
    assert len(sha256_file(p)) == 64


def test_write_manifest_round_trip(tmp_path: Path) -> None:
    import json

    out = tmp_path / "MANIFEST.json"
    write_manifest(
        [{"slug": "a/b", "rows": 10, "sha256": "x" * 64}],
        out,
    )
    loaded = json.loads(out.read_text())
    assert loaded["sources"][0]["slug"] == "a/b"
    assert "generated_at" in loaded


def test_known_source_loaders_registered() -> None:
    """The three brief-listed slugs must have loaders."""
    for slug in (
        "maharshipandya/-spotify-tracks-dataset",
        "rodolfofigueroa/spotify-12m-songs",
        "paradisejoy/top-hits-spotify-from-20002019",
    ):
        assert slug in SOURCE_LOADERS
