"""Invariant tests that enforce the v2 evaluation protocol.

Each test pins one property of the protocol: artist-disjoint grouped
splits, a popularity threshold fit on train rows only, an F1 threshold
frozen before the bootstrap, the artist as the grouped-arm resampling
unit, and the dedup contract under the random split. Each carries a
behavioral discriminator constructed so that a compiling-but-wrong
implementation (old semantics under new names) fails, not just an
import-level check.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st
from sklearn.metrics import roc_auc_score

from audio_priors import data as ap_data
from audio_priors import models
from audio_priors.evaluation import (
    bootstrap_metric,
    evaluate_with_ci,
    f1_at_threshold,
    f1_in_sample_oracle,
    pick_f1_threshold_on_train,
)
from audio_priors.labels import sticky_top_q, sticky_top_q_train_threshold
from audio_priors.splits import protocol_split


def _artist_corpus(n_artists: int = 60, tracks_per_artist: int = 5, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for a in range(n_artists):
        for t in range(tracks_per_artist):
            rows.append(
                {
                    "track_name": f"track_{a}_{t}",
                    "artist_name": f"artist_{a}",
                    "popularity": int(rng.integers(0, 101)),
                }
            )
    return pd.DataFrame(rows)


def _artist_sets(df: pd.DataFrame, positions: np.ndarray) -> set[str]:
    return set(df.iloc[positions]["artist_name"])


def test_grouped_split_is_artist_disjoint() -> None:
    df = _artist_corpus()
    sp = protocol_split(df, "grouped")
    fit_a = _artist_sets(df, sp.train_fit)
    val_a = _artist_sets(df, sp.val)
    test_a = _artist_sets(df, sp.test)
    assert fit_a & val_a == set()
    assert fit_a & test_a == set()
    assert val_a & test_a == set()
    # Exact partition of the row positions.
    union = np.sort(np.concatenate([sp.train_fit, sp.val, sp.test]))
    assert np.array_equal(union, np.arange(len(df)))
    assert 0.12 <= len(sp.test) / len(df) <= 0.28


@settings(max_examples=50, deadline=None)
@given(
    sizes=st.lists(st.integers(min_value=1, max_value=12), min_size=12, max_size=40),
)
def test_grouped_split_disjoint_for_arbitrary_artist_sizes(sizes: list[int]) -> None:
    rows = []
    for a, k in enumerate(sizes):
        for t in range(k):
            rows.append({"artist_name": f"artist_{a}", "popularity": (a * 7 + t) % 101})
    df = pd.DataFrame(rows)
    sp = protocol_split(df, "grouped")
    assert _artist_sets(df, sp.train_fit) & _artist_sets(df, sp.test) == set()
    assert _artist_sets(df, sp.train_fit) & _artist_sets(df, sp.val) == set()
    assert _artist_sets(df, sp.val) & _artist_sets(df, sp.test) == set()
    union = np.sort(np.concatenate([sp.train_fit, sp.val, sp.test]))
    assert np.array_equal(union, np.arange(len(df)))


def test_random_split_leaks_artists_grouped_does_not() -> None:
    """Documents WHY the grouped arm exists: the random arm shares artists."""

    df = _artist_corpus(n_artists=40, tracks_per_artist=10, seed=1)
    sp_random = protocol_split(df, "random")
    sp_grouped = protocol_split(df, "grouped")
    shared_random = _artist_sets(df, sp_random.train_fit) & _artist_sets(df, sp_random.test)
    shared_grouped = _artist_sets(df, sp_grouped.train_fit) & _artist_sets(df, sp_grouped.test)
    assert shared_random != set()  # 40 artists x 10 tracks: sharing is near-certain
    assert shared_grouped == set()


def test_label_threshold_is_fit_on_train_only() -> None:
    """Discriminator: train-only and full-corpus thresholds disagree on the
    probe rows. Train median is 4.5; the high test rows drag the full-corpus
    median up to 7.5, so the 4.8 probes flip depending on which fence fit
    the threshold."""

    train = pd.DataFrame({"popularity": list(range(10))})  # rows 0-9, median 4.5
    probe = pd.DataFrame({"popularity": [4.8] * 3})  # rows 10-12
    high = pd.DataFrame({"popularity": [100.0] * 9})  # rows 13-21
    df = pd.concat([train, probe, high], ignore_index=True)
    train_index = df.index[:10]
    probe_rows = slice(10, 13)

    y, threshold = sticky_top_q_train_threshold(df, train_index, q=0.5)
    assert threshold == float(df.loc[train_index, "popularity"].quantile(0.5))  # 4.5
    assert bool(y.iloc[probe_rows].eq(1).all())  # 4.8 >= 4.5

    # The legacy full-corpus path puts the cutoff ABOVE the probes (the
    # median of all 22 rows is 7.5), labeling them zero. If the new function
    # secretly used the old fence, the assertions above would fail.
    y_legacy = sticky_top_q(df, q=0.5)
    assert bool(y_legacy.iloc[probe_rows].eq(0).all())
    assert bool((y != y_legacy).any())


@settings(max_examples=50, deadline=None)
@given(
    test_pop=st.lists(st.integers(min_value=0, max_value=100), min_size=1, max_size=30),
    q=st.floats(min_value=0.05, max_value=0.95),
)
def test_train_threshold_invariant_to_test_rows(test_pop: list[int], q: float) -> None:
    """The threshold and train labels never move when test rows change."""

    train = pd.DataFrame({"popularity": [3, 11, 27, 35, 42, 58, 66, 71, 88, 94]})
    df_a = pd.concat([train, pd.DataFrame({"popularity": test_pop})], ignore_index=True)
    df_b = pd.concat([train, pd.DataFrame({"popularity": [0] * len(test_pop)})], ignore_index=True)
    train_index = df_a.index[: len(train)]

    y_a, thr_a = sticky_top_q_train_threshold(df_a, train_index, q=q)
    y_b, thr_b = sticky_top_q_train_threshold(df_b, train_index, q=q)
    assert thr_a == thr_b
    assert y_a.iloc[: len(train)].equals(y_b.iloc[: len(train)])


def test_frozen_f1_threshold_is_not_refit_per_resample() -> None:
    """Perfectly separable data is the exact frozen-vs-oracle discriminator."""

    rng = np.random.default_rng(0)
    score = rng.uniform(0, 1, size=400)
    y = (score >= 0.5).astype(int)

    rep_natural = evaluate_with_ci(y, score, n_resamples=200, f1_threshold=0.5)
    assert rep_natural.metrics["f1_frozen"] == (1.0, 1.0, 1.0)

    rep_off = evaluate_with_ci(y, score, n_resamples=200, f1_threshold=0.9)
    point, _, hi = rep_off.metrics["f1_frozen"]
    assert point == f1_at_threshold(y, score, 0.9)
    assert point < 1.0
    # A re-optimizing implementation recovers F1=1.0 in every resample of
    # separable data regardless of the passed threshold; the upper bound
    # below is impossible under that behavior.
    assert hi < 1.0


@settings(max_examples=50, deadline=None)
@given(
    seed=st.integers(min_value=0, max_value=10_000),
    threshold=st.floats(min_value=0.05, max_value=0.95),
)
def test_frozen_f1_point_is_exactly_at_passed_threshold(seed: int, threshold: float) -> None:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=120)
    if len(np.unique(y)) < 2:
        y[0], y[1] = 0, 1
    score = y * 0.4 + rng.uniform(0, 0.6, size=120)
    rep = evaluate_with_ci(y, score, n_resamples=20, f1_threshold=threshold)
    assert rep.metrics["f1_frozen"][0] == f1_at_threshold(y, score, threshold)


@settings(max_examples=50, deadline=None)
@given(
    seed=st.integers(min_value=0, max_value=10_000),
    threshold=st.floats(min_value=0.0, max_value=1.0),
)
def test_oracle_dominates_any_fixed_threshold(seed: int, threshold: float) -> None:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=80)
    if len(np.unique(y)) < 2:
        y[0], y[1] = 0, 1
    score = rng.uniform(0, 1, size=80)
    assert f1_in_sample_oracle(y, score) >= f1_at_threshold(y, score, threshold) - 1e-12


def test_grouped_ci_uses_artist_cluster_bootstrap() -> None:
    """Resamples are unions of whole artists, and cluster CIs are wider
    than row CIs when rows within an artist are correlated."""

    # Part 1: whole-group resampling, observed through the metric_fn. Each
    # artist is two rows (one of each class) sharing one unique score
    # value, so any resample built from whole artists pairs every score
    # value with equal positives and negatives. A row bootstrap breaks the
    # pairing almost surely.
    n_groups = 40
    y = np.array([0, 1] * n_groups)
    score = np.repeat(np.linspace(0.01, 0.99, n_groups), 2)
    groups = np.repeat([f"artist_{i}" for i in range(n_groups)], 2)

    def pairing_checker(yt: np.ndarray, st_: np.ndarray) -> float:
        frame = pd.DataFrame({"y": yt, "s": st_})
        per_score = frame.groupby("s")["y"].agg(["sum", "count"])
        assert (per_score["count"] == 2 * per_score["sum"]).all(), (
            "resample is not a union of whole artist groups"
        )
        return 0.5

    bootstrap_metric(y, score, pairing_checker, n_resamples=50, groups=groups)

    # Part 2: width. A strong per-artist random effect drives both label
    # and score, so rows within an artist are nearly copies; treating them
    # as independent shrinks the interval.
    rng = np.random.default_rng(7)
    n_art, per = 50, 20
    u = rng.normal(0, 1.0, size=n_art)
    y2 = (np.repeat(u, per) + rng.normal(0, 0.1, size=n_art * per) > 0).astype(int)
    s2 = np.repeat(u, per) + rng.normal(0, 0.3, size=n_art * per)
    g2 = np.repeat([f"a{i}" for i in range(n_art)], per)

    _, lo_row, hi_row = bootstrap_metric(y2, s2, roc_auc_score, n_resamples=400, seed=42)
    _, lo_cl, hi_cl = bootstrap_metric(y2, s2, roc_auc_score, n_resamples=400, seed=42, groups=g2)
    assert (hi_cl - lo_cl) > (hi_row - lo_row)


def test_grouped_auc_below_random_on_coupled_synthetic() -> None:
    """Sensitivity check: when features fingerprint artists and the label is
    artist-driven, the random arm MUST beat the grouped arm by a margin. A
    protocol port that yields near-identical arms on this corpus is broken
    even if it passes every structural test."""

    rng = np.random.default_rng(11)
    n_art, per = 40, 15
    quality = rng.normal(0, 1, size=n_art)
    fingerprint = rng.normal(0, 1, size=(n_art, 5))
    rows = []
    for a in range(n_art):
        for _ in range(per):
            feats = fingerprint[a] + rng.normal(0, 0.05, size=5)
            rows.append(
                {
                    "artist_name": f"artist_{a}",
                    "popularity": float(50 + 20 * quality[a] + rng.normal(0, 1)),
                    **{f"f{i}": feats[i] for i in range(5)},
                }
            )
    df = pd.DataFrame(rows)
    feature_cols = [f"f{i}" for i in range(5)]

    aucs: dict[str, float] = {}
    for kind in ("random", "grouped"):
        sp = protocol_split(df, kind, random_state=3)
        y_all, _ = sticky_top_q_train_threshold(df, df.index[sp.train], q=0.4)
        m = models.train_random_forest(
            df.iloc[sp.train_fit][feature_cols], y_all.iloc[sp.train_fit], n_estimators=50
        )
        s = m.predict_proba(df.iloc[sp.test][feature_cols])[:, 1]
        aucs[kind] = float(roc_auc_score(y_all.iloc[sp.test], s))

    assert aucs["random"] - aucs["grouped"] >= 0.15, aucs


def test_no_normalized_duplicate_crosses_random_split() -> None:
    """The dedup contract: after data.deduplicate, no two rows share the
    normalized (track_name, artist_name) key, so no duplicate can cross
    ANY split. Pins the corpus-build + split interaction against future
    loader changes."""

    rng = np.random.default_rng(5)
    base = pd.DataFrame(
        {
            "track_name": [f"Song {i}" for i in range(200)],
            "artist_name": [f"Artist {i % 40}" for i in range(200)],
            "popularity": rng.integers(0, 101, size=200).astype(float),
        }
    )
    # Casing/whitespace variants of the first 50 rows: exact normalized dupes.
    variants = base.head(50).copy()
    variants["track_name"] = variants["track_name"].str.upper() + " "
    raw = pd.concat([base, variants], ignore_index=True)

    deduped, _ = ap_data.deduplicate(raw)
    deduped = deduped.reset_index(drop=True)
    sp = protocol_split(deduped, "random")

    def norm_keys(positions: np.ndarray) -> set[tuple[str, str]]:
        sub = deduped.iloc[positions]
        return set(
            zip(
                sub["track_name"].str.strip().str.lower(),
                sub["artist_name"].str.strip().str.lower(),
                strict=True,
            )
        )

    assert norm_keys(sp.train_fit) & norm_keys(sp.test) == set()
    assert norm_keys(sp.train) & norm_keys(sp.test) == set()


def test_protocol_end_to_end_smoke() -> None:
    """splits + labels + frozen threshold + evaluate, both arms, one model."""

    rng = np.random.default_rng(2)
    n_art, per = 50, 40
    rows = []
    for a in range(n_art):
        for _t in range(per):
            d = rng.uniform(0, 1)
            rows.append(
                {
                    "artist_name": f"artist_{a}",
                    "genre": ["pop", "rock", "jazz"][a % 3],
                    "popularity": float(np.clip(60 * d + rng.normal(0, 15), 0, 100)),
                    "danceability": d,
                    "energy": rng.uniform(0, 1),
                }
            )
    df = pd.DataFrame(rows)
    feature_cols = ["danceability", "energy"]

    for kind in ("random", "grouped"):
        sp = protocol_split(df, kind)
        y_all, pop_thr = sticky_top_q_train_threshold(df, df.index[sp.train], q=0.3)
        assert pop_thr == float(df.iloc[sp.train]["popularity"].quantile(0.7))
        if kind == "grouped":
            assert _artist_sets(df, sp.train_fit) & _artist_sets(df, sp.test) == set()

        m = models.train_logistic(df.iloc[sp.train_fit][feature_cols], y_all.iloc[sp.train_fit])
        f1_thr = pick_f1_threshold_on_train(
            y_all.iloc[sp.val].to_numpy(),
            m.predict_proba(df.iloc[sp.val][feature_cols])[:, 1],
        )
        groups = df.iloc[sp.test]["artist_name"].to_numpy() if kind == "grouped" else None
        rep = evaluate_with_ci(
            y_all.iloc[sp.test].to_numpy(),
            m.predict_proba(df.iloc[sp.test][feature_cols])[:, 1],
            n_resamples=100,
            f1_threshold=f1_thr,
            groups=groups,
        )
        assert set(rep.metrics) == {"roc_auc", "pr_auc", "brier", "f1_at_05", "f1_frozen"}
        for name, (point, lo, hi) in rep.metrics.items():
            assert lo <= point <= hi, f"{kind}/{name}: {lo} <= {point} <= {hi}"
        assert 0.0 <= rep.f1_oracle <= 1.0
        assert rep.n_degenerate_resamples == 0
