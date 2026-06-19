"""Protocol splits: train_fit / val / test with a random and a grouped arm.

The v2 evaluation protocol splits BEFORE any label exists, so neither arm
stratifies. The grouped arm keeps every artist's tracks on one side of
every fence (``GroupShuffleSplit`` on ``artist_name``), which is what the
cold-start framing claims; the random arm uses the same structure without
groups so the arm delta isolates grouping alone.

Slice shares are ``(1 - test_size) * (1 - val_size)`` / ``(1 - test_size)
* val_size`` / ``test_size`` of the corpus; the defaults give 64/16/20.
The validation carve exists to freeze the F1 decision threshold, so in
the grouped arm it is itself artist-grouped: an artist shared between
``train_fit`` and ``val`` would let memorized score inflation tune the
threshold.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit

SPLIT_KINDS = ("random", "grouped")
_UNKNOWN_GROUP = "__unknown_artist__"


@dataclass(frozen=True)
class ProtocolSplit:
    """Positional indices into the frame handed to :func:`protocol_split`."""

    kind: str
    train_fit: np.ndarray
    val: np.ndarray
    test: np.ndarray

    @property
    def train(self) -> np.ndarray:
        """The label-threshold fence: ``train_fit`` plus ``val``."""

        return np.concatenate([self.train_fit, self.val])


def protocol_split(
    df: pd.DataFrame,
    kind: str,
    *,
    group_col: str = "artist_name",
    test_size: float = 0.20,
    val_size: float = 0.20,
    random_state: int = 42,
) -> ProtocolSplit:
    """Split ``df`` into train_fit / val / test under the named protocol arm.

    ``kind`` is ``"random"`` or ``"grouped"``. ``val_size`` is the fraction
    of the TRAIN side carved out for validation, so the defaults yield
    64/16/20 of the corpus. Neither arm stratifies: the sticky label does
    not exist at split time under the train-only-threshold protocol.

    Returns positional indices (``np.ndarray`` of int) into ``df``'s row
    order, not index labels.
    """

    if kind not in SPLIT_KINDS:
        raise ValueError(f"kind must be one of {SPLIT_KINDS}, got {kind!r}")
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be in (0, 1), got {test_size}")
    if not 0.0 < val_size < 1.0:
        raise ValueError(f"val_size must be in (0, 1), got {val_size}")

    positions = np.arange(len(df))
    if kind == "grouped":
        groups = df[group_col].fillna(_UNKNOWN_GROUP).astype(str).to_numpy()
        outer = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_pos, test_pos = next(outer.split(positions, groups=groups))
        inner = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state + 1)
        fit_rel, val_rel = next(inner.split(train_pos, groups=groups[train_pos]))
    else:
        outer_r = ShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_pos, test_pos = next(outer_r.split(positions))
        inner_r = ShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state + 1)
        fit_rel, val_rel = next(inner_r.split(train_pos))

    return ProtocolSplit(
        kind=kind,
        train_fit=np.sort(train_pos[fit_rel]),
        val=np.sort(train_pos[val_rel]),
        test=np.sort(test_pos),
    )


def assert_min_class_counts(
    y: pd.Series,
    split: ProtocolSplit,
    min_count: int = 50,
) -> None:
    """Fail loudly if any slice has fewer than ``min_count`` of either class.

    Called after labels are derived (they do not exist at split time).
    """

    arr = y.to_numpy()
    for name, pos in (
        ("train_fit", split.train_fit),
        ("val", split.val),
        ("test", split.test),
    ):
        slice_y = arr[pos]
        n_pos = int(slice_y.sum())
        n_neg = int(len(slice_y) - n_pos)
        if n_pos < min_count or n_neg < min_count:
            raise ValueError(
                f"slice {name!r} has {n_pos} positives / {n_neg} negatives; "
                f"need at least {min_count} of each"
            )
