# Methodology audit for v0.2.0

A static read of the v0.1.0 evaluation pipeline, walking every path test
information can take into training decisions. Each finding names the file
and line in the v0.1.0 tree, states the direction of the bias, and names
the fix and the test that pins it. Line references are to the tree at tag
`v0.1.0`.

## Findings

### 1. Popularity label threshold fit on the full corpus

`src/audio_priors/labels.py:66` computes the sticky cutoff as a quantile
of every available popularity value. The call sites apply it before any
split: `scripts/train.py:74`, `scripts/interpret.py:64`,
`scripts/recommend_eval.py:61`. Test rows therefore move the cutoff that
defines their own labels. The bias is small at this corpus size (an
integer quantile over 82k rows rarely moves when 20% of rows are held
out) but it is structural, and it makes every published metric depend on
test data twice: once through the label, once through evaluation.

Fix: fit the threshold on the train fence only and apply the frozen
scalar everywhere (`labels.sticky_top_q_train_threshold`). Pinned by
`test_label_threshold_is_fit_on_train_only`, which constructs a corpus
where the train-only and full-corpus cutoffs disagree and asserts the
pipeline produces the train-only labels, plus a property-test variant
asserting the threshold is invariant to arbitrary changes in test-only
rows.

### 2. No artist grouping in the split

`scripts/train.py:82-86` uses a random split with no `groups=` argument.
The same artist's tracks can sit on both sides, so a model can score
artist identity (timbre, production, mastering) rather than the
audio-only signal the cold-start framing claims. The size of the
inflation is unknown until measured; both a large and a small measured
gap are reportable results.

Fix: an artist-grouped arm (`GroupShuffleSplit` on `artist_name`)
reported beside the random arm, with the gap estimated by a paired
cluster bootstrap that scores the same artist-disjoint test rows under
both arms' models. Pinned by `test_grouped_split_is_artist_disjoint` and
by a sensitivity check, `test_grouped_auc_below_random_on_coupled_synthetic`,
which builds a corpus with engineered artist-popularity coupling and
fails if the grouped arm does not sit below the random arm there.

### 3. Split stratification depends on the leaky label

`scripts/train.py:83` stratifies on the label from finding 1, so split
membership itself is a function of test popularity statistics. Under the
v2 protocol the label does not exist at split time, so stratification is
dropped in both arms and realized class rates are reported per slice
instead of being forced equal.

### 4. F1 reported at a test-optimal threshold

`src/audio_priors/evaluation.py:70-78` picks the F1-maximizing threshold
on the same arrays it scores, and the bootstrap at line 133 re-optimizes
the threshold inside every resample. The published F1 is an in-sample
upper bound and its interval is the variance of an oracle, not of a
deployable policy.

Fix: the threshold is chosen on a validation carve inside the train
fence (grouped in the grouped arm), frozen, and bootstrapped at that
fixed value; F1 at 0.5 is reported beside it. The oracle figure survives
one release as a single labeled upper bound with no interval. Pinned by
`test_frozen_f1_threshold_is_not_refit_per_resample`, whose separable
construction makes a re-optimizing implementation return a perfect score
at any threshold.

### 5. Row bootstrap on artist-clustered data

`src/audio_priors/evaluation.py:24-55` resamples rows independently.
Tracks cluster by artist, so when the unit of generalization is the
artist, row resampling understates interval width. Fix: grouped-arm
intervals resample artists with replacement and keep all of each sampled
artist's rows; the random arm keeps the row bootstrap as primary and
reports the artist-cluster interval beside it. Degenerate single-class
resamples are skipped deterministically and counted in the output table.
Pinned by `test_grouped_ci_uses_artist_cluster_bootstrap`.

### 6. Retrieval credits same-artist hits

`src/audio_priors/recommend.py:152-159` counts a retrieved track as
relevant when it shares the query's genre or its artist, and no baseline
(`recommend.py:190-283`) excludes the query artist's tracks from the
candidate set. Audio cosine distance is smallest between one artist's
own tracks, so the audio baseline is partly an artist matcher and the
published lift over random is inflated by an unmeasured amount.

Fix: artist-disjoint candidates with the same-artist rows removed from
the relevance denominator as well (otherwise recall has unreachable
relevant items), a genre-only relevance variant, and an artist-grouped
query split. The v0.1 configuration is retained as a comparison row so
the inflation is measured rather than asserted. Pinned by
`test_relevance_mask_coherent_under_exclusion` and
`test_retrieval_underflow_small_corpus`.

### 7. Calibration fit is clean but the card mixes protocols

`scripts/interpret.py` fits isotonic calibration on the training side
only, which is correct in isolation. The risk is protocol mixing: if the
metrics tables move to the v2 protocol while the calibration and
attribution tables stay on v0.1 splits and labels, the model card quotes
two incompatible experiments side by side. Fix: `scripts/interpret.py`
is ported in the same change as the protocol.

### 8. Near-duplicate exposure in the random arm

Exact duplicates on the normalized `(track_name, artist_name)` key are
collapsed at corpus build (`src/audio_priors/data.py`), and artist
grouping keeps any same-artist variants on one side of the grouped
split. The residual surface is cross-artist near-duplicates (covers,
compilation re-releases) crossing the random split. The count is
audited and reported; a handling layer is added only if it is material.
Pinned for the guaranteed part by
`test_no_normalized_duplicate_crosses_random_split`.

## Reading order for the fixes

Findings 1 to 5 land together (they are one protocol), finding 6 lands
next, finding 7 inside the first change, finding 8 as an audit number in
the findings write-up. The fixed protocol, its frozen hyperparameters,
and the bootstrap units are specified in `MODEL.md` and enforced by
`tests/test_invariants.py`, which runs as a blocking CI job.
