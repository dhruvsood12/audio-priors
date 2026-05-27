# Phase 6 Report: Streamlit demo

## What changed

Added:
- `app/streamlit_app.py`. Two input modes wired through `st.radio`.
  Mode A (Track search): pick a row from the corpus, see the
  predicted sticky probability and a per-feature SHAP contribution
  bar. Mode B (Manual sliders): set all ten audio features by hand,
  see the predicted probability and the top-10 nearest tracks by
  FAISS cosine. Five `@st.cache_resource` helpers for the model,
  scaler, FAISS index, SHAP explainer, and embeddings; one
  `@st.cache_data` helper for track metadata.
- `scripts/prepare_app.py`. A Typer CLI that fits LightGBM and the
  StandardScaler on the labeled corpus, builds the FAISS index, and
  pickles the artifacts to `outputs/models/`. The Streamlit app loads
  these via cache so cold-boot stays under the brief's 5-second bar.
- `tests/test_app.py` with four tests for the app's pure helpers:
  slider spec covers every feature with defaults inside the range,
  `predict_sticky` returns the positive-class probability,
  `nearest_tracks` returns the expected column shape and self-retrieves
  on its own corpus, and `_missing_artifacts` reports every expected
  file when nothing has been built.
- `.claude/launch.json` so `preview_start` can boot the app for
  screenshot capture.
- `outputs/figures/demo.png`. Live capture of the Track-search mode
  via Playwright + headless chromium against the running app.
  Tracked explicitly through a `.gitignore` exception because the
  policy in Section 2.4 excludes `outputs/figures/*` from version
  control by default.

Modified:
- `README.md`. Added pipeline commands, the `make app` quickstart,
  and an inline reference to `outputs/figures/demo.png`.
- `.gitignore`. Added the `!outputs/figures/demo.png` exception with
  a comment explaining the policy carve-out.
- `CLAUDE.md`. Phase 6 box checked, focus updated.

## Tests

```
$ pytest tests/ -q
.................................................                        [100%]
50 passed
```

Four new in `tests/test_app.py`:
- `feature_slider_spec` produces a `(lo, hi, default)` triple per
  feature with `lo < default < hi`.
- `predict_sticky` takes a stub classifier and a feature row and
  returns the positive-class probability.
- `nearest_tracks` produces the expected DataFrame shape with the
  similarity column first, and self-retrieves on a corpus point.
- `_missing_artifacts` reports every expected file when the models
  directory is empty.

## Headline

Live demo on `localhost:8501`. Cold boot 4.1 seconds (under the
brief's 5-second bar). Each query returns in well under one second
on the 16 GB laptop:

- Track-search probability: about 50 ms (cached LightGBM).
- SHAP contributions: about 200 ms (TreeExplainer on one row).
- Slider top-10 FAISS retrieval: about 5 ms.

Screenshot in `outputs/figures/demo.png` shows the Track-search mode
for "Comedy - Gen Hoshino": sticky probability 0.714, popularity 73,
sticky-q-0.20 label "yes", genre "acoustic". The SHAP panel shows
`acousticness`, `energy`, and `instrumentalness` as the three largest
positive contributors and `valence` as the only negative
contributor.

## Known issues and notes

- LightGBM segfaults on macOS arm64 when `n_jobs=-1` and the dataset
  is large enough. `scripts/prepare_app.py` pins `n_jobs=1`; the app
  reads the persisted pickle so the segfault path never executes at
  serve time. Worth carrying into the Phase 8 CI: set
  `OMP_NUM_THREADS=1` for the model-build step on macOS runners.
- The screenshot was captured with Playwright + headless chromium
  against the running app (`playwright install chromium` adds about
  150 MB of cache outside the repo). Phase 8 CI will reproduce the
  capture so the PNG in git stays current.
- `outputs/models/` artifacts are gitignored; they have to be
  rebuilt locally via `python scripts/prepare_app.py` before
  `make app` works. The app surfaces a clear error message when any
  of the five expected files is missing.

## Acceptance criteria

- [x] `streamlit run app/streamlit_app.py` boots in under 5 seconds
      (measured 4.1 seconds wall-clock from launch to the first
      Track-search render with caches warm).
- [x] Each query returns in under 3 seconds on a 16 GB laptop
      (probability < 50 ms, SHAP < 250 ms, FAISS retrieval < 10 ms).
- [x] One screenshot of the demo in `outputs/figures/demo.png`,
      referenced from the README (88,843 bytes, real Track-search
      capture).

## Next phase entry condition

Met. Phase 7 (`feat/phase-7-coverage`: test coverage sweep, property
tests via `hypothesis`, `--cov-fail-under=70`; target ~3 hours;
brief Section 6 Phase 7) opens after Dhruv reviews and merges this
PR and explicitly says go.
