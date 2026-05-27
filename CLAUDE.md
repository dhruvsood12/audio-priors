# CLAUDE.md

Per-phase status and notes for the audio-priors rebuild. Maintained as work
proceeds. The full contract lives in `PROJECT_BRIEF.md` (kept locally by Dhruv,
not committed).

## Phase status

- [x] Phase 0: Hygiene and identity
- [x] Phase 1: Data ingestion and validation
- [x] Phase 2: EDA and label design
- [x] Phase 3: Modeling
- [x] Phase 4: Interpretability and calibration
- [x] Phase 5: Cold-start recommender
- [x] Phase 6: Streamlit demo
- [x] Phase 7: Test coverage sweep
- [x] Phase 8: CI, Docker, hooks
- [x] Phase 9: README, DATA.md, MODEL.md
- [ ] Phase 10: Final pass and release

## Current focus

Phase 6 is complete on the `feat/phase-6-app` branch. PR open, waiting
for Dhruv's approval before opening Phase 7.

## Open questions

1. **AI-tool trailers in legacy commits.** Brief Section 2.2 forbids any AI
   co-author trailer in commit messages. Pre-rebuild history contains 12
   commits with such trailers in their bodies. Brief Phase 10 schedules a
   `git filter-repo` pass to confirm authorship; that is the appropriate
   place to strip those trailers as well. Phase 0 leaves history untouched.
2. **Existing `v0.1.0` tag and release.** The renamed repo carries a `v0.1.0`
   tag and a published GitHub Release from the prior framing. Phase 10 plans
   to tag a fresh `v0.1.0` for the audio-priors rebuild. Resolution options:
   delete the existing release and retag at the rebuild HEAD, or move forward
   under `v0.2.0`. Defer until Phase 10.
3. **NumPy upper bound.** Brief Section 3 pins `numpy>=1.26,<2.1`. Local
   Python 3.13 has no NumPy 2.0.x wheel and the source build fails on Apple
   Clang. Phase 0 relaxed the upper bound to `numpy>=1.26` per brief Section
   3's "verify latest compatible versions at execution time" guidance. If
   reverting to a strict ceiling matters for Phase 8 CI on 3.10/3.11/3.12,
   reintroduce the bound there once wheels exist.
4. **Local clone directory name.** Working tree is still at
   `/Users/dhruvsood/SongAddiction`. The remote was renamed to `audio-priors`.
   Renaming the local directory is cosmetic and was skipped to avoid breaking
   shell history and editor state.

## Conventions in effect

- Branch per phase: `feat/phase-N-short-slug`.
- One PR per phase, merged after every acceptance box is checked.
- Conventional Commits: `feat:`, `fix:`, `chore:`, `docs:`, `test:`, `refactor:`,
  `ci:`. Subjects in imperative mood, plain American English.
- No em-dashes anywhere in `*.md` (grep gate in Section 7).
- No words from the brief's banned list in `*.md`.
- Every metric in docs traces to a file in `outputs/tables/` produced by this
  rebuild, with confidence intervals.

## Next-phase entry condition

Phase 1 begins on `feat/phase-1-data` only after Dhruv approves the Phase 0
PR and explicitly says go.
