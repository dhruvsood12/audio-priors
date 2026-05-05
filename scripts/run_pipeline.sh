#!/usr/bin/env bash
# Headless end-to-end run: ensure data exists, then execute the three notebooks.

set -euo pipefail

mkdir -p data/raw data/processed outputs/figures outputs/tables

python scripts/make_demo_data.py

for nb in 01_data_cleaning 02_eda 03_modeling; do
    jupyter nbconvert --to notebook --execute "notebooks/${nb}.ipynb" \
        --inplace --ExecutePreprocessor.timeout=600
done

ls -la outputs/figures outputs/tables
