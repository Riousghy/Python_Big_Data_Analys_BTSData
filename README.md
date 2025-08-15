# Python_Big_Data_Analys_BTSData

Week 3 baselines & evaluation on BTS On-Time dataset.

## Quickstart
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/run_week3.sh

## CLI
python week3_baselines.py --root ./bts_on_time_data --sample eda/sample_100k_week2_features.parquet

## Notes
- Recompute route/carrier profiles on TRAIN only (leakage-safe).
- Keep raw data in ./bts_on_time_data/ (ignored by .gitignore).
