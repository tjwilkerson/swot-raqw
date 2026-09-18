from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
PIPELINE_DATA_ROOT = DATA_ROOT / "pipeline_2024"

SWORD_DIR = DATA_ROOT / "Petrohue_SWORD_reaches"
CHILE_REACHES_GPKG = SWORD_DIR / "chile_reaches.gpkg"
VALID_DISCHARGE_REACHES_CSV = SWORD_DIR / "chile_reaches_with_valid_discharge.csv"

RUN_ROOT = PIPELINE_DATA_ROOT / "chile_reaches_with_valid_discharge_run"
PROCESSED_ROOT = RUN_ROOT / "processed"
TAU_EXPERIMENT_ROOT = PIPELINE_DATA_ROOT / "tau_range_update_experiment"
RIVERSP_COMPARISON_ROOT = PIPELINE_DATA_ROOT / "riversp_comparison_updated_filter"

