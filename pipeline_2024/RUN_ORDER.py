"""Run order for the clean 2024 pipeline.

This folder is the new working area. The repo-root notebooks are left alone and
copied/extracted only as reference material under ``legacy_notebook_code``.

Suggested order:

1. ``python -m pipeline_2024.run_01_reaches_with_discharge``
   Builds ``data/Petrohue_SWORD_reaches/chile_reaches_with_valid_discharge.csv``
   from the Chile SWORD geopackage plus DAWG/SOS discharge data.

2. ``python -m pipeline_2024.run_02_preprocess_2024_pixc``
   Downloads/processes 2024 PIXC for the valid-discharge reaches and writes the
   processed reach/year folders under ``data/pipeline_2024``.

3. ``python pipeline_2024/train_tau_filter_2024.py``
   Trains the updated tau-window selection from the 2024 processed PIXC outputs.

4. ``python -m pipeline_2024.run_04_compare_with_riversp``
   Compares raw PIXC, updated filter results, and RiverSP slopes.

2025 application is deliberately excluded for now.
"""

