# Contributing

Bug reports and focused pull requests are welcome. Please open an issue before a
change that alters scientific defaults, output schemas, or filtering behavior.

## Development setup

```bash
conda env create -f environment.yml
conda activate swot-raqw
python -m pip install --no-deps -e ".[test]"
python -m pytest
```

Every behavior change should include a synthetic regression test. Changes to a
publication parameter must also update `configs/publication_2024.toml`, the
README, and the Supporting Information parameter table. Never commit Earthdata
credentials or restricted/local data products.

