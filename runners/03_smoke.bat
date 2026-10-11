@echo off
cd /d "%~dp0\.."
python tests\make_smoke_fixtures.py
rare-al run --config configs\smoke\smoke.json --only synthetic rarity real image
rare-al run --config configs\smoke\smoke.json --only oracle
rare-al run --config configs\smoke\smoke.json --only observability
rare-al analyze --config configs\smoke\smoke.json
echo SMOKE OK
