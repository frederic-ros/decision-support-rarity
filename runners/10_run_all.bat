@echo off
cd /d "%~dp0\.."
set CONFIG=%1
if "%CONFIG%"=="" set CONFIG=configs\eswa\full.json
rare-al run --config %CONFIG% --only synthetic rarity real image
rare-al run --config %CONFIG% --only oracle
rare-al run --config %CONFIG% --only observability
