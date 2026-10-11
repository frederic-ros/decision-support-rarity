@echo off
cd /d "%~dp0\.."
set CONFIG=%1
if "%CONFIG%"=="" set CONFIG=configs\eswa\full.json
rare-al prepare-data --config %CONFIG%
