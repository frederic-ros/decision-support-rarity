@echo off
cd /d "%~dp0\.."
set CONFIG=%1
if "%CONFIG%"=="" set CONFIG=configs\eswa\campaign.json
rare-al validate-guidelines --config %CONFIG%
