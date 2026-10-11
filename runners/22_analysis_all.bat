@echo off
cd /d "%~dp0\.."
set CONFIG=%1
if "%CONFIG%"=="" set CONFIG=configs\eswa\campaign.json
rare-al analyze --config %CONFIG%
rare-al validate-guidelines --config %CONFIG%
echo ANALYSIS DONE -> results (assets + validation + tab_validation.tex)
