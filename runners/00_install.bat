@echo off
cd /d "%~dp0\.."
python -m pip install -e .
echo Core installed. GPU image extras: pip install -e ".[images]"
