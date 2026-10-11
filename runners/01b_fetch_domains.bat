@echo off
cd /d "%~dp0\.."
python scripts\fetch_domains.py %*
