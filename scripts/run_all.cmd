@echo off
REM Full reproducible run (Windows). Each stage resumes safely: Optuna studies are persisted in SQLite.
cd /d %~dp0\..
python -u -m healthgraph all --config configs/uci.yaml --trials 16 > results\uci\log.txt 2>&1
echo UCI EXIT %errorlevel% >> results\uci\log.txt
if not exist results\synthetic mkdir results\synthetic
python -u -m healthgraph experiment --config configs/synthetic.yaml > results\synthetic\log.txt 2>&1
python -u -m healthgraph ablation --config configs/synthetic.yaml >> results\synthetic\log.txt 2>&1
python -u -m healthgraph analyze --config configs/synthetic.yaml >> results\synthetic\log.txt 2>&1
echo SYNTHETIC EXIT %errorlevel% >> results\synthetic\log.txt
