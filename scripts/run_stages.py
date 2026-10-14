"""Run pipeline stages sequentially, detached from any console (Windows-safe, resumable).

    pythonw scripts/run_stages.py uci:ablation uci:analyze uci:benchmark synthetic:experiment ...

Each stage runs `python -m healthgraph <stage> --config configs/<cfg>.yaml` in its own process
group without a console window, appending output to results/<cfg>/log.txt, so console Ctrl+C or
close events cannot interrupt it. Stages resume from their own checkpoints.
"""
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON = os.path.join(os.path.dirname(sys.executable), "python.exe") if os.name == "nt" else sys.executable
FLAGS = (subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP) if os.name == "nt" else 0


def main(stages):
    for item in stages:
        cfg, stage = item.split(":")
        out_dir = os.path.join(ROOT, "results", cfg)
        os.makedirs(out_dir, exist_ok=True)
        log = os.path.join(out_dir, "log.txt")
        with open(log, "a", encoding="utf-8") as f:
            f.write(f"\n=== {stage} started {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
            f.flush()
            code = subprocess.call([PYTHON, "-u", "-m", "healthgraph", stage, "--config", f"configs/{cfg}.yaml"],
                                   cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                   creationflags=FLAGS)
            f.write(f"=== {stage} EXIT {code} {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")


if __name__ == "__main__":
    main(sys.argv[1:])
