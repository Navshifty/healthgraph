"""List number macros used in the report that make_tables.py has not defined (yet)."""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
tex = open(os.path.join(HERE, "healthgraph_report.tex"), encoding="utf-8").read()
nums = open(os.path.join(HERE, "tables", "numbers.tex"), encoding="utf-8").read()
defined = set(re.findall(r"newcommand\{\\([A-Za-z]+)\}", nums))
builtin = set(re.findall(r"newcommand\{\\([A-Za-z]+)\}", tex))
prefixes = ("auroc", "auprc", "ece", "brier", "sens", "ppv", "topdec", "bench", "abl", "syn", "hom", "att",
            "err", "sub", "best", "nTrials", "nPruned", "nComplete", "nInterrupted", "nEdges", "nFeatures", "nRef",
            "prevalence", "pcaVar", "threshold", "shapModel", "edges")
used = {u for u in re.findall(r"\\([A-Za-z]+)", tex) if u.startswith(prefixes)}
missing = sorted(used - defined - builtin)
print("missing macros:", missing if missing else "none")
