import os
import subprocess
import sys

scripts = [
    "01_ingest_and_audit_corpus.py",
    "02_metric_space_and_tda.py",
    "03_centrality_ablations_and_cases.py",
]

here = os.path.dirname(os.path.abspath(__file__))
for sc in scripts:
    print(f"=== Running {sc} ===")
    subprocess.run([sys.executable, os.path.join(here, sc)], check=True)
print("=== All pipeline steps completed successfully! ===")
