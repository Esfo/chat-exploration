"""
Run every training-stage simulation in order.
Each prints its annotated trace and saves a visualization to visualizations/.
"""

import subprocess
import glob
import os
import sys

here = os.path.dirname(os.path.abspath(__file__))
scripts = sorted(
    f for f in glob.glob(os.path.join(here, "[0-9][0-9]_*.py"))
)

for s in scripts:
    name = os.path.basename(s)
    print("\n" + "#" * 70)
    print(f"# RUNNING {name}")
    print("#" * 70)
    result = subprocess.run([sys.executable, s])
    if result.returncode != 0:
        print(f"!! {name} exited with code {result.returncode}")
        sys.exit(result.returncode)

print("\nAll stages complete. See simulation/visualizations/ for the figures.")
