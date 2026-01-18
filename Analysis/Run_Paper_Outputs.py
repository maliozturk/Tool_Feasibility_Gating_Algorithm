# =============================================================================
#  TOOL FEASIBILITY GATING ALGORITHM (TFG)
#  Product Signature: TFG
# ------------------------------------------------------------------------------
#  File: Analysis/Run_Paper_Outputs.py
#  Purpose: Run the analysis script pipeline for paper outputs.
#  Author: Muhammet Ali Ozturk
#  Generated: 2026-01-18
#  Environment: Python 3.9.13
# =============================================================================

import subprocess
import sys
from pathlib import Path


SCRIPT_ORDER = [
    "DP_vs_TFG_V_Sweep.py",
    "Trace_ECDF.py",
    "Policy_GridSearch_vs_TFG.py",
    "Agent_Results_Stats.py",
    "Premium_Priority_Table.py",
    "Journal_Experiments.py",
]


def main() -> int:
    root = Path(__file__).resolve().parent
    py_exe = sys.executable
    for script_name in SCRIPT_ORDER:
        script_path = root / script_name
        if not script_path.exists():
            print(f"[pipeline] missing script: {script_path}", flush=True)
            return 1
        print(f"[pipeline] running {script_name}", flush=True)
        result = subprocess.run([py_exe, str(script_path)], check=False)
        if result.returncode != 0:
            print(f"[pipeline] failed: {script_name} (code={result.returncode})", flush=True)
            return int(result.returncode)
    print("[pipeline] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
