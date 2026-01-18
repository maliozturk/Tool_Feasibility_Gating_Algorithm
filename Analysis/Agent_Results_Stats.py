# =============================================================================
#  TOOL FEASIBILITY GATING ALGORITHM (TFG)
#  Product Signature: TFG
# ------------------------------------------------------------------------------
#  File: Analysis/Agent_Results_Stats.py
#  Purpose: Summarize agent trace statistics for reporting.
#  Author: Muhammet Ali Ozturk
#  Generated: 2026-01-18
#  Environment: Python 3.9.13
# =============================================================================

import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from Configurations import Simulation_Config


@dataclass
class SummaryRow:
    mode: str
    n: int
    mean: float
    std: float
    p50: float
    p90: float
    p95: float
    p99: float


def _Summary(values: Sequence[float], mode: str) -> SummaryRow:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return SummaryRow(mode, 0, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))
    return SummaryRow(
        mode=mode,
        n=int(arr.size),
        mean=float(np.mean(arr)),
        std=float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        p50=float(np.percentile(arr, 50)),
        p90=float(np.percentile(arr, 90)),
        p95=float(np.percentile(arr, 95)),
        p99=float(np.percentile(arr, 99)),
    )


def _Read_Trace(
    csv_path: str,
    fast_col: str,
    slow_col: str,
    fast_error_col: str,
    slow_error_col: str,
    drop_error_rows: bool,
) -> Dict[str, List[float]]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Trace CSV not found: {csv_path}")

    fast_vals: List[float] = []
    slow_vals: List[float] = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if drop_error_rows:
                if fast_error_col and row.get(fast_error_col):
                    continue
                if slow_error_col and row.get(slow_error_col):
                    continue

            try:
                fast_v = float(row.get(fast_col, "nan"))
                slow_v = float(row.get(slow_col, "nan"))
            except ValueError:
                continue

            if fast_v > 0:
                fast_vals.append(fast_v)
            if slow_v > 0:
                slow_vals.append(slow_v)

    return {"fast": fast_vals, "slow": slow_vals}


def _Take_First(values: Sequence[float], n: int) -> List[float]:
    return list(values[:n])


def main() -> None:
    sim_cfg = Simulation_Config()
    svc_cfg = sim_cfg.service_config

    csv_path = svc_cfg.trace_csv_path_str
    fast_col = svc_cfg.trace_fast_column_str
    slow_col = svc_cfg.trace_slow_column_str
    drop_error_rows = bool(svc_cfg.trace_drop_error_rows_bool)
    fast_error_col = "fast_error"
    slow_error_col = "slow_error"

    values = _Read_Trace(
        csv_path="../"+csv_path,
        fast_col=fast_col,
        slow_col=slow_col,
        fast_error_col=fast_error_col,
        slow_error_col=slow_error_col,
        drop_error_rows=drop_error_rows,
    )

    sample_count = 1335
    fast_vals = _Take_First(values["fast"], sample_count)
    slow_vals = _Take_First(values["slow"], sample_count)

    rows = [
        _Summary(fast_vals, "fast"),
        _Summary(slow_vals, "slow"),
    ]

    out_dir = os.path.join("Results", "Agent_Stats")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "agent_results_stats.csv")

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["mode", "n", "mean", "std", "p50", "p90", "p95", "p99"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


if __name__ == "__main__":
    main()
