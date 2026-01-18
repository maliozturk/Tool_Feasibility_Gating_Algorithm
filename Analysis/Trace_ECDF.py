# =============================================================================
#  TOOL FEASIBILITY GATING ALGORITHM (TFG)
#  Product Signature: TFG
# ------------------------------------------------------------------------------
#  File: Analysis/Trace_ECDF.py
#  Purpose: Plot ECDFs for trace-based service times.
#  Author: Muhammet Ali Ozturk
#  Generated: 2026-01-18
#  Environment: Python 3.9.13
# =============================================================================

import argparse
import csv
import os
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from Configurations import Simulation_Config


def _Apply_Plot_Style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "lines.linewidth": 2.0,
            "lines.markersize": 4,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _ecdf(values: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.asarray([]), np.asarray([])
    arr = np.sort(arr)
    ys = np.arange(1, arr.size + 1, dtype=float) / float(arr.size)
    return arr, ys


def _read_trace_rows(
    csv_path: str,
    fast_col: str,
    slow_col: str,
    prompt_type_col: str,
    fast_error_col: Optional[str],
    slow_error_col: Optional[str],
    drop_error_rows: bool,
) -> List[Dict[str, str]]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Trace CSV not found: {csv_path}")

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            if drop_error_rows:
                if fast_error_col and row.get(fast_error_col):
                    continue
                if slow_error_col and row.get(slow_error_col):
                    continue
            rows.append(row)

    if not rows:
        raise ValueError("No rows loaded from trace CSV.")

                                                         
    missing_cols = []
    for col in [fast_col, slow_col, prompt_type_col]:
        if col and col not in rows[0]:
            missing_cols.append(col)
    if missing_cols:
        raise ValueError(f"Missing expected columns in trace CSV: {missing_cols}")

    return rows


def _extract_values(rows: Sequence[Dict[str, str]], col_name: str) -> List[float]:
    vals: List[float] = []
    for row in rows:
        raw = row.get(col_name, "")
        try:
            v = float(raw)
        except (TypeError, ValueError):
            continue
        if v <= 0.0:
            continue
        vals.append(v)
    return vals


def _plot_ecdf_pair(
    out_path: str,
    fast_vals: Sequence[float],
    slow_vals: Sequence[float],
    title: str,
) -> None:
    fx, fy = _ecdf(fast_vals)
    sx, sy = _ecdf(slow_vals)

    plt.figure(figsize=(6.5, 4))
    if fx.size:
        plt.step(fx, fy, where="post", label="FAST", color="tab:blue")
    if sx.size:
        plt.step(sx, sy, where="post", label="SLOW", color="tab:orange")
    plt.xlabel("generation time (s)")
    plt.ylabel("ECDF")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_ecdf_by_prompt_type(
    out_path: str,
    rows: Sequence[Dict[str, str]],
    fast_col: str,
    slow_col: str,
    prompt_type_col: str,
    prompt_types: Sequence[str],
    title: str,
) -> None:
    _Apply_Plot_Style()
    fig, axes = plt.subplots(1, len(prompt_types), figsize=(6.5 * len(prompt_types), 4), sharey=True)
    if len(prompt_types) == 1:
        axes = [axes]

    for ax, pt in zip(axes, prompt_types):
        sub_rows = [r for r in rows if r.get(prompt_type_col) == pt]
        fast_vals = _extract_values(sub_rows, fast_col)
        slow_vals = _extract_values(sub_rows, slow_col)
        fx, fy = _ecdf(fast_vals)
        sx, sy = _ecdf(slow_vals)
        if fx.size:
            ax.step(fx, fy, where="post", label="FAST", color="tab:blue")
        if sx.size:
            ax.step(sx, sy, where="post", label="SLOW", color="tab:orange")
        ax.set_title(f"{title}: {pt}")
        ax.set_xlabel("generation time (s)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("ECDF")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def main() -> None:
    sim_cfg = Simulation_Config()
    svc_cfg = sim_cfg.service_config

    parser = argparse.ArgumentParser(description="Plot ECDFs for empirical service times.")
    parser.add_argument("--csv-path", default=svc_cfg.trace_csv_path_str)
    parser.add_argument("--fast-col", default=svc_cfg.trace_fast_column_str)
    parser.add_argument("--slow-col", default=svc_cfg.trace_slow_column_str)
    parser.add_argument("--prompt-type-col", default="prompt_type")
    parser.add_argument("--fast-error-col", default="fast_error")
    parser.add_argument("--slow-error-col", default="slow_error")
    parser.add_argument("--drop-error-rows", action="store_true", default=svc_cfg.trace_drop_error_rows_bool)
    parser.add_argument("--outdir", default="Results/Trace_ECDF")
    parser.add_argument("--split-by-prompt-type", action="store_true")
    parser.add_argument("--prompt-types", default="simple,complex")

    args = parser.parse_args()
    _Apply_Plot_Style()

    rows = _read_trace_rows(
        csv_path="../" + str(args.csv_path),
        fast_col=str(args.fast_col),
        slow_col=str(args.slow_col),
        prompt_type_col=str(args.prompt_type_col),
        fast_error_col=str(args.fast_error_col) if args.fast_error_col else None,
        slow_error_col=str(args.slow_error_col) if args.slow_error_col else None,
        drop_error_rows=bool(args.drop_error_rows),
    )

    os.makedirs(args.outdir, exist_ok=True)

    fast_vals = _extract_values(rows, str(args.fast_col))
    slow_vals = _extract_values(rows, str(args.slow_col))
    _plot_ecdf_pair(
        out_path=os.path.join(args.outdir, "ecdf_fast_vs_slow.png"),
        fast_vals=fast_vals,
        slow_vals=slow_vals,
        title="Empirical service-time ECDF (FAST vs SLOW)",
    )

    if args.split_by_prompt_type:
        prompt_types = [p.strip() for p in str(args.prompt_types).split(",") if p.strip()]
        if prompt_types:
            _plot_ecdf_by_prompt_type(
                out_path=os.path.join(args.outdir, "ecdf_fast_vs_slow_by_prompt_type.png"),
                rows=rows,
                fast_col=str(args.fast_col),
                slow_col=str(args.slow_col),
                prompt_type_col=str(args.prompt_type_col),
                prompt_types=prompt_types,
                title="Empirical service-time ECDF",
            )


if __name__ == "__main__":
    main()
