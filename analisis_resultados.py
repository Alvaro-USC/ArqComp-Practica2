"""
analisis_resultados.py
=====================

Analiza los resultados de la Practica 2 de Arquitectura de Computadores.

Numeracion correcta usada en este repositorio:
  - v1: secuencial base
  - v2: secuencial optimizada para cache
  - v3: OpenMP
  - v4: SIMD AVX256 + FMA

Ficheros de entrada esperados:
  - resultados/v1_O0.txt
  - resultados/v1_O3.txt
  - resultados/v2_O0.txt
  - resultados/v2_O3.txt
  - resultados/v3_O3_static.txt
  - resultados/v3_O3_dynamic.txt
  - resultados/v3_O3_guided.txt
  - resultados/v3_O3_critical.txt
  - resultados/v4_O0.txt
  - resultados/v4_O3.txt

Graficas generadas:
  - g1: speedup v2 vs v1 con -O0
  - g2: speedup v2 / v3(mejor) / v4 vs v1-O3
  - g3: speedup v3(mejor) / v4 vs v2-O3
  - g4: speedup OpenMP por hilos (schedule static)
  - g5: comparacion de schedulings OpenMP
  - g6: comparacion reduction vs critical
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SIZES = [1250, 2000, 3200]
PLOT_FILES = {
    "g1": "g1_speedup_v2_vs_v1_O0.png",
    "g2": "g2_speedup_vs_v1O3.png",
    "g3": "g3_speedup_v3_v4_vs_v2O3.png",
    "g4": "g4_speedup_hilos.png",
    "g5": "g5_schedulings.png",
    "g6": "g6_atomic_vs_critical.png",
}
SEQ_FILES = {
    "v1_O0": "v1_O0.txt",
    "v1_O3": "v1_O3.txt",
    "v2_O0": "v2_O0.txt",
    "v2_O3": "v2_O3.txt",
    "v4_O0": "v4_O0.txt",
    "v4_O3": "v4_O3.txt",
}
OMP_FILES = {
    "static": "v3_O3_static.txt",
    "dynamic": "v3_O3_dynamic.txt",
    "guided": "v3_O3_guided.txt",
    "critical": "v3_O3_critical.txt",
}


@dataclass
class AnalysisData:
    seq_raw: Dict[str, pd.DataFrame]
    seq_med: Dict[str, pd.DataFrame]
    omp_raw: Dict[str, pd.DataFrame]
    omp_med: Dict[str, pd.DataFrame]
    best_openmp: pd.DataFrame


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "font.size": 11,
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.alpha": 0.45,
            "lines.linewidth": 2,
            "lines.markersize": 7,
        }
    )


def load_sequential(path: str) -> pd.DataFrame:
    rows: List[dict] = []
    if not os.path.isfile(path):
        print(f"[AVISO] No se encontro {path}")
        return pd.DataFrame(columns=["version", "n", "iter", "norm2", "ciclos"])

    with open(path, encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) != 5:
                continue
            try:
                rows.append(
                    {
                        "version": parts[0],
                        "n": int(parts[1]),
                        "iter": int(parts[2]),
                        "norm2": float(parts[3]),
                        "ciclos": float(parts[4]),
                    }
                )
            except ValueError:
                continue

    return pd.DataFrame(rows)


def load_openmp(path: str) -> pd.DataFrame:
    rows: List[dict] = []
    if not os.path.isfile(path):
        print(f"[AVISO] No se encontro {path}")
        return pd.DataFrame(columns=["version", "n", "threads", "iter", "norm2", "ciclos"])

    with open(path, encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) != 6:
                continue
            try:
                rows.append(
                    {
                        "version": parts[0],
                        "n": int(parts[1]),
                        "threads": int(parts[2]),
                        "iter": int(parts[3]),
                        "norm2": float(parts[4]),
                        "ciclos": float(parts[5]),
                    }
                )
            except ValueError:
                continue

    return pd.DataFrame(rows)


def median_table(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=group_cols + ["iter", "norm2", "ciclos"])
    return df.groupby(group_cols)[["iter", "norm2", "ciclos"]].median().reset_index()


def get_metric(med_df: pd.DataFrame, n: int, metric: str, threads: Optional[int] = None) -> float:
    if med_df.empty:
        return np.nan

    if threads is None:
        row = med_df[med_df["n"] == n]
    else:
        row = med_df[(med_df["n"] == n) & (med_df["threads"] == threads)]

    if row.empty:
        return np.nan
    return float(row.iloc[0][metric])


def safe_div(num: float, den: float) -> float:
    if np.isnan(num) or np.isnan(den) or den == 0.0:
        return np.nan
    return num / den


def build_best_openmp(omp_med: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: List[dict] = []

    for n in SIZES:
        best_row: Optional[dict] = None
        for variant, med_df in omp_med.items():
            if med_df.empty:
                continue
            for _, row in med_df[med_df["n"] == n].iterrows():
                candidate = {
                    "n": n,
                    "variant": variant,
                    "threads": int(row["threads"]),
                    "iter": float(row["iter"]),
                    "norm2": float(row["norm2"]),
                    "ciclos": float(row["ciclos"]),
                }
                if best_row is None or candidate["ciclos"] < best_row["ciclos"]:
                    best_row = candidate
        if best_row is not None:
            rows.append(best_row)

    return pd.DataFrame(rows)


def load_all_data(results_dir: str) -> AnalysisData:
    seq_raw = {key: load_sequential(os.path.join(results_dir, filename)) for key, filename in SEQ_FILES.items()}
    seq_med = {key: median_table(df, ["n"]) for key, df in seq_raw.items()}

    omp_raw = {key: load_openmp(os.path.join(results_dir, filename)) for key, filename in OMP_FILES.items()}
    omp_med = {key: median_table(df, ["n", "threads"]) for key, df in omp_raw.items()}

    best_openmp = build_best_openmp(omp_med)
    return AnalysisData(seq_raw=seq_raw, seq_med=seq_med, omp_raw=omp_raw, omp_med=omp_med, best_openmp=best_openmp)


def print_result_consistency_warnings(data: AnalysisData) -> None:
    static_med = data.omp_med["static"]
    v2_med = data.seq_med["v2_O3"]
    v4_med = data.seq_med["v4_O3"]

    if static_med.empty or v2_med.empty or v4_med.empty:
        return

    print("\nChequeo rapido de consistencia numerica:")
    issues = 0
    for n in SIZES:
        iter_v2 = get_metric(v2_med, n, "iter")
        iter_v4 = get_metric(v4_med, n, "iter")
        iter_v3 = get_metric(static_med, n, "iter", threads=1)
        norm_v2 = get_metric(v2_med, n, "norm2")
        norm_v4 = get_metric(v4_med, n, "norm2")
        norm_v3 = get_metric(static_med, n, "norm2", threads=1)

        if np.isnan(iter_v3) or np.isnan(norm_v3):
            continue

        if not (abs(iter_v2 - iter_v3) < 0.5 and abs(iter_v4 - iter_v3) < 0.5):
            issues += 1
            print(f"  [AVISO] n={n}: v3(1 hilo) no coincide en iteraciones con v2/v4.")
        if not (np.isclose(norm_v2, norm_v3, rtol=1e-9, atol=1e-12) and np.isclose(norm_v4, norm_v3, rtol=1e-9, atol=1e-12)):
            issues += 1
            print(f"  [AVISO] n={n}: v3(1 hilo) no coincide en norm2 con v2/v4.")

    if issues == 0:
        print("  OK: v2, v3(1 hilo) y v4 son coherentes en los resultados disponibles.")


def ensure_plot_dir(plots_dir: str) -> None:
    os.makedirs(plots_dir, exist_ok=True)


def save_plot(fig: plt.Figure, plots_dir: str, key: str) -> None:
    ensure_plot_dir(plots_dir)
    path = os.path.join(plots_dir, PLOT_FILES[key])
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  -> {path}")


def generate_g1(data: AnalysisData, plots_dir: str) -> None:
    print("Generando G1: speedup v2 vs v1 con -O0...")
    x = np.arange(len(SIZES))
    values = [safe_div(get_metric(data.seq_med["v1_O0"], n, "ciclos"), get_metric(data.seq_med["v2_O0"], n, "ciclos")) for n in SIZES]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(x, values, width=0.55, color="#1f77b4", alpha=0.9)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in SIZES])
    ax.set_xlabel("Tamano n")
    ax.set_ylabel("Speedup")
    ax.set_title("G1 - Speedup v2 vs v1 (ambos -O0)")
    for bar, value in zip(bars, values):
        if not np.isnan(value):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.2f}x", ha="center", va="bottom")

    save_plot(fig, plots_dir, "g1")


def generate_g2(data: AnalysisData, plots_dir: str) -> None:
    print("Generando G2: v2 / v3(mejor) / v4 vs v1-O3...")
    labels = ["v2 -O3", "v3 OpenMP (mejor)", "v4 SIMD -O3"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    x = np.arange(len(SIZES))
    width = 0.24

    best_v3 = data.best_openmp
    series = []
    for n in SIZES:
        base = get_metric(data.seq_med["v1_O3"], n, "ciclos")
        v2 = safe_div(base, get_metric(data.seq_med["v2_O3"], n, "ciclos"))
        v4 = safe_div(base, get_metric(data.seq_med["v4_O3"], n, "ciclos"))
        if best_v3.empty:
            v3 = np.nan
        else:
            row = best_v3[best_v3["n"] == n]
            v3 = np.nan if row.empty else safe_div(base, float(row.iloc[0]["ciclos"]))
        series.append([v2, v3, v4])

    values = np.array(series, dtype=float)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for idx, label in enumerate(labels):
        col = values[:, idx]
        valid = ~np.isnan(col)
        ax.bar(x[valid] + (idx - 1) * width, col[valid], width=width, label=label, color=colors[idx], alpha=0.9)

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in SIZES])
    ax.set_xlabel("Tamano n")
    ax.set_ylabel("Speedup")
    ax.set_title("G2 - Speedup v2 / v3(mejor) / v4 vs v1-O3")
    ax.legend()

    save_plot(fig, plots_dir, "g2")


def generate_g3(data: AnalysisData, plots_dir: str) -> None:
    print("Generando G3: v3(mejor) / v4 vs v2-O3...")
    x = np.arange(len(SIZES))
    width = 0.28
    sp_v3 = []
    sp_v4 = []

    for n in SIZES:
        base = get_metric(data.seq_med["v2_O3"], n, "ciclos")
        best_row = data.best_openmp[data.best_openmp["n"] == n]
        best_v3_cycles = np.nan if best_row.empty else float(best_row.iloc[0]["ciclos"])
        sp_v3.append(safe_div(base, best_v3_cycles))
        sp_v4.append(safe_div(base, get_metric(data.seq_med["v4_O3"], n, "ciclos")))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars_v3 = ax.bar(x - width / 2, sp_v3, width=width, label="v3 OpenMP (mejor)", color="#ff7f0e", alpha=0.9)
    bars_v4 = ax.bar(x + width / 2, sp_v4, width=width, label="v4 SIMD -O3", color="#2ca02c", alpha=0.9)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in SIZES])
    ax.set_xlabel("Tamano n")
    ax.set_ylabel("Speedup")
    ax.set_title("G3 - Speedup v3(mejor) / v4 vs v2-O3")
    ax.legend()

    for bar, value in zip(list(bars_v3) + list(bars_v4), sp_v3 + sp_v4):
        if not np.isnan(value):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.2f}x", ha="center", va="bottom")

    save_plot(fig, plots_dir, "g3")


def generate_g4(data: AnalysisData, plots_dir: str) -> None:
    print("Generando G4: speedup OpenMP por hilos (schedule static)...")
    med_static = data.omp_med["static"]
    if med_static.empty:
        print("  [AVISO] No hay datos de v3_O3_static.txt; se omite G4.")
        return

    threads = sorted(med_static["threads"].unique())
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for color, n in zip(colors, SIZES):
        base_1t = get_metric(med_static, n, "ciclos", threads=1)
        speedups = [safe_div(base_1t, get_metric(med_static, n, "ciclos", threads=t)) for t in threads]
        ax.plot(threads, speedups, marker="o", color=color, label=f"n={n}")

    ax.plot(threads, threads, "k--", linewidth=1, label="Ideal")
    ax.set_xticks(threads)
    ax.set_xlabel("Numero de hilos")
    ax.set_ylabel("Speedup")
    ax.set_title("G4 - Speedup v3 OpenMP (schedule static)")
    ax.legend()

    save_plot(fig, plots_dir, "g4")


def generate_g5(data: AnalysisData, plots_dir: str) -> None:
    print("Generando G5: comparacion de schedulings...")
    sched_names = [("static", "#1f77b4"), ("dynamic", "#ff7f0e"), ("guided", "#2ca02c")]

    if all(data.omp_med[name].empty for name, _ in sched_names):
        print("  [AVISO] No hay datos suficientes de OpenMP; se omite G5.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)
    fig.suptitle("G5 - Comparacion de schedulings OpenMP (speedup vs 1 hilo)")

    for ax, n in zip(axes, SIZES):
        ax.set_title(f"n = {n}")
        for sched_name, color in sched_names:
            med_df = data.omp_med[sched_name]
            if med_df.empty:
                continue
            threads = sorted(med_df["threads"].unique())
            base_1t = get_metric(med_df, n, "ciclos", threads=1)
            speedups = [safe_div(base_1t, get_metric(med_df, n, "ciclos", threads=t)) for t in threads]
            label = "dynamic(16)" if sched_name == "dynamic" else sched_name
            ax.plot(threads, speedups, marker="o", color=color, label=label)
            ax.set_xticks(threads)

        static_threads = sorted(data.omp_med["static"]["threads"].unique()) if not data.omp_med["static"].empty else []
        if static_threads:
            ax.plot(static_threads, static_threads, "k--", linewidth=1, label="Ideal")

        ax.set_xlabel("Hilos")
        ax.set_ylabel("Speedup")
        ax.legend(fontsize=8)

    save_plot(fig, plots_dir, "g5")


def generate_g6(data: AnalysisData, plots_dir: str) -> None:
    print("Generando G6: reduction vs critical...")
    med_static = data.omp_med["static"]
    med_critical = data.omp_med["critical"]

    if med_static.empty or med_critical.empty:
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "No hay datos suficientes para comparar\nreduction vs critical.\n\n"
            "Vuelve a ejecutar EXPv3.sh para regenerar\nresultados/v3_O3_critical.txt.",
            ha="center",
            va="center",
            fontsize=12,
        )
        save_plot(fig, plots_dir, "g6")
        print("  [AVISO] Faltan datos de static o critical; G6 se guarda como figura informativa.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)
    fig.suptitle("G6 - Reduccion reduction vs critical (speedup vs 1 hilo)")

    for ax, n in zip(axes, SIZES):
        threads_static = sorted(med_static["threads"].unique())
        threads_critical = sorted(med_critical["threads"].unique())

        base_static = get_metric(med_static, n, "ciclos", threads=1)
        base_critical = get_metric(med_critical, n, "ciclos", threads=1)

        speedup_static = [safe_div(base_static, get_metric(med_static, n, "ciclos", threads=t)) for t in threads_static]
        speedup_critical = [safe_div(base_critical, get_metric(med_critical, n, "ciclos", threads=t)) for t in threads_critical]

        ax.plot(threads_static, speedup_static, marker="o", color="#1f77b4", label="reduction")
        ax.plot(threads_critical, speedup_critical, marker="s", color="#d62728", label="critical")
        ax.plot(threads_static, threads_static, "k--", linewidth=1, label="Ideal")
        ax.set_title(f"n = {n}")
        ax.set_xlabel("Hilos")
        ax.set_ylabel("Speedup")
        ax.set_xticks(threads_static)
        ax.legend(fontsize=8)

    save_plot(fig, plots_dir, "g6")


def print_summary(data: AnalysisData) -> None:
    print("\nResumen de medianas de ciclos (x10^9):")
    print(f"{'Version':<26} {'n=1250':>10} {'n=2000':>10} {'n=3200':>10}")
    print("-" * 60)

    def fmt(value: float) -> str:
        return "---" if np.isnan(value) else f"{value / 1e9:>8.2f}"

    rows = [
        ("v1 -O0", data.seq_med["v1_O0"], None),
        ("v1 -O3", data.seq_med["v1_O3"], None),
        ("v2 -O0", data.seq_med["v2_O0"], None),
        ("v2 -O3", data.seq_med["v2_O3"], None),
        ("v4 -O0 (SIMD)", data.seq_med["v4_O0"], None),
        ("v4 -O3 (SIMD)", data.seq_med["v4_O3"], None),
    ]

    for label, med_df, threads in rows:
        vals = [get_metric(med_df, n, "ciclos", threads=threads) for n in SIZES]
        print(f"{label:<26} {' '.join(fmt(v) for v in vals)}")

    med_static = data.omp_med["static"]
    if not med_static.empty:
        max_threads = int(med_static["threads"].max())
        for threads in (1, max_threads):
            vals = [get_metric(med_static, n, "ciclos", threads=threads) for n in SIZES]
            print(f"{f'v3 -O3 (OpenMP) {threads}T':<26} {' '.join(fmt(v) for v in vals)}")

    if not data.best_openmp.empty:
        print("\nMejor configuracion OpenMP por tamano:")
        for _, row in data.best_openmp.iterrows():
            print(
                f"  n={int(row['n'])}: {row['variant']} con {int(row['threads'])} hilos "
                f"-> {row['ciclos'] / 1e9:.2f}e9 ciclos"
            )


def format_tex_value(value: float) -> str:
    return "---" if np.isnan(value) else f"{value / 1e9:.1f}"


def format_speedup_tex(value: float) -> str:
    return "---" if np.isnan(value) else f"{value:.2f}$\\times$"


def format_percent_tex(value: float) -> str:
    return "---" if np.isnan(value) else f"{value:.1f}\\%"


def variant_label(name: str) -> str:
    return "dynamic(16)" if name == "dynamic" else name


def best_openmp_row_for_n(data: AnalysisData, n: int) -> Optional[pd.Series]:
    row = data.best_openmp[data.best_openmp["n"] == n]
    return None if row.empty else row.iloc[0]


def series_for_sizes(values: List[float]) -> str:
    parts = [f"$n={n}$: \\textbf{{{format_speedup_tex(value)}}}" for n, value in zip(SIZES, values) if not np.isnan(value)]
    if not parts:
        return "---"
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " y " + parts[-1]


def speedup_range_tex(values: List[float]) -> str:
    valid = [value for value in values if not np.isnan(value)]
    if not valid:
        return "---"
    return f"entre \\textbf{{{format_speedup_tex(min(valid))}}} y \\textbf{{{format_speedup_tex(max(valid))}}}"


def winner_counts(series_map: Dict[str, List[float]]) -> Dict[str, int]:
    counts = {label: 0 for label in series_map}
    for idx in range(len(SIZES)):
        best_label = None
        best_value = -np.inf
        for label, values in series_map.items():
            value = values[idx]
            if np.isnan(value):
                continue
            if value > best_value:
                best_value = value
                best_label = label
        if best_label is not None:
            counts[best_label] += 1
    return counts


def max_threads_available(data: AnalysisData) -> Optional[int]:
    med_static = data.omp_med["static"]
    if med_static.empty:
        return None
    return int(med_static["threads"].max())


def generate_table_rows(data: AnalysisData) -> str:
    med_static = data.omp_med["static"]
    max_threads = max_threads_available(data)
    rows = [
        ("v1 \\texttt{-O0}", [get_metric(data.seq_med["v1_O0"], n, "ciclos") for n in SIZES]),
        ("v1 \\texttt{-O3}", [get_metric(data.seq_med["v1_O3"], n, "ciclos") for n in SIZES]),
        ("v2 \\texttt{-O0}", [get_metric(data.seq_med["v2_O0"], n, "ciclos") for n in SIZES]),
        ("v2 \\texttt{-O3}", [get_metric(data.seq_med["v2_O3"], n, "ciclos") for n in SIZES]),
        ("v4 \\texttt{-O0}", [get_metric(data.seq_med["v4_O0"], n, "ciclos") for n in SIZES]),
        ("v4 \\texttt{-O3}", [get_metric(data.seq_med["v4_O3"], n, "ciclos") for n in SIZES]),
    ]

    if not med_static.empty:
        rows.append(("v3 \\texttt{-O3} (1T)", [get_metric(med_static, n, "ciclos", threads=1) for n in SIZES]))
        if max_threads is not None:
            rows.append((f"v3 \\texttt{{-O3}} ({max_threads}T)", [get_metric(med_static, n, "ciclos", threads=max_threads) for n in SIZES]))

    return "\n".join(f"{label} & {' & '.join(format_tex_value(value) for value in values)} \\\\" for label, values in rows)


def generate_abstract_results(data: AnalysisData) -> str:
    simd_speedup_3200 = safe_div(get_metric(data.seq_med["v2_O3"], 3200, "ciclos"), get_metric(data.seq_med["v4_O3"], 3200, "ciclos"))
    med_static = data.omp_med["static"]
    omp_row_3200 = best_openmp_row_for_n(data, 3200)

    if omp_row_3200 is not None and not med_static.empty:
        omp_speedup = safe_div(get_metric(med_static, 3200, "ciclos", threads=1), float(omp_row_3200["ciclos"]))
        first_sentence = (
            f"Para $n{{=}}3200$, la versión SIMD (v4) alcanza un \\textit{{speedup}} de "
            f"\\textbf{{{format_speedup_tex(simd_speedup_3200)}}} sobre v2\\texttt{{-O3}}, "
            f"mientras que la mejor configuración OpenMP (v3, \\texttt{{{variant_label(str(omp_row_3200['variant']))}}}, "
            f"{int(omp_row_3200['threads'])} hilos) consigue \\textbf{{{format_speedup_tex(omp_speedup)}}} "
            f"respecto a v3 con 1 hilo. "
        )
    else:
        first_sentence = (
            f"Para $n{{=}}3200$, la versión SIMD (v4) alcanza un \\textit{{speedup}} de "
            f"\\textbf{{{format_speedup_tex(simd_speedup_3200)}}} sobre v2\\texttt{{-O3}}. "
        )

    max_threads = max_threads_available(data)
    if max_threads is not None and not med_static.empty:
        sp_small = safe_div(get_metric(med_static, 1250, "ciclos", threads=1), get_metric(med_static, 1250, "ciclos", threads=max_threads))
        sp_large = safe_div(get_metric(med_static, 3200, "ciclos", threads=1), get_metric(med_static, 3200, "ciclos", threads=max_threads))
        second_sentence = (
            f"Los experimentos muestran que el escalado OpenMP depende del tamaño: "
            f"a {max_threads} hilos, v3 pasa de \\textbf{{{format_speedup_tex(sp_small)}}} "
            f"en $n{{=}}1250$ a \\textbf{{{format_speedup_tex(sp_large)}}} en $n{{=}}3200$.\\\\[2pt]"
        )
    else:
        second_sentence = (
            "Los experimentos muestran que el escalado OpenMP depende del tamaño del problema "
            "y de la política de reparto utilizada.\\\\[2pt]"
        )

    return first_sentence + second_sentence


def generate_g1_text(data: AnalysisData) -> str:
    speedups = [safe_div(get_metric(data.seq_med["v1_O0"], n, "ciclos"), get_metric(data.seq_med["v2_O0"], n, "ciclos")) for n in SIZES]
    if all(np.isnan(value) for value in speedups):
        return "La Figura~\\ref{fig:g1} muestra el \\textit{speedup} de v2 respecto a v1 con ambas versiones en \\texttt{-O0}."
    if all(abs(value - 1.0) < 0.15 for value in speedups if not np.isnan(value)):
        conclusion = "Las diferencias son moderadas, así que sin \\texttt{-O3} las optimizaciones manuales apenas cambian el rendimiento de forma sistemática."
    elif all(value > 1.0 for value in speedups if not np.isnan(value)):
        conclusion = "Incluso sin \\texttt{-O3}, v2 mejora de forma consistente a la versión base."
    else:
        conclusion = "El efecto depende del tamaño: algunas configuraciones mejoran frente a v1 y otras quedan por debajo."
    return (
        "La Figura~\\ref{fig:g1} muestra el \\textit{speedup} de v2 respecto a v1 con ambas "
        f"versiones en \\texttt{{-O0}}. Los valores medidos son {series_for_sizes(speedups)}. "
        f"{conclusion}"
    )


def generate_g2_text(data: AnalysisData) -> str:
    sp_v2 = [safe_div(get_metric(data.seq_med["v1_O3"], n, "ciclos"), get_metric(data.seq_med["v2_O3"], n, "ciclos")) for n in SIZES]
    sp_v4 = [safe_div(get_metric(data.seq_med["v1_O3"], n, "ciclos"), get_metric(data.seq_med["v4_O3"], n, "ciclos")) for n in SIZES]
    sp_v3 = []
    best_cfg_parts = []
    for n in SIZES:
        row = best_openmp_row_for_n(data, n)
        if row is None:
            sp_v3.append(np.nan)
            continue
        sp_v3.append(safe_div(get_metric(data.seq_med["v1_O3"], n, "ciclos"), float(row["ciclos"])))
        best_cfg_parts.append(f"$n={n}$: \\texttt{{{variant_label(str(row['variant']))}}} con {int(row['threads'])} hilos")

    counts = winner_counts({"v2": sp_v2, "v3": sp_v3, "v4": sp_v4})
    dominant = max(counts, key=counts.get)
    dominant_label = {"v2": "v2", "v3": "v3", "v4": "v4"}[dominant]
    config_sentence = "" if not best_cfg_parts else " Las mejores configuraciones de v3 son " + ", ".join(best_cfg_parts) + "."
    return (
        "La Figura~\\ref{fig:g2} compara la versión secuencial optimizada (v2), la mejor "
        "configuración OpenMP (v3) y la versión SIMD (v4) respecto a v1\\texttt{-O3}. "
        f"Los speedups de v2 son {series_for_sizes(sp_v2)}; los de v3, {series_for_sizes(sp_v3)}; "
        f"y los de v4, {series_for_sizes(sp_v4)}. En estos resultados, \\textbf{{{dominant_label}}} "
        f"es la serie más rápida en {counts[dominant]}/3 tamaños.{config_sentence}"
    )


def generate_g3_text(data: AnalysisData) -> str:
    sp_v4 = [safe_div(get_metric(data.seq_med["v2_O3"], n, "ciclos"), get_metric(data.seq_med["v4_O3"], n, "ciclos")) for n in SIZES]
    sp_v3 = []
    for n in SIZES:
        row = best_openmp_row_for_n(data, n)
        sp_v3.append(np.nan if row is None else safe_div(get_metric(data.seq_med["v2_O3"], n, "ciclos"), float(row["ciclos"])))

    winner = "v3" if np.nanmax(sp_v3) >= np.nanmax(sp_v4) else "v4"
    return (
        "La Figura~\\ref{fig:g3} muestra el \\textit{speedup} de la mejor configuración "
        "OpenMP (v3) y de la versión SIMD (v4) respecto a v2\\texttt{-O3}. "
        f"OpenMP obtiene speedups {speedup_range_tex(sp_v3)}, mientras que SIMD logra "
        f"{speedup_range_tex(sp_v4)}. En conjunto, \\textbf{{{winner}}} aporta la mayor "
        "ganancia respecto a la mejor versión secuencial."
    )


def generate_g4_text(data: AnalysisData) -> str:
    med_static = data.omp_med["static"]
    max_threads = max_threads_available(data)
    if med_static.empty or max_threads is None:
        return (
            "La Figura~\\ref{fig:g4} es la gráfica \\textbf{obligatoria} del enunciado y resume "
            "el speedup de v3\\texttt{-O3} con \\texttt{schedule(static)} frente a 1 hilo."
        )

    speedups = [safe_div(get_metric(med_static, n, "ciclos", threads=1), get_metric(med_static, n, "ciclos", threads=max_threads)) for n in SIZES]
    best_n = SIZES[int(np.nanargmax(speedups))]
    worst_n = SIZES[int(np.nanargmin(speedups))]
    return (
        "La Figura~\\ref{fig:g4} es la gráfica \\textbf{obligatoria} del enunciado y resume "
        "el speedup de v3\\texttt{-O3} con \\texttt{schedule(static)} frente a 1 hilo. "
        f"A {max_threads} hilos, los speedups alcanzan {series_for_sizes(speedups)}. "
        f"El mejor escalado se observa en $n={best_n}$, mientras que $n={worst_n}$ es el caso "
        "más sensible al overhead de sincronización."
    )


def generate_g5_text(data: AnalysisData) -> str:
    max_threads = max_threads_available(data)
    if max_threads is None:
        return "La Figura~\\ref{fig:g5} compara los tres modos de reparto de OpenMP sobre v3."

    parts = []
    for n in SIZES:
        best_variant = None
        best_speedup = -np.inf
        for variant in ("static", "dynamic", "guided"):
            med_df = data.omp_med[variant]
            if med_df.empty:
                continue
            speedup = safe_div(get_metric(med_df, n, "ciclos", threads=1), get_metric(med_df, n, "ciclos", threads=max_threads))
            if not np.isnan(speedup) and speedup > best_speedup:
                best_speedup = speedup
                best_variant = variant_label(variant)
        if best_variant is not None:
            parts.append(f"$n={n}$: \\texttt{{{best_variant}}} con \\textbf{{{format_speedup_tex(best_speedup)}}}")

    if not parts:
        return "La Figura~\\ref{fig:g5} compara los tres modos de reparto de OpenMP sobre v3."

    return (
        "La Figura~\\ref{fig:g5} compara los tres modos de reparto de OpenMP sobre v3. "
        f"A {max_threads} hilos, la mejor política para cada tamaño es " + ", ".join(parts) + "."
    )


def generate_g6_text(data: AnalysisData) -> str:
    med_static = data.omp_med["static"]
    med_critical = data.omp_med["critical"]
    max_threads = max_threads_available(data)

    if med_static.empty or med_critical.empty or max_threads is None:
        return (
            "La Figura~\\ref{fig:g6} compara la variante base con "
            "\\texttt{reduction(+:norm2)} frente a la variante con \\texttt{critical}. "
            "Si falta alguno de los ficheros de resultados, esta comparación no podrá "
            "actualizarse automáticamente."
        )

    summary_parts = []
    diff_parts = []
    for n in SIZES:
        reduction_sp = safe_div(get_metric(med_static, n, "ciclos", threads=1), get_metric(med_static, n, "ciclos", threads=max_threads))
        critical_sp = safe_div(get_metric(med_critical, n, "ciclos", threads=1), get_metric(med_critical, n, "ciclos", threads=max_threads))
        if np.isnan(reduction_sp) or np.isnan(critical_sp):
            continue
        faster = "reduction" if reduction_sp >= critical_sp else "critical"
        rel_gap = abs(reduction_sp - critical_sp) / max(reduction_sp, critical_sp) * 100.0
        summary_parts.append(
            f"$n={n}$: \\texttt{{reduction}}={format_speedup_tex(reduction_sp)}, "
            f"\\texttt{{critical}}={format_speedup_tex(critical_sp)}"
        )
        diff_parts.append((n, faster, rel_gap))

    if not summary_parts:
        return (
            "La Figura~\\ref{fig:g6} compara la variante base con "
            "\\texttt{reduction(+:norm2)} frente a la variante con \\texttt{critical}."
        )

    max_diff = max(diff_parts, key=lambda item: item[2])
    return (
        "La Figura~\\ref{fig:g6} compara la variante base con "
        "\\texttt{reduction(+:norm2)} frente a la variante con \\texttt{critical}. "
        f"A {max_threads} hilos, se obtiene " + "; ".join(summary_parts) + ". "
        f"La mayor diferencia relativa aparece en $n={max_diff[0]}$, donde "
        f"\\texttt{{{max_diff[1]}}} aventaja a la otra opción en aproximadamente "
        f"\\textbf{{{format_percent_tex(max_diff[2])}}}."
    )


def generate_conclusions(data: AnalysisData) -> str:
    sp_v2_o0 = [safe_div(get_metric(data.seq_med["v1_O0"], n, "ciclos"), get_metric(data.seq_med["v2_O0"], n, "ciclos")) for n in SIZES]
    sp_v4_vs_v2 = [safe_div(get_metric(data.seq_med["v2_O3"], n, "ciclos"), get_metric(data.seq_med["v4_O3"], n, "ciclos")) for n in SIZES]
    med_static = data.omp_med["static"]
    max_threads = max_threads_available(data)

    if not med_static.empty and max_threads is not None:
        sp_v3_static = [safe_div(get_metric(med_static, n, "ciclos", threads=1), get_metric(med_static, n, "ciclos", threads=max_threads)) for n in SIZES]
        static_text = f"Con \\texttt{{schedule(static)}}, v3 alcanza {series_for_sizes(sp_v3_static)} a {max_threads} hilos."
    else:
        static_text = "La evaluación de v3 depende de disponer de los ficheros OpenMP generados por EXPv3.sh."

    if not data.omp_med["critical"].empty and not med_static.empty and max_threads is not None:
        reduction_sp_3200 = safe_div(get_metric(med_static, 3200, "ciclos", threads=1), get_metric(med_static, 3200, "ciclos", threads=max_threads))
        critical_sp_3200 = safe_div(get_metric(data.omp_med["critical"], 3200, "ciclos", threads=1), get_metric(data.omp_med["critical"], 3200, "ciclos", threads=max_threads))
        reduction_text = (
            f"La comparación \\texttt{{reduction}} vs. \\texttt{{critical}} en $n=3200$ y {max_threads} hilos "
            f"da \\textbf{{{format_speedup_tex(reduction_sp_3200)}}} frente a \\textbf{{{format_speedup_tex(critical_sp_3200)}}}."
        )
    else:
        reduction_text = (
            "La comparación entre \\texttt{reduction} y \\texttt{critical} debe interpretarse "
            "solo cuando esté disponible el fichero \\texttt{resultados/v3\\_O3\\_critical.txt}."
        )

    return (
        "\\begin{enumerate}\n"
        f"  \\item Las optimizaciones de caché de v2 frente a v1 con \\texttt{{-O0}} producen speedups {speedup_range_tex(sp_v2_o0)}.\n"
        f"  \\item La versión SIMD (v4) mejora a v2\\texttt{{-O3}} con speedups {speedup_range_tex(sp_v4_vs_v2)}.\n"
        f"  \\item {static_text}\n"
        "  \\item La mejor política de reparto de OpenMP depende del tamaño del problema, por lo que conviene estudiar \\texttt{static}, \\texttt{dynamic(16)} y \\texttt{guided} de forma separada.\n"
        f"  \\item {reduction_text}\n"
        "\\end{enumerate}"
    )


def replace_marker_block(src: str, marker: str, body: str) -> tuple[str, bool]:
    pattern = re.compile(
        rf"(^[ \t]*% {re.escape(marker)}_BEGIN[ \t]*\n)(.*?)(\n^[ \t]*% {re.escape(marker)}_END[ \t]*$)",
        re.DOTALL | re.MULTILINE,
    )
    match = pattern.search(src)
    if not match:
        print(f"[AVISO] No se localizo el bloque marcado {marker}.")
        return src, False
    replaced = match.group(1) + body.rstrip() + "\n" + match.group(3)
    return src[: match.start()] + replaced + src[match.end() :], True


def patch_latex_blocks(src: str, data: AnalysisData) -> tuple[str, bool]:
    blocks = {
        "AUTO_ABSTRACT_RESULTS": generate_abstract_results(data),
        "AUTO_TABLE_ROWS": generate_table_rows(data),
        "AUTO_G1_TEXT": generate_g1_text(data),
        "AUTO_G2_TEXT": generate_g2_text(data),
        "AUTO_G3_TEXT": generate_g3_text(data),
        "AUTO_G4_TEXT": generate_g4_text(data),
        "AUTO_G5_TEXT": generate_g5_text(data),
        "AUTO_G6_TEXT": generate_g6_text(data),
        "AUTO_CONCLUSIONS": generate_conclusions(data),
    }

    changed = False
    for marker, body in blocks.items():
        src, block_changed = replace_marker_block(src, marker, body)
        changed = changed or block_changed
    return src, changed


def compile_latex_document(latex_file: str, plots_dir: str, data: AnalysisData) -> None:
    if not os.path.isfile(latex_file):
        print(f"[AVISO] No se encontro {latex_file}; se omite la compilacion LaTeX.")
        return

    if os.path.basename(plots_dir) != "graficas":
        print("[AVISO] La compilacion LaTeX se omite porque las graficas no se estan escribiendo en 'graficas/'.")
        return

    latex_bin = shutil.which("lualatex") or shutil.which("pdflatex")
    if latex_bin is None:
        print("[AVISO] No se encontro lualatex/pdflatex; se omite la compilacion LaTeX.")
        return

    with open(latex_file, encoding="utf-8") as handle:
        src = handle.read()

    patched_src, changed = patch_latex_blocks(src, data)
    build_tex = os.path.splitext(latex_file)[0] + "_build.tex"

    with open(build_tex, "w", encoding="utf-8") as handle:
        handle.write(patched_src)

    if changed:
        print(f"Tabla de medianas actualizada en {build_tex}.")

    for plot_name in PLOT_FILES.values():
        plot_path = os.path.join(plots_dir, plot_name)
        if not os.path.isfile(plot_path):
            print(f"[AVISO] Falta la grafica {plot_path}; la compilacion puede fallar.")

    command = [latex_bin, "-interaction=nonstopmode", "-halt-on-error", build_tex]
    for pass_idx in (1, 2):
        print(f"Compilando LaTeX (pasada {pass_idx}/2)...")
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            print("[ERROR] Fallo al compilar LaTeX.")
            for line in result.stdout.splitlines()[-25:]:
                print(f"  {line}")
            return

    generated_pdf = os.path.splitext(build_tex)[0] + ".pdf"
    if os.path.isfile(generated_pdf):
        shutil.move(generated_pdf, "Memoria.pdf")
        print("PDF actualizado: Memoria.pdf")


def run_analysis(results_dir: str = "resultados", plots_dir: str = "graficas", compile_latex: bool = True, latex_file: str = "Memoria.tex") -> AnalysisData:
    configure_style()
    print(f"Cargando resultados desde: {results_dir}")

    data = load_all_data(results_dir)
    print_result_consistency_warnings(data)

    ensure_plot_dir(plots_dir)
    generate_g1(data, plots_dir)
    generate_g2(data, plots_dir)
    generate_g3(data, plots_dir)
    generate_g4(data, plots_dir)
    generate_g5(data, plots_dir)
    generate_g6(data, plots_dir)
    print_summary(data)

    if compile_latex:
        compile_latex_document(latex_file, plots_dir, data)

    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analiza los resultados de la Practica 2 de AC.")
    parser.add_argument("--results-dir", default="resultados", help="Directorio con los ficheros de resultados.")
    parser.add_argument("--plots-dir", default="graficas", help="Directorio donde guardar las graficas.")
    parser.add_argument("--latex-file", default="Memoria.tex", help="Documento LaTeX a parchear y compilar.")
    parser.add_argument("--skip-latex", action="store_true", help="No recompilar la memoria en PDF.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_analysis(
        results_dir=args.results_dir,
        plots_dir=args.plots_dir,
        compile_latex=not args.skip_latex,
        latex_file=args.latex_file,
    )


if __name__ == "__main__":
    main()
