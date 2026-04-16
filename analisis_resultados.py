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
        print("  [AVISO] Faltan datos de static o critical; se omite G6.")
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


def build_table_lookup(data: AnalysisData) -> Dict[str, List[float]]:
    med_static = data.omp_med["static"]
    table = {
        "v1_O0": [get_metric(data.seq_med["v1_O0"], n, "ciclos") for n in SIZES],
        "v1_O3": [get_metric(data.seq_med["v1_O3"], n, "ciclos") for n in SIZES],
        "v2_O0": [get_metric(data.seq_med["v2_O0"], n, "ciclos") for n in SIZES],
        "v2_O3": [get_metric(data.seq_med["v2_O3"], n, "ciclos") for n in SIZES],
        "v4_O0": [get_metric(data.seq_med["v4_O0"], n, "ciclos") for n in SIZES],
        "v4_O3": [get_metric(data.seq_med["v4_O3"], n, "ciclos") for n in SIZES],
    }

    if not med_static.empty:
        for threads in sorted(med_static["threads"].unique()):
            table[f"v3_static_{int(threads)}"] = [get_metric(med_static, n, "ciclos", threads=int(threads)) for n in SIZES]

    return table


def format_tex_value(value: float) -> str:
    return "---" if np.isnan(value) else f"{value / 1e9:.1f}"


def detect_table_scheme(row_labels: Iterable[str]) -> str:
    labels = list(row_labels)
    has_legacy_v3_o0 = any(re.search(r"v3\s+\\texttt\{-O0\}", label) for label in labels)
    has_legacy_v4_threads = any(re.search(r"v4\s+\\texttt\{-O3\}\s*\(\d+T\)", label) for label in labels)
    return "legacy_swapped" if has_legacy_v3_o0 or has_legacy_v4_threads else "canonical"


def values_for_row_label(label: str, scheme: str, table_lookup: Dict[str, List[float]]) -> Optional[List[float]]:
    if re.search(r"v1\s+\\texttt\{-O0\}", label):
        return table_lookup.get("v1_O0")
    if re.search(r"v1\s+\\texttt\{-O3\}", label):
        return table_lookup.get("v1_O3")
    if re.search(r"v2\s+\\texttt\{-O0\}", label):
        return table_lookup.get("v2_O0")
    if re.search(r"v2\s+\\texttt\{-O3\}", label):
        return table_lookup.get("v2_O3")

    thread_match = re.search(r"\((\d+)T\)", label)
    threads = int(thread_match.group(1)) if thread_match else None

    if scheme == "legacy_swapped":
        if re.search(r"v3\s+\\texttt\{-O0\}", label):
            return table_lookup.get("v4_O0")
        if re.search(r"v3\s+\\texttt\{-O3\}", label) and threads is None:
            return table_lookup.get("v4_O3")
        if re.search(r"v4\s+\\texttt\{-O3\}", label) and threads is not None:
            return table_lookup.get(f"v3_static_{threads}")
    else:
        if re.search(r"v4\s+\\texttt\{-O0\}", label):
            return table_lookup.get("v4_O0")
        if re.search(r"v4\s+\\texttt\{-O3\}", label) and threads is None:
            return table_lookup.get("v4_O3")
        if re.search(r"v3\s+\\texttt\{-O3\}", label) and threads is not None:
            return table_lookup.get(f"v3_static_{threads}")

    return None


def patch_latex_table(src: str, table_lookup: Dict[str, List[float]]) -> tuple[str, bool]:
    pattern = re.compile(r"(\\label\{tab:medianas\}.*?\\midrule\n)(.*?)(\\bottomrule)", re.DOTALL)
    match = pattern.search(src)
    if not match:
        print("[AVISO] No se localizo la tabla tab:medianas en el .tex.")
        return src, False

    old_rows = []
    for raw_line in match.group(2).splitlines():
        line = raw_line.strip()
        if "&" in line and line.endswith("\\\\"):
            old_rows.append(line)

    row_labels = [line.split("&", 1)[0].strip() for line in old_rows]
    scheme = detect_table_scheme(row_labels)

    if scheme == "legacy_swapped":
        print("[AVISO] Memoria.tex parece usar la numeracion antigua (v3=SIMD, v4=OpenMP).")
        print("        Se actualizan solo los valores numericos respetando las etiquetas existentes.")

    new_rows = []
    for line, label in zip(old_rows, row_labels):
        values = values_for_row_label(label, scheme, table_lookup)
        if values is None:
            new_rows.append(line)
            continue
        cols = " & ".join(format_tex_value(value) for value in values)
        new_rows.append(f"{label} & {cols} \\\\")

    new_block = match.group(1) + "\n".join(new_rows) + "\n" + match.group(3)
    new_src = src[: match.start()] + new_block + src[match.end() :]
    return new_src, True


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

    patched_src, changed = patch_latex_table(src, build_table_lookup(data))
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
