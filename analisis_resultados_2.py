"""
analisis_resultados.py  –  Análisis de experimentos Jacobi AC P2
=================================================================
Genera 6 gráficas en graficas/ y tabla de resumen en consola.

Gráficas:
  g1 – Speedup v2 vs v1-O0  (barras por n)
  g2 – Speedup v2/v3/v4(1hilo) vs v1-O3  (barras agrupadas)
  g3 – Speedup v3-O3 y v4-O3(1hilo) vs v2-O3  (barras)
  g4 – Speedup v4 por número de hilos [OBLIGATORIA]  (líneas + ideal)
  g5 – Comparación schedulings static/dynamic/guided por n
  g6 – Atomic vs critical por n
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from collections import defaultdict

# ── Directorios ────────────────────────────────────────────────
RES  = "resultados"
GRAF = "graficas2"
os.makedirs(GRAF, exist_ok=True)

SIZES   = [1250, 2000, 3200]
THREADS = [1, 2, 4, 8, 16, 32]
HZ      = 2.2e9          # frecuencia nominal FinisTerrae III (GHz)

# ══════════════════════════════════════════════════════════════
# CARGA DE DATOS
# ══════════════════════════════════════════════════════════════

def load_seq(fname):
    """Lee fichero secuencial: v<x> n iter norm2 ciclos
    Devuelve dict  {n: mediana_ciclos}"""
    data = defaultdict(list)
    with open(fname) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            n      = int(parts[1])
            ciclos = float(parts[4])
            data[n].append(ciclos)
    return {n: np.median(v) for n, v in data.items()}


def load_omp(fname):
    """Lee fichero OpenMP: v4 n threads iter norm2 ciclos
    Devuelve dict  {(n, threads): mediana_ciclos}"""
    data = defaultdict(list)
    with open(fname) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 6:
                continue
            n       = int(parts[1])
            threads = int(parts[2])
            ciclos  = float(parts[5])
            data[(n, threads)].append(ciclos)
    return {k: np.median(v) for k, v in data.items()}


# Cargar todos los ficheros
v1_O0 = load_seq(f"{RES}/v1_O0.txt")
v1_O3 = load_seq(f"{RES}/v1_O3.txt")
v2_O0 = load_seq(f"{RES}/v2_O0.txt")
v2_O3 = load_seq(f"{RES}/v2_O3.txt")
v3_O0 = load_seq(f"{RES}/v3_O0.txt")
v3_O3 = load_seq(f"{RES}/v3_O3.txt")
v4_static   = load_omp(f"{RES}/v4_O3_static.txt")
v4_dynamic  = load_omp(f"{RES}/v4_O3_dynamic.txt")
v4_guided   = load_omp(f"{RES}/v4_O3_guided.txt")
v4_critical = load_omp(f"{RES}/v4_O3_critical.txt")

# Filtrar outliers: para v4 con 32 hilos hay medidas muy dispersas
# Se eliminan valores > 3× mediana antes de recalcular
def load_omp_filtered(fname):
    data = defaultdict(list)
    with open(fname) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 6:
                continue
            n       = int(parts[1])
            threads = int(parts[2])
            ciclos  = float(parts[5])
            data[(n, threads)].append(ciclos)
    result = {}
    for k, v in data.items():
        med = np.median(v)
        filtered = [x for x in v if x < 3 * med]
        result[k] = np.median(filtered)
    return result

v4_static   = load_omp_filtered(f"{RES}/v4_O3_static.txt")
v4_dynamic  = load_omp_filtered(f"{RES}/v4_O3_dynamic.txt")
v4_guided   = load_omp_filtered(f"{RES}/v4_O3_guided.txt")
v4_critical = load_omp_filtered(f"{RES}/v4_O3_critical.txt")

# ══════════════════════════════════════════════════════════════
# TABLA DE RESUMEN
# ══════════════════════════════════════════════════════════════
print("=" * 70)
print(f"{'Versión':<18} {'n':>6} {'Ciclos (M)':>12} {'Tiempo (s)':>11} {'GFlops*':>8}")
print("=" * 70)
for n in SIZES:
    for label, d in [("v1-O0", v1_O0), ("v1-O3", v1_O3),
                     ("v2-O0", v2_O0), ("v2-O3", v2_O3),
                     ("v3-O0", v3_O0), ("v3-O3", v3_O3)]:
        if n in d:
            c = d[n]
            print(f"  {label:<16} {n:>6} {c/1e6:>12.1f} {c/HZ:>11.2f}")
    print()

print("\nv4-O3-static (mediana ciclos por configuración):")
print(f"{'n':>6} {'T':>4} {'Ciclos (M)':>12} {'Tiempo (s)':>11} {'Speedup vs T=1':>15}")
for n in SIZES:
    base = v4_static.get((n, 1), None)
    for t in THREADS:
        c = v4_static.get((n, t), None)
        if c and base:
            sp = base / c
            print(f"  {n:>6} {t:>4} {c/1e6:>12.1f} {c/HZ:>11.2f} {sp:>15.2f}x")
    print()

# ══════════════════════════════════════════════════════════════
# ESTILO COMÚN
# ══════════════════════════════════════════════════════════════
plt.rcParams.update({
    "figure.dpi":     150,
    "font.size":      11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize":10,
    "axes.grid":      True,
    "grid.alpha":     0.4,
})
COLORS = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800", "#00BCD4"]
N_LABELS = [str(n) for n in SIZES]

def save(name):
    path = f"{GRAF}/{name}.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  → {path}")

# ══════════════════════════════════════════════════════════════
# G1 – Speedup v2-O0 vs v1-O0  (efecto de optimizaciones caché sin -O3)
# ══════════════════════════════════════════════════════════════
sp_v2_vs_v1_O0 = [v1_O0[n] / v2_O0[n] for n in SIZES]

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(N_LABELS, sp_v2_vs_v1_O0, color=COLORS[0], width=0.5, edgecolor="white")
ax.axhline(1, color="gray", linewidth=1, linestyle="--")
for b, v in zip(bars, sp_v2_vs_v1_O0):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
            f"{v:.2f}×", ha="center", va="bottom", fontsize=10)
ax.set_xlabel("Tamaño n")
ax.set_ylabel("Speedup")
ax.set_title("G1 – Speedup v2 vs v1  (ambos -O0)\nEfecto puro de optimizaciones caché")
ax.set_ylim(0, max(sp_v2_vs_v1_O0) * 1.2)
save("g1_speedup_v2_vs_v1_O0")

# ══════════════════════════════════════════════════════════════
# G2 – Speedup v2/v3/v4(1hilo) vs v1-O3
# ══════════════════════════════════════════════════════════════
labels_g2 = ["v2-O0", "v2-O3", "v3-O0", "v3-O3", "v4-O3 (1T)"]
speedups_g2 = {lbl: [] for lbl in labels_g2}
for n in SIZES:
    ref = v1_O3[n]
    speedups_g2["v2-O0"].append(ref / v2_O0[n])
    speedups_g2["v2-O3"].append(ref / v2_O3[n])
    speedups_g2["v3-O0"].append(ref / v3_O0[n])
    speedups_g2["v3-O3"].append(ref / v3_O3[n])
    speedups_g2["v4-O3 (1T)"].append(ref / v4_static.get((n, 1), ref))

x   = np.arange(len(SIZES))
w   = 0.15
fig, ax = plt.subplots(figsize=(9, 5))
for i, (lbl, vals) in enumerate(speedups_g2.items()):
    offset = (i - 2) * w
    bars = ax.bar(x + offset, vals, w, label=lbl,
                  color=COLORS[i], edgecolor="white")
ax.axhline(1, color="gray", linewidth=1, linestyle="--")
ax.set_xticks(x); ax.set_xticklabels(N_LABELS)
ax.set_xlabel("Tamaño n")
ax.set_ylabel("Speedup")
ax.set_title("G2 – Speedup v2 / v3 / v4(1T) vs v1-O3\nReferencia: versión base compilada con -O3")
ax.legend(loc="upper left")
save("g2_speedup_vs_v1O3")

# ══════════════════════════════════════════════════════════════
# G3 – Speedup v3-O3 y v4-O3(1hilo) vs v2-O3
# ══════════════════════════════════════════════════════════════
sp_v3_vs_v2 = [v2_O3[n] / v3_O3[n] for n in SIZES]
sp_v4_vs_v2 = [v2_O3[n] / v4_static.get((n, 1), v2_O3[n]) for n in SIZES]

x = np.arange(len(SIZES))
w = 0.3
fig, ax = plt.subplots(figsize=(7, 4))
b1 = ax.bar(x - w/2, sp_v3_vs_v2, w, label="v3-O3 (SIMD)",   color=COLORS[2], edgecolor="white")
b2 = ax.bar(x + w/2, sp_v4_vs_v2, w, label="v4-O3 (1 hilo)", color=COLORS[3], edgecolor="white")
ax.axhline(1, color="gray", linewidth=1, linestyle="--")
for b, v in zip(list(b1)+list(b2), sp_v3_vs_v2+sp_v4_vs_v2):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.02,
            f"{v:.2f}×", ha="center", va="bottom", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(N_LABELS)
ax.set_xlabel("Tamaño n")
ax.set_ylabel("Speedup")
ax.set_title("G3 – Speedup v3 y v4(1T) vs v2-O3\nReferencia: mejor versión secuencial")
ax.legend()
save("g3_speedup_v3_v4_vs_v2O3")

# ══════════════════════════════════════════════════════════════
# G4 – Speedup v4 por número de hilos  [OBLIGATORIA]
# ══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))
for i, n in enumerate(SIZES):
    base = v4_static.get((n, 1))
    if base is None:
        continue
    speedups = [base / v4_static.get((n, t), base) for t in THREADS]
    ax.plot(THREADS, speedups, marker="o", linewidth=2,
            color=COLORS[i], label=f"n={n}")
# Línea ideal
ax.plot(THREADS, THREADS, linestyle="--", color="black",
        linewidth=1.2, label="Ideal (lineal)")
ax.set_xlabel("Número de hilos")
ax.set_ylabel("Speedup")
ax.set_title("G4 – Speedup v4-O3 (schedule static) por número de hilos\nvs ejecución con 1 hilo")
ax.set_xticks(THREADS)
ax.legend()
ax.set_xlim(0.5, 34)
ax.set_ylim(0)
save("g4_speedup_hilos")

# ══════════════════════════════════════════════════════════════
# G5 – Comparación schedulings static / dynamic / guided
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)
sched_data  = [v4_static, v4_dynamic, v4_guided]
sched_names = ["static", "dynamic(16)", "guided"]

for ax, n in zip(axes, SIZES):
    for i, (sd, sn) in enumerate(zip(sched_data, sched_names)):
        base = sd.get((n, 1))
        if base is None:
            continue
        speedups = [base / sd.get((n, t), base) for t in THREADS]
        ax.plot(THREADS, speedups, marker="o", linewidth=2,
                color=COLORS[i], label=sn)
    ax.plot(THREADS, THREADS, linestyle="--", color="black",
            linewidth=1, label="Ideal")
    ax.set_title(f"n = {n}")
    ax.set_xlabel("Hilos")
    ax.set_xticks(THREADS)
    ax.set_ylim(0)
    ax.legend(fontsize=8)

axes[0].set_ylabel("Speedup")
fig.suptitle("G5 – Comparación schedulings OpenMP (speedup vs 1 hilo)", fontsize=13)
plt.tight_layout()
save("g5_schedulings")

# ══════════════════════════════════════════════════════════════
# G6 – Atomic vs Critical
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)
for ax, n in zip(axes, SIZES):
    sp_at   = []
    sp_crit = []
    base_at   = v4_static.get((n, 1))
    base_crit = v4_critical.get((n, 1))
    for t in THREADS:
        sp_at.append(base_at   / v4_static.get((n, t), base_at))
        sp_crit.append(base_crit / v4_critical.get((n, t), base_crit))
    ax.plot(THREADS, sp_at,   marker="o", linewidth=2, color=COLORS[0], label="atomic")
    ax.plot(THREADS, sp_crit, marker="s", linewidth=2, color=COLORS[1], label="critical")
    ax.plot(THREADS, THREADS, linestyle="--", color="black", linewidth=1, label="Ideal")
    ax.set_title(f"n = {n}")
    ax.set_xlabel("Hilos")
    ax.set_xticks(THREADS)
    ax.set_ylim(0)
    ax.legend(fontsize=9)

axes[0].set_ylabel("Speedup")
fig.suptitle("G6 – Reducción atomic vs critical (speedup vs 1 hilo)", fontsize=13)
plt.tight_layout()
save("g6_atomic_vs_critical")

# ══════════════════════════════════════════════════════════════
# G7 – Eficiencia paralela  E = Speedup / nº_hilos
# ══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))
for i, n in enumerate(SIZES):
    base = v4_static.get((n, 1))
    if base is None:
        continue
    eficiencias = []
    for t in THREADS:
        sp = base / v4_static.get((n, t), base)
        eficiencias.append(sp / t)
    ax.plot(THREADS, eficiencias, marker="o", linewidth=2,
            color=COLORS[i], label=f"n={n}")

ax.axhline(1.0, linestyle="--", color="black", linewidth=1.2, label="Ideal (E=1)")
ax.set_xlabel("Número de hilos")
ax.set_ylabel("Eficiencia  E = Speedup / T")
ax.set_title("G7 – Eficiencia paralela v4-O3 (schedule static)\nE = Speedup / nº_hilos")
ax.set_xticks(THREADS)
ax.set_ylim(0, 1.2)
ax.legend()
save("g7_eficiencia_paralela")

print("\nTodas las gráficas generadas en graficas/")
