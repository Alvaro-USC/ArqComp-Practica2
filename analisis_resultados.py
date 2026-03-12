"""
analisis_resultados.py
======================
Lee los ficheros de resultados generados por los scripts SLURM y
produce todas las gráficas pedidas en el enunciado:

  1. Speedup v2 vs v1-O0  (optimizaciones caché vs referencia sin optimizar)
  2. Speedup v2, v3, v4 vs v1-O3  (todas las versiones vs referencia compilada)
  3. Speedup v3, v4 vs v2-O3  (vectorial y paralelo vs secuencial optimizado)
  4. Speedup v4 por número de hilos para cada N  (gráfica obligatoria del enunciado)
  5. Comparación schedulings OpenMP (static / dynamic / guided)
  6. Comparación atomic vs critical en v4

Formato esperado de cada fichero de resultados:
  · v1_O0.txt, v1_O3.txt, v2_O0.txt, v2_O3.txt
      <version> <n> <iter> <norm2> <ciclos>
      Ejemplo: v1 1250 87 4.231234e-10 125438920

  · v3_O0.txt, v3_O3.txt
      <version> <n> <iter> <norm2> <ciclos>

  · v4_O3_static.txt, v4_O3_dynamic.txt, v4_O3_guided.txt, v4_O3_critical.txt
      <version> <n> <threads> <iter> <norm2> <ciclos>
      Ejemplo: v4 1250 4 87 4.231234e-10 98230120

Uso:
  python3 analisis_resultados.py
  (ejecutar desde el directorio que contiene la carpeta 'resultados/')
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Configuración de estilo ──────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":       150,
    "font.size":        11,
    "axes.grid":        True,
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
    "lines.linewidth":  2,
    "lines.markersize": 7,
})

RESULTS_DIR = "resultados"
PLOTS_DIR   = "graficas"
os.makedirs(PLOTS_DIR, exist_ok=True)

SIZES   = [1250, 2000, 3200]
COLORS  = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
MARKERS = ["o", "s", "^", "D", "v"]

# ── Funciones de carga ────────────────────────────────────────────────────────

def load_sequential(filename):
    """
    Carga ficheros de v1/v2/v3 con formato:
      <version> <n> <iter> <norm2> <ciclos>
    Devuelve DataFrame con columnas: version, n, iter, norm2, ciclos
    """
    rows = []
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.isfile(path):
        print(f"  [AVISO] No se encontró: {path}")
        return pd.DataFrame()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                rows.append({
                    "version": parts[0],
                    "n":       int(parts[1]),
                    "iter":    int(parts[2]),
                    "norm2":   float(parts[3]),
                    "ciclos":  float(parts[4]),
                })
            except ValueError:
                continue
    return pd.DataFrame(rows)


def load_openmp(filename):
    """
    Carga ficheros de v4 con formato:
      <version> <n> <threads> <iter> <norm2> <ciclos>
    Devuelve DataFrame con columnas: version, n, threads, iter, norm2, ciclos
    """
    rows = []
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.isfile(path):
        print(f"  [AVISO] No se encontró: {path}")
        return pd.DataFrame()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            try:
                rows.append({
                    "version": parts[0],
                    "n":       int(parts[1]),
                    "threads": int(parts[2]),
                    "iter":    int(parts[3]),
                    "norm2":   float(parts[4]),
                    "ciclos":  float(parts[5]),
                })
            except ValueError:
                continue
    return pd.DataFrame(rows)


def median_ciclos(df, group_cols):
    """Agrupa por group_cols y calcula la mediana de ciclos."""
    if df.empty:
        return pd.DataFrame()
    return df.groupby(group_cols)["ciclos"].median().reset_index()


# ── Carga de todos los ficheros ───────────────────────────────────────────────

print("Cargando datos...")
df_v1_O0 = load_sequential("v1_O0.txt")
df_v1_O3 = load_sequential("v1_O3.txt")
df_v2_O0 = load_sequential("v2_O0.txt")
df_v2_O3 = load_sequential("v2_O3.txt")
df_v3_O0 = load_sequential("v3_O0.txt")
df_v3_O3 = load_sequential("v3_O3.txt")
df_v4_static  = load_openmp("v4_O3_static.txt")
df_v4_dynamic = load_openmp("v4_O3_dynamic.txt")
df_v4_guided  = load_openmp("v4_O3_guided.txt")
df_v4_critical= load_openmp("v4_O3_critical.txt")


# ── Medianas ─────────────────────────────────────────────────────────────────

med_v1_O0 = median_ciclos(df_v1_O0, ["n"])
med_v1_O3 = median_ciclos(df_v1_O3, ["n"])
med_v2_O0 = median_ciclos(df_v2_O0, ["n"])
med_v2_O3 = median_ciclos(df_v2_O3, ["n"])
med_v3_O0 = median_ciclos(df_v3_O0, ["n"])
med_v3_O3 = median_ciclos(df_v3_O3, ["n"])
med_v4_static   = median_ciclos(df_v4_static,   ["n", "threads"])
med_v4_dynamic  = median_ciclos(df_v4_dynamic,  ["n", "threads"])
med_v4_guided   = median_ciclos(df_v4_guided,   ["n", "threads"])
med_v4_critical = median_ciclos(df_v4_critical, ["n", "threads"])


def get_ciclos(med_df, n_val, threads_val=None):
    """Extrae el valor de ciclos para un n (y opcionalmente threads) dado."""
    if med_df.empty:
        return np.nan
    if threads_val is not None:
        row = med_df[(med_df["n"] == n_val) & (med_df["threads"] == threads_val)]
    else:
        row = med_df[med_df["n"] == n_val]
    if row.empty:
        return np.nan
    return row["ciclos"].values[0]


# ════════════════════════════════════════════════════════════════════════════
# GRÁFICA 1 — Speedup de v2-O0 y v2-O3 respecto a v1-O0
#             (ganancia de las optimizaciones caché respecto a la ref. sin opt.)
# ════════════════════════════════════════════════════════════════════════════
print("Generando Gráfica 1: Speedup v2 vs v1-O0 ...")

fig, ax = plt.subplots()
x = np.arange(len(SIZES))
width = 0.35

sp_v2_O0 = [get_ciclos(med_v1_O0, n) / get_ciclos(med_v2_O0, n) for n in SIZES]
sp_v2_O3 = [get_ciclos(med_v1_O0, n) / get_ciclos(med_v2_O3, n) for n in SIZES]

bars1 = ax.bar(x - width/2, sp_v2_O0, width, label="v2 -O0 / v1 -O0",
               color=COLORS[0], alpha=0.85)
bars2 = ax.bar(x + width/2, sp_v2_O3, width, label="v2 -O3 / v1 -O0",
               color=COLORS[1], alpha=0.85)

ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
ax.set_xlabel("Tamaño del problema (n)")
ax.set_ylabel("Speedup (ciclos v1-O0 / ciclos vX)")
ax.set_title("Speedup de v2 (optimizaciones caché) respecto a v1 -O0")
ax.set_xticks(x)
ax.set_xticklabels([str(n) for n in SIZES])
ax.legend()

for bar in list(bars1) + list(bars2):
    h = bar.get_height()
    if not np.isnan(h):
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.02,
                f"{h:.2f}×", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "g1_speedup_v2_vs_v1O0.png"))
plt.close()


# ════════════════════════════════════════════════════════════════════════════
# GRÁFICA 2 — Speedup de v2, v3, v4 respecto a v1-O3
#             (todas las versiones optimizadas vs referencia compilada)
# ════════════════════════════════════════════════════════════════════════════
print("Generando Gráfica 2: Speedup v2/v3/v4 vs v1-O3 ...")

fig, ax = plt.subplots()
x = np.arange(len(SIZES))
width = 0.2

labels_sp2 = ["v2 -O3", "v3 -O3", "v4 1hilo", "v4 max hilos"]
data_sp2 = []

for n in SIZES:
    base = get_ciclos(med_v1_O3, n)
    # v4: 1 hilo y el máximo de hilos disponible en los datos
    v4_1t   = get_ciclos(med_v4_static, n, 1)
    max_t   = df_v4_static["threads"].max() if not df_v4_static.empty else 1
    v4_maxt = get_ciclos(med_v4_static, n, max_t)
    data_sp2.append([
        base / get_ciclos(med_v2_O3, n),
        base / get_ciclos(med_v3_O3, n),
        base / v4_1t,
        base / v4_maxt,
    ])

data_sp2 = np.array(data_sp2, dtype=float)   # shape (3 tamaños, 4 versiones)

offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * width
for k, (label, color, marker) in enumerate(zip(labels_sp2, COLORS, MARKERS)):
    ax.bar(x + offsets[k], data_sp2[:, k], width,
           label=label, color=color, alpha=0.85)

ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
ax.set_xlabel("Tamaño del problema (n)")
ax.set_ylabel("Speedup (ciclos v1-O3 / ciclos vX)")
ax.set_title("Speedup de v2, v3, v4 respecto a v1 -O3")
ax.set_xticks(x)
ax.set_xticklabels([str(n) for n in SIZES])
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "g2_speedup_all_vs_v1O3.png"))
plt.close()


# ════════════════════════════════════════════════════════════════════════════
# GRÁFICA 3 — Speedup de v3 y v4 respecto a v2-O3
#             (vectorial y paralelo vs secuencial optimizado)
# ════════════════════════════════════════════════════════════════════════════
print("Generando Gráfica 3: Speedup v3/v4 vs v2-O3 ...")

fig, ax = plt.subplots()
x = np.arange(len(SIZES))
width = 0.2

if not df_v4_static.empty:
    thread_vals = sorted(df_v4_static["threads"].unique())
else:
    thread_vals = [1]

for k, T in enumerate(thread_vals):
    sp = [get_ciclos(med_v2_O3, n) / get_ciclos(med_v4_static, n, T)
          for n in SIZES]
    ax.plot(SIZES, sp, label=f"v4 {T} hilos",
            color=COLORS[k % len(COLORS)], marker=MARKERS[k % len(MARKERS)])

sp_v3 = [get_ciclos(med_v2_O3, n) / get_ciclos(med_v3_O3, n) for n in SIZES]
ax.plot(SIZES, sp_v3, label="v3 (AVX256)", color="black",
        marker="*", linewidth=2.5, markersize=10)

ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
ax.set_xlabel("Tamaño del problema (n)")
ax.set_ylabel("Speedup (ciclos v2-O3 / ciclos vX)")
ax.set_title("Speedup de v3 y v4 respecto a v2 (secuencial optimizado) -O3")
ax.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "g3_speedup_v3v4_vs_v2O3.png"))
plt.close()


# ════════════════════════════════════════════════════════════════════════════
# GRÁFICA 4 — Speedup v4 por número de hilos, una curva por cada N
#             (GRÁFICA OBLIGATORIA del enunciado apartado iii)
# ════════════════════════════════════════════════════════════════════════════
print("Generando Gráfica 4: Speedup v4 por hilos (obligatoria) ...")

if not med_v4_static.empty:
    fig, ax = plt.subplots()

    all_threads = sorted(med_v4_static["threads"].unique())

    for k, n in enumerate(SIZES):
        # Speedup relativo a 1 hilo del mismo n
        base_1t = get_ciclos(med_v4_static, n, 1)
        sp = [base_1t / get_ciclos(med_v4_static, n, T) for T in all_threads]
        ax.plot(all_threads, sp, label=f"n={n}",
                color=COLORS[k], marker=MARKERS[k])

    # Speedup lineal ideal
    ax.plot(all_threads, all_threads, "k--", label="Ideal lineal", linewidth=1)

    ax.set_xlabel("Número de hilos")
    ax.set_ylabel("Speedup (ciclos 1 hilo / ciclos T hilos)")
    ax.set_title("Speedup de v4 (OpenMP) en función del número de hilos")
    ax.set_xticks(all_threads)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "g4_speedup_v4_hilos.png"))
    plt.close()
else:
    print("  [AVISO] Sin datos de v4 estático; omitiendo Gráfica 4.")


# ════════════════════════════════════════════════════════════════════════════
# GRÁFICA 5 — Comparación de schedulings (static / dynamic / guided) en v4
# ════════════════════════════════════════════════════════════════════════════
print("Generando Gráfica 5: Comparación schedulings v4 ...")

sched_data = {
    "static":  med_v4_static,
    "dynamic": med_v4_dynamic,
    "guided":  med_v4_guided,
}

for n in SIZES:
    fig, ax = plt.subplots()
    for k, (sched, med_df) in enumerate(sched_data.items()):
        if med_df.empty:
            continue
        threads_available = sorted(med_df["threads"].unique())
        ciclos_vals = [get_ciclos(med_df, n, T) for T in threads_available]
        ax.plot(threads_available, ciclos_vals,
                label=sched, color=COLORS[k], marker=MARKERS[k])

    ax.set_xlabel("Número de hilos")
    ax.set_ylabel("Ciclos (mediana)")
    ax.set_title(f"Comparación de schedulings — v4, n={n}")
    ax.set_xticks(threads_available if not med_v4_static.empty
                  else [1, 2, 4, 8, 16, 32])
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda val, _: f"{val/1e9:.2f}G"))
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"g5_schedulings_n{n}.png"))
    plt.close()


# ════════════════════════════════════════════════════════════════════════════
# GRÁFICA 6 — Comparación atomic vs critical en v4
# ════════════════════════════════════════════════════════════════════════════
print("Generando Gráfica 6: Atomic vs Critical en v4 ...")

if not med_v4_static.empty and not med_v4_critical.empty:
    for n in SIZES:
        fig, ax = plt.subplots()
        threads_s = sorted(med_v4_static["threads"].unique())
        threads_c = sorted(med_v4_critical["threads"].unique())

        sp_atomic   = [get_ciclos(med_v4_static,   n, T) for T in threads_s]
        sp_critical = [get_ciclos(med_v4_critical, n, T) for T in threads_c]

        ax.plot(threads_s, sp_atomic,   label="atomic",   color=COLORS[0], marker="o")
        ax.plot(threads_c, sp_critical, label="critical",  color=COLORS[3], marker="s")

        ax.set_xlabel("Número de hilos")
        ax.set_ylabel("Ciclos (mediana)")
        ax.set_title(f"Reducción: atomic vs critical — v4, n={n}")
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(
            lambda val, _: f"{val/1e9:.2f}G"))
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"g6_atomic_critical_n{n}.png"))
        plt.close()
else:
    print("  [AVISO] Faltan datos de atomic o critical; omitiendo Gráfica 6.")


# ════════════════════════════════════════════════════════════════════════════
# TABLA RESUMEN — medianas de ciclos por versión y tamaño
# ════════════════════════════════════════════════════════════════════════════
print("\n══════════════════════════════════════════════════════")
print("TABLA RESUMEN — Ciclos (mediana)")
print("══════════════════════════════════════════════════════")
print(f"{'Versión':<18}  {'n=1250':>14}  {'n=2000':>14}  {'n=3200':>14}")
print("-" * 66)

rows_tabla = [
    ("v1 -O0",         med_v1_O0),
    ("v1 -O3",         med_v1_O3),
    ("v2 -O0",         med_v2_O0),
    ("v2 -O3",         med_v2_O3),
    ("v3 -O0 (AVX)",   med_v3_O0),
    ("v3 -O3 (AVX)",   med_v3_O3),
]
for label, med_df in rows_tabla:
    vals = [get_ciclos(med_df, n) for n in SIZES]
    formatted = ["  N/A" if np.isnan(v) else f"{v/1e9:>12.3f}G" for v in vals]
    print(f"{label:<18}  {'  '.join(formatted)}")

if not med_v4_static.empty:
    max_t = med_v4_static["threads"].max()
    for T in [1, max_t]:
        vals = [get_ciclos(med_v4_static, n, T) for n in SIZES]
        formatted = ["  N/A" if np.isnan(v) else f"{v/1e9:>12.3f}G" for v in vals]
        print(f"{'v4 -O3 '+str(T)+'h':<18}  {'  '.join(formatted)}")

print("══════════════════════════════════════════════════════")
print(f"\nGráficas guardadas en: {PLOTS_DIR}/")
print("Ficheros generados:")
for f in sorted(os.listdir(PLOTS_DIR)):
    print(f"  {f}")