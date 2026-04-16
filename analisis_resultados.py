"""
analisis_resultados.py
======================
Lee los ficheros de resultados generados por los scripts SLURM,
produce todas las gráficas y recompila la memoria en PDF.

Formato de ficheros de resultados:
  · v1_O0.txt, v1_O3.txt, v2_O0.txt, v2_O3.txt, v4_O0.txt, v4_O3.txt
      <version> <n> <iter> <norm2> <ciclos>

  · v3_O3_static.txt, v3_O3_dynamic.txt, v3_O3_guided.txt, v3_O3_critical.txt
      <version> <n> <threads> <iter> <norm2> <ciclos>

Uso:
  python3 analisis_resultados.py
  (ejecutar desde el directorio que contiene 'resultados/' y 'Memoria.tex')
"""

import os
import re
import shutil
import subprocess
import sys
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
TEX_FILE    = "Memoria.tex"          # fuente LaTeX (se lee pero NO se modifica)
TEX_WORK    = "Memoria_build.tex"    # copia de trabajo donde se parchean datos
PDF_OUT     = "Memoria.pdf"

os.makedirs(PLOTS_DIR, exist_ok=True)

SIZES   = [1250, 2000, 3200]
COLORS  = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
MARKERS = ["o", "s", "^", "D", "v"]

# ── Nombres de imagen que usa el .tex (deben coincidir exactamente) ──────────
# Mapeamos: nombre_clave -> ruta relativa que \includegraphics referencia en el .tex
IMG_NAMES = {
    "g1": "graficas/g1_speedup_v2_vs_v1_O0.png",
    "g2": "graficas/g2_speedup_vs_v1O3.png",
    "g3": "graficas/g3_speedup_v3_v4_vs_v2O3.png",
    "g4": "graficas/g4_speedup_hilos.png",
    "g5": "graficas/g5_schedulings.png",
    "g6": "graficas/g6_atomic_vs_critical.png",
}


# ══════════════════════════════════════════════════════════════════════════════
# FUNCIONES DE CARGA
# ══════════════════════════════════════════════════════════════════════════════

def load_sequential(filename):
    """Carga v1/v2/v4: <version> <n> <iter> <norm2> <ciclos>"""
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
    """Carga v3: <version> <n> <threads> <iter> <norm2> <ciclos>"""
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
    if df.empty:
        return pd.DataFrame()
    return df.groupby(group_cols)["ciclos"].median().reset_index()


def get_ciclos(med_df, n_val, threads_val=None):
    if med_df.empty:
        return np.nan
    if threads_val is not None:
        row = med_df[(med_df["n"] == n_val) & (med_df["threads"] == threads_val)]
    else:
        row = med_df[med_df["n"] == n_val]
    if row.empty:
        return np.nan
    return row["ciclos"].values[0]


# ══════════════════════════════════════════════════════════════════════════════
# CARGA DE DATOS
# ══════════════════════════════════════════════════════════════════════════════

print("Cargando datos...")
df_v1_O0 = load_sequential("v1_O0.txt")
df_v1_O3 = load_sequential("v1_O3.txt")
df_v2_O0 = load_sequential("v2_O0.txt")
df_v2_O3 = load_sequential("v2_O3.txt")
df_v4_O0 = load_sequential("v4_O0.txt")
df_v4_O3 = load_sequential("v4_O3.txt")

df_v3_static   = load_openmp("v3_O3_static.txt")
df_v3_dynamic  = load_openmp("v3_O3_dynamic.txt")
df_v3_guided   = load_openmp("v3_O3_guided.txt")
df_v3_critical = load_openmp("v3_O3_critical.txt")

med_v1_O0 = median_ciclos(df_v1_O0, ["n"])
med_v1_O3 = median_ciclos(df_v1_O3, ["n"])
med_v2_O0 = median_ciclos(df_v2_O0, ["n"])
med_v2_O3 = median_ciclos(df_v2_O3, ["n"])
med_v4_O0 = median_ciclos(df_v4_O0, ["n"])
med_v4_O3 = median_ciclos(df_v4_O3, ["n"])

med_v3_static   = median_ciclos(df_v3_static,   ["n", "threads"])
med_v3_dynamic  = median_ciclos(df_v3_dynamic,  ["n", "threads"])
med_v3_guided   = median_ciclos(df_v3_guided,   ["n", "threads"])
med_v3_critical = median_ciclos(df_v3_critical, ["n", "threads"])


# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICA 1 — Speedup v2-O0 vs v1-O0  (solo barras -O0, ambas referencia -O0)
# Nombre exacto que referencia el .tex: g1_speedup_v2_vs_v1_O0.png
# ══════════════════════════════════════════════════════════════════════════════
print("Generando Gráfica 1: Speedup v2 vs v1-O0 ...")

fig, ax = plt.subplots()
x     = np.arange(len(SIZES))
width = 0.5

sp_v2_O0 = [get_ciclos(med_v1_O0, n) / get_ciclos(med_v2_O0, n) for n in SIZES]

bars = ax.bar(x, sp_v2_O0, width, color=COLORS[0], alpha=0.85)
ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
ax.set_xlabel("Tamaño n")
ax.set_ylabel("Speedup")
ax.set_title("G1 – Speedup v2 vs v1  (ambos -O0)\nEfecto puro de optimizaciones caché")
ax.set_xticks(x)
ax.set_xticklabels([str(n) for n in SIZES])

for bar, val in zip(bars, sp_v2_O0):
    if not np.isnan(val):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                f"{val:.2f}×", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig(IMG_NAMES["g1"])
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICA 2 — Speedup v2/v3/v4 vs v1-O3
# Nombre: g2_speedup_vs_v1O3.png
# ══════════════════════════════════════════════════════════════════════════════
print("Generando Gráfica 2: Speedup v2/v4/v3 vs v1-O3 ...")

fig, ax = plt.subplots()
x     = np.arange(len(SIZES))
width = 0.15

labels_sp2  = ["v2-O0", "v2-O3", "v3-O0", "v3-O3", "v4-O3 (1T)"]
bar_colors  = [COLORS[0], COLORS[1], "#2ca02c", "#9467bd", "#ff7f0e"]

if not df_v3_static.empty:
    max_t = int(df_v3_static["threads"].max())
else:
    max_t = 1

data_sp2 = []
for n in SIZES:
    base = get_ciclos(med_v1_O3, n)
    # v3-O0: 1 hilo con -O0 no existe en nuestros ficheros; usamos med_v4_O0 si disponible
    # Aquí usamos v3_static con 1 hilo como proxy de "v3 secuencial -O3"
    v3_O0_proxy = get_ciclos(med_v4_O0, n)   # si existe v4_O0; si no, NaN
    data_sp2.append([
        base / get_ciclos(med_v2_O0, n),
        base / get_ciclos(med_v2_O3, n),
        base / v3_O0_proxy,
        base / get_ciclos(med_v3_static, n, 1),
        base / get_ciclos(med_v4_O3, n),
    ])

data_sp2 = np.array(data_sp2, dtype=float)
offsets = np.array([-2, -1, 0, 1, 2]) * width

for k, (label, color) in enumerate(zip(labels_sp2, bar_colors)):
    vals = data_sp2[:, k]
    valid = ~np.isnan(vals)
    ax.bar(x[valid] + offsets[k], vals[valid], width,
           label=label, color=color, alpha=0.85)

ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
ax.set_xlabel("Tamaño n")
ax.set_ylabel("Speedup")
ax.set_title("G2 – Speedup v2 / v3 / v4(1T) vs v1-O3\nReferencia: versión base compilada con -O3")
ax.set_xticks(x)
ax.set_xticklabels([str(n) for n in SIZES])
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(IMG_NAMES["g2"])
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICA 3 — Speedup v3(SIMD) y v4(1T) vs v2-O3
# Nombre: g3_speedup_v3_v4_vs_v2O3.png
# ══════════════════════════════════════════════════════════════════════════════
print("Generando Gráfica 3: Speedup v3/v4 vs v2-O3 ...")

fig, ax = plt.subplots()
x     = np.arange(len(SIZES))
width = 0.3

sp_v3 = [get_ciclos(med_v2_O3, n) / get_ciclos(med_v3_static, n, 1) for n in SIZES]
sp_v4 = [get_ciclos(med_v2_O3, n) / get_ciclos(med_v4_O3, n)        for n in SIZES]

bars3 = ax.bar(x - width/2, sp_v3, width, label="v3-O3 (SIMD)",    color="#2ca02c", alpha=0.85)
bars4 = ax.bar(x + width/2, sp_v4, width, label="v4-O3 (1 hilo)",  color="#9467bd", alpha=0.85)

ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
ax.set_xlabel("Tamaño n")
ax.set_ylabel("Speedup")
ax.set_title("G3 – Speedup v3 y v4(1T) vs v2-O3\nReferencia: mejor versión secuencial")
ax.set_xticks(x)
ax.set_xticklabels([str(n) for n in SIZES])
ax.legend()

for bar, val in zip(list(bars3) + list(bars4), sp_v3 + sp_v4):
    if not np.isnan(val):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                f"{val:.2f}×", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig(IMG_NAMES["g3"])
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICA 4 — Speedup v3 (schedule static) por número de hilos
# Nombre: g4_speedup_hilos.png
# ══════════════════════════════════════════════════════════════════════════════
print("Generando Gráfica 4: Speedup v3 por hilos ...")

if not med_v3_static.empty:
    fig, ax = plt.subplots()
    all_threads = sorted(med_v3_static["threads"].unique())

    for k, n in enumerate(SIZES):
        base_1t = get_ciclos(med_v3_static, n, 1)
        sp = [base_1t / get_ciclos(med_v3_static, n, T) for T in all_threads]
        ax.plot(all_threads, sp, label=f"n={n}",
                color=COLORS[k], marker=MARKERS[k])

    ax.plot(all_threads, all_threads, "k--", label="Ideal (lineal)", linewidth=1)
    ax.set_xlabel("Número de hilos")
    ax.set_ylabel("Speedup")
    ax.set_title("G4 – Speedup v4-O3 (schedule static) por número de hilos\nvs ejecución con 1 hilo")
    ax.set_xticks(all_threads)
    ax.legend()
    plt.tight_layout()
    plt.savefig(IMG_NAMES["g4"])
    plt.close()
else:
    print("  [AVISO] Sin datos de v3 estático; omitiendo Gráfica 4.")


# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICA 5 — Comparación schedulings (static / dynamic / guided), speedup vs 1h
# Nombre: g5_schedulings.png  (figura* de ancho completo en el .tex)
# ══════════════════════════════════════════════════════════════════════════════
print("Generando Gráfica 5: Comparación schedulings v3 ...")

sched_data = {
    "static":      med_v3_static,
    "dynamic(16)": med_v3_dynamic,
    "guided":      med_v3_guided,
}
sched_colors = [COLORS[0], COLORS[1], "#2ca02c"]

fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
fig.suptitle("G5 – Comparación schedulings OpenMP (speedup vs 1 hilo)")

for ax, n in zip(axes, SIZES):
    ax.set_title(f"n = {n}")
    for k, (sched, med_df) in enumerate(sched_data.items()):
        if med_df.empty:
            continue
        threads_avail = sorted(med_df["threads"].unique())
        base_1t = get_ciclos(med_df, n, 1)
        sp = [base_1t / get_ciclos(med_df, n, T) for T in threads_avail]
        ax.plot(threads_avail, sp, label=sched,
                color=sched_colors[k], marker=MARKERS[k])

    if not med_v3_static.empty:
        thr = sorted(med_v3_static["threads"].unique())
        ax.plot(thr, thr, "k--", label="Ideal", linewidth=1)
        ax.set_xticks(thr)

    ax.set_xlabel("Hilos")
    ax.set_ylabel("Speedup")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(IMG_NAMES["g5"])
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICA 6 — atomic vs critical (speedup vs 1 hilo)
# Nombre: g6_atomic_vs_critical.png  (figura* de ancho completo en el .tex)
# ══════════════════════════════════════════════════════════════════════════════
print("Generando Gráfica 6: atomic vs critical ...")

# "atomic" es la variante static (reduction nativa o atomic); "critical" es v3_critical
if not med_v3_static.empty and not med_v3_critical.empty:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
    fig.suptitle("G6 – Reducción atomic vs critical (speedup vs 1 hilo)")

    for ax, n in zip(axes, SIZES):
        ax.set_title(f"n = {n}")
        threads_s = sorted(med_v3_static["threads"].unique())
        threads_c = sorted(med_v3_critical["threads"].unique())

        base_1t_s = get_ciclos(med_v3_static,   n, 1)
        base_1t_c = get_ciclos(med_v3_critical, n, 1)

        sp_atom = [base_1t_s / get_ciclos(med_v3_static,   n, T) for T in threads_s]
        sp_crit = [base_1t_c / get_ciclos(med_v3_critical, n, T) for T in threads_c]

        ax.plot(threads_s, sp_atom, label="atomic",   color=COLORS[0], marker="o")
        ax.plot(threads_c, sp_crit, label="critical", color=COLORS[1], marker="s")

        if threads_s:
            ax.plot(threads_s, threads_s, "k--", label="Ideal", linewidth=1)
            ax.set_xticks(threads_s)

        ax.set_xlabel("Hilos")
        ax.set_ylabel("Speedup")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(IMG_NAMES["g6"])
    plt.close()
else:
    print("  [AVISO] Faltan datos atomic/critical; omitiendo Gráfica 6.")


# ══════════════════════════════════════════════════════════════════════════════
# TABLA RESUMEN — impresión en consola
# ══════════════════════════════════════════════════════════════════════════════
print("\n══════════════════════════════════════════════════════")
print("TABLA RESUMEN — Ciclos (mediana, ×10⁹)")
print("══════════════════════════════════════════════════════")
print(f"{'Versión':<20}  {'n=1250':>12}  {'n=2000':>12}  {'n=3200':>12}")
print("-" * 64)

def fmt(v):
    return "  N/A" if np.isnan(v) else f"{v/1e9:>10.3f}G"

rows_tabla = [
    ("v1 -O0",        med_v1_O0, None),
    ("v1 -O3",        med_v1_O3, None),
    ("v2 -O0",        med_v2_O0, None),
    ("v2 -O3",        med_v2_O3, None),
    ("v4 -O0 (SIMD)", med_v4_O0, None),
    ("v4 -O3 (SIMD)", med_v4_O3, None),
]
table_data = {}   # para parchear el .tex
for label, med_df, _ in rows_tabla:
    vals = [get_ciclos(med_df, n) for n in SIZES]
    table_data[label] = vals
    print(f"{label:<20}  {'  '.join(fmt(v) for v in vals)}")

if not med_v3_static.empty:
    max_t = int(med_v3_static["threads"].max())
    for T in [1, max_t]:
        lbl  = f"v3 -O3 (OMP) {T}h"
        vals = [get_ciclos(med_v3_static, n, T) for n in SIZES]
        table_data[lbl] = vals
        print(f"{lbl:<20}  {'  '.join(fmt(v) for v in vals)}")

print("══════════════════════════════════════════════════════")
print(f"\nGráficas guardadas en: {PLOTS_DIR}/")
for f in sorted(os.listdir(PLOTS_DIR)):
    print(f"  {f}")


# ══════════════════════════════════════════════════════════════════════════════
# COMPILACIÓN LaTeX → PDF
# ══════════════════════════════════════════════════════════════════════════════

def _fmt_tex(val):
    """Formatea un ciclo como string LaTeX (e.g. 57.5)."""
    if np.isnan(val):
        return "---"
    return f"{val/1e9:.1f}"


def build_latex_table(table_data, med_v3_static):
    """
    Construye el cuerpo de la tabla LaTeX (filas de midrule hacia abajo)
    con los valores reales de table_data.
    """
    # Orden de filas y su clave en table_data
    row_specs = [
        (r"v1 \texttt{-O0}",       "v1 -O0"),
        (r"v1 \texttt{-O3}",       "v1 -O3"),
        (r"v2 \texttt{-O0}",       "v2 -O0"),
        (r"v2 \texttt{-O3}",       "v2 -O3"),
        (r"v3 \texttt{-O3} (SIMD)", "v4 -O3 (SIMD)"),   # v3 SIMD en la memoria
        (r"v4 \texttt{-O3} (1T)",   "v4 -O3 (SIMD)"),   # placeholder si no hay v4 separado
    ]

    lines = []
    for tex_label, key in row_specs:
        vals = table_data.get(key, [np.nan, np.nan, np.nan])
        cols = " & ".join(_fmt_tex(v) for v in vals)
        lines.append(f"{tex_label} & {cols} \\\\")

    # Filas de v4 OpenMP con 1 hilo y máx hilos
    if not med_v3_static.empty:
        max_t = int(med_v3_static["threads"].max())
        for T in [1, max_t]:
            key  = f"v3 -O3 (OMP) {T}h"
            vals = table_data.get(key, [np.nan, np.nan, np.nan])
            cols = " & ".join(_fmt_tex(v) for v in vals)
            tex_label = rf"v4 \texttt{{-O3}} ({T}T)"
            lines.append(f"{tex_label} & {cols} \\\\")

    return "\n".join(lines)


def patch_and_compile_latex():
    """
    Copia Memoria.tex → Memoria_build.tex, reemplaza la tabla de medianas
    con valores reales y compila con lualatex (2 pasadas).
    """
    if not os.path.isfile(TEX_FILE):
        print(f"\n[AVISO] No se encontró {TEX_FILE}; saltando compilación LaTeX.")
        return

    # Verificar que lualatex está disponible
    latex_bin = shutil.which("lualatex") or shutil.which("pdflatex")
    if latex_bin is None:
        print("\n[AVISO] lualatex/pdflatex no encontrado en PATH; saltando compilación.")
        print("  Instala TeX Live o MiKTeX y asegúrate de que esté en el PATH.")
        return

    compiler = os.path.basename(latex_bin)
    print(f"\n── Parcheando {TEX_FILE} → {TEX_WORK} ──")

    with open(TEX_FILE, encoding="utf-8") as f:
        src = f.read()

    # ── Parche 1: tabla de medianas ──────────────────────────────────────────
    # Localiza el bloque entre \midrule y \bottomrule dentro de tab:medianas
    # y lo sustituye por datos reales.
    new_rows = build_latex_table(table_data, med_v3_static)

    # Patrón: desde \midrule hasta \bottomrule (dentro de la tabla tab:medianas)
    tabla_pattern = re.compile(
        r'(\\label\{tab:medianas\}.*?\\midrule\n)'   # cabecera hasta \midrule
        r'(.*?)'                                       # filas actuales (sustituir)
        r'(\\bottomrule)',                             # cierre
        re.DOTALL
    )
    new_src, n_subs = tabla_pattern.subn(
        lambda m: m.group(1) + new_rows + "\n" + m.group(3),
        src
    )
    if n_subs == 0:
        print("  [AVISO] No se localizó la tabla tab:medianas; se usará sin parche.")
        new_src = src
    else:
        print(f"  Tabla de medianas actualizada con datos reales ({n_subs} sustitución).")

    # ── Parche 2: verificar rutas de imágenes ────────────────────────────────
    # Comprueba que todos los ficheros referenciados existen
    missing_imgs = []
    for key, path in IMG_NAMES.items():
        if not os.path.isfile(path):
            missing_imgs.append(path)
    if missing_imgs:
        print("  [AVISO] Faltan estas imágenes (la compilación puede fallar):")
        for p in missing_imgs:
            print(f"    {p}")

    with open(TEX_WORK, "w", encoding="utf-8") as f:
        f.write(new_src)

    # ── Compilación (2 pasadas para referencias cruzadas) ────────────────────
    compile_flags = [
        latex_bin,
        "-interaction=nonstopmode",
        "-halt-on-error",
        TEX_WORK,
    ]

    for pasada in (1, 2):
        print(f"── Compilando con {compiler}: pasada {pasada}/2 ──")
        result = subprocess.run(
            compile_flags,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"  [ERROR] {compiler} falló en la pasada {pasada}.")
            # Mostrar las últimas 30 líneas del log para diagnóstico
            log_lines = result.stdout.splitlines()
            print("  Últimas líneas del log:")
            for line in log_lines[-30:]:
                print(f"    {line}")
            print(f"  Comprueba el fichero {TEX_WORK} manualmente.")
            return

    # ── Renombrar PDF generado ────────────────────────────────────────────────
    generated_pdf = TEX_WORK.replace(".tex", ".pdf")
    if os.path.isfile(generated_pdf):
        shutil.move(generated_pdf, PDF_OUT)
        print(f"\n✓ PDF generado correctamente: {PDF_OUT}")
    else:
        print(f"\n[AVISO] No se encontró el PDF generado ({generated_pdf}).")

    # ── Limpiar auxiliares LaTeX ──────────────────────────────────────────────
    base = TEX_WORK.replace(".tex", "")
    for ext in (".aux", ".log", ".out", ".toc", ".lof", ".lot",
                ".bbl", ".blg", ".fls", ".fdb_latexmk"):
        aux = base + ext
        if os.path.isfile(aux):
            os.remove(aux)
    # Mantener TEX_WORK por si el usuario quiere inspeccionarlo
    print(f"  (Fichero .tex parcheado conservado como {TEX_WORK})")


patch_and_compile_latex()