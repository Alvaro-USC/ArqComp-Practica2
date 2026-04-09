#!/bin/bash
#SBATCH -n 1
#SBATCH -c 64
#SBATCH --mem=1G
#SBATCH -t 08:00:00
#SBATCH --job-name jacobi_v3
#SBATCH --output=logs/v3_%j.out
#SBATCH --error=logs/v3_%j.err

# ==============================================================
# Experimentos v3 (OpenMP)
#
# Tamaños : n=1250, 2000, 3200
# Hilos   : 1 2 4 8 16 32
# Variantes:
#   · static   — reduction(+:norm2) + schedule(static)    [v3.c]
#   · dynamic  — reduction(+:norm2) + schedule(dynamic,16)
#   · guided   — reduction(+:norm2) + schedule(guided)
#   · critical — critical en lugar de reduction
# Reps    : 10 por configuración
#
# Salida  : resultados/v3_O3_static.txt
#           resultados/v3_O3_dynamic.txt
#           resultados/v3_O3_guided.txt
#           resultados/v3_O3_critical.txt
# Formato : v3 <n> <threads> <iter> <norm2> <ciclos>
# ==============================================================

echo "===== Jacobi v3 (OpenMP): experimentos paralelos ====="
echo "Nodo : $(hostname)"
echo "Fecha: $(date)"
echo "CPUs : $(nproc)"

WORKDIR="$HOME/ArqComp-Practica2/"
cd "$WORKDIR" || { echo "ERROR: no existe $WORKDIR"; exit 1; }

mkdir -p resultados logs

SIZES="1250 2000 3200"
THREADS="1 2 4 8 16 32"
REPS=10

# ── Función auxiliar: ejecuta un binario REPS veces para todos los tamaños y hilos
run_experiments() {
    local binary="$1"
    local outfile="$2"
    for N in $SIZES; do
        for T in $THREADS; do
            for i in $(seq 1 $REPS); do
                ./"$binary" "$N" "$T" >> "$outfile"
            done
        done
    done
}

# ══════════════════════════════════════════════════════════════
# VARIANTE 1: reduction(+:norm2) + schedule(static)
# v3.c ya tiene esta configuración, se compila directamente.
# ══════════════════════════════════════════════════════════════
echo ""
echo "── Compilando v3_static  [reduction + schedule(static)] ──"
gcc -O3 -fopenmp -Wall -o v3_static v3.c -lm \
    || { echo "ERROR compilando v3_static"; exit 1; }

echo "── Ejecutando v3_static ──"
run_experiments v3_static resultados/v3_O3_static.txt

# ══════════════════════════════════════════════════════════════
# VARIANTE 2: reduction(+:norm2) + schedule(dynamic,16)
# sed solo toca la cadena "schedule(static)" dentro de las
# directivas omp for; la cláusula reduction permanece intacta.
# Hay dos omp for en v3.c (cómputo y copia); ambos pasan a dynamic,
# lo cual es coherente para la comparación de schedules.
# ══════════════════════════════════════════════════════════════
echo ""
echo "── Generando v3_dyn.c  [reduction + schedule(dynamic,16)] ──"
sed 's/schedule(static)/schedule(dynamic,16)/g' v3.c > v3_dyn.c
echo "── Compilando v3_dyn ──"
gcc -O3 -fopenmp -Wall -o v3_dyn v3_dyn.c -lm \
    || { echo "ERROR compilando v3_dyn"; exit 1; }

echo "── Ejecutando v3_dyn ──"
run_experiments v3_dyn resultados/v3_O3_dynamic.txt

# ══════════════════════════════════════════════════════════════
# VARIANTE 3: reduction(+:norm2) + schedule(guided)
# ══════════════════════════════════════════════════════════════
echo ""
echo "── Generando v3_gui.c  [reduction + schedule(guided)] ──"
sed 's/schedule(static)/schedule(guided)/g' v3.c > v3_gui.c
echo "── Compilando v3_gui ──"
gcc -O3 -fopenmp -Wall -o v3_gui v3_gui.c -lm \
    || { echo "ERROR compilando v3_gui"; exit 1; }

echo "── Ejecutando v3_gui ──"
run_experiments v3_gui resultados/v3_O3_guided.txt

# ══════════════════════════════════════════════════════════════
# VARIANTE 4: critical en lugar de reduction
#
# Transformaciones sobre v3.c (cadenas exactas del fuente entregado):
#   1. Quitar "reduction(+:norm2)" del pragma omp for de cómputo.
#   2. Añadir "double local_norm2 = 0.0;" antes del bucle for.
#   3. Cambiar "norm2 += diff * diff;" por "local_norm2 += diff * diff;".
#   4. Insertar el bloque "#pragma omp critical" antes de la barrera.
#
# Se usa Python embebido para evitar la fragilidad de sed multilínea.
# ══════════════════════════════════════════════════════════════
echo ""
echo "── Generando v3_crit.c  [critical en lugar de reduction] ──"

python3 - <<'PYEOF'
with open("v3.c") as f:
    src = f.read()

# 1. Quitar "reduction(+:norm2)" del pragma omp for de cómputo.
#    El omp for de copia no tiene reduction, así que solo hay una ocurrencia.
src = src.replace(
    "#pragma omp for schedule(static) nowait reduction(+:norm2)",
    "#pragma omp for schedule(static) nowait"
)

# 2. Insertar "double local_norm2 = 0.0;" justo antes del for de cómputo.
#    Se ancla en el texto exacto que aparece en v3.c después del pragma.
src = src.replace(
    "            for (int i = 0; i < n; i++) {\n\n                const double *row = &a[i * n];",
    "            double local_norm2 = 0.0;\n            for (int i = 0; i < n; i++) {\n\n                const double *row = &a[i * n];",
    1   # solo primera ocurrencia: el for de cómputo
)

# 3. Cambiar la acumulación de norm2 dentro del bucle.
src = src.replace(
    "                norm2 += diff * diff;   /* reduction acumula por hilo */",
    "                local_norm2 += diff * diff;   /* acumula local; critical combinará al salir */"
)

# 4. Insertar el bloque critical entre el cierre del for y la barrera.
#    Ancla: comentario exacto de la barrera explícita en v3.c.
src = src.replace(
    "            /* Barrera explícita: garantiza que TODOS los hilos han",
    (
        "            /* Reducción con critical: un solo hilo a la vez puede\n"
        "             * actualizar norm2 -> serializa la reducción.\n"
        "             * Con muchos hilos genera alta contención (comparar\n"
        "             * tiempos con la variante reduction). */\n"
        "            #pragma omp critical\n"
        "            { norm2 += local_norm2; }\n\n"
        "            /* Barrera explícita: garantiza que TODOS los hilos han"
    ),
    1   # solo primera ocurrencia
)

with open("v3_crit.c", "w") as f:
    f.write(src)

print("v3_crit.c generado correctamente")
PYEOF

# Verificar que Python tuvo éxito y que las transformaciones son correctas
if [ $? -ne 0 ]; then
    echo "ERROR: fallo en la generación de v3_crit.c"
    exit 1
fi

python3 - <<'PYCHECK'
with open("v3_crit.c") as f:
    src = f.read()
parte_activa = src.split("VERSIÓN CON CRITICAL")[0]
pragma_con_reduction = [l for l in parte_activa.splitlines()
                        if "#pragma" in l and "reduction" in l]
ok = (
    len(pragma_con_reduction) == 0
    and "double local_norm2 = 0.0;" in parte_activa
    and "local_norm2 += diff * diff;" in parte_activa
    and "#pragma omp critical" in parte_activa
    and "{ norm2 += local_norm2; }" in parte_activa
)
if ok:
    print("  Verificación v3_crit.c: todas las transformaciones correctas")
else:
    print("  ERROR en v3_crit.c:")
    if pragma_con_reduction:
        print("    - pragma omp for todavía tiene reduction")
    if "double local_norm2 = 0.0;" not in parte_activa:
        print("    - local_norm2 no declarada")
    if "local_norm2 += diff * diff;" not in parte_activa:
        print("    - acumulación no sustituida")
    if "#pragma omp critical" not in parte_activa:
        print("    - bloque critical no insertado")
    if "{ norm2 += local_norm2; }" not in parte_activa:
        print("    - combinación en critical no encontrada")
    exit(1)
PYCHECK

if [ $? -ne 0 ]; then
    echo "ERROR: v3_crit.c no pasó la verificación — abortando"
    exit 1
fi

echo "── Compilando v3_crit ──"
gcc -O3 -fopenmp -Wall -o v3_crit v3_crit.c -lm \
    || { echo "ERROR compilando v3_crit — revisa v3_crit.c manualmente"; }

if [ -x ./v3_crit ]; then
    echo "── Ejecutando v3_crit ──"
    run_experiments v3_crit resultados/v3_O3_critical.txt
else
    echo "AVISO: v3_crit no se ejecutó (fallo de compilación)"
fi

# ── Limpiar binarios y fuentes temporales ─────────────────────
rm -f v3_static v3_dyn v3_dyn.c v3_gui v3_gui.c v3_crit v3_crit.c

echo ""
echo "===== v3 finalizado ====="
echo "Líneas esperadas por fichero: $((3 * 6 * REPS))  (3 tamaños x 6 hilos x ${REPS} reps)"
for f in resultados/v3_O3_static.txt  resultados/v3_O3_dynamic.txt \
          resultados/v3_O3_guided.txt  resultados/v3_O3_critical.txt; do
    lines=$(wc -l < "$f" 2>/dev/null || echo 0)
    echo "  $f  ($lines líneas)"
done