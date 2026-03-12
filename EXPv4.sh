#!/bin/bash
#SBATCH -n 1
#SBATCH -c 64
#SBATCH --mem=512M
#SBATCH -t 02:00:00
#SBATCH --job-name jacobi_v4
#SBATCH --output=logs/v4_%j.out
#SBATCH --error=logs/v4_%j.err

# ==============================================================
# Experimentos v4 (OpenMP)
#
# Tamaños : n=1250, 2000, 3200
# Hilos   : 1 2 4 8 16 32
# Variantes:
#   · static   (v4.c tal cual)
#   · dynamic  (sed sobre v4.c → v4_dyn.c)
#   · guided   (sed sobre v4.c → v4_gui.c)
#   · critical (sed sobre v4.c → v4_crit.c)
# Reps    : 10 por configuración
#
# Salida  : resultados/v4_O3_static.txt
#           resultados/v4_O3_dynamic.txt
#           resultados/v4_O3_guided.txt
#           resultados/v4_O3_critical.txt
# Formato : v4 <n> <threads> <iter> <norm2> <ciclos>
# ==============================================================

echo "===== Jacobi v4 (OpenMP): experimentos paralelos ====="
echo "Nodo : $(hostname)"
echo "Fecha: $(date)"
echo "CPUs : $(nproc)"

WORKDIR="$HOME/ArqComp-Practica2/ACP2"
cd "$WORKDIR" || { echo "ERROR: no existe $WORKDIR"; exit 1; }

mkdir -p resultados logs

SIZES="1250 2000 3200"
THREADS="1 2 4 8 16 32"

# ══════════════════════════════════════════════════════════════
# VARIANTE 1: schedule(static) + atomic  ← v4.c original
# ══════════════════════════════════════════════════════════════
echo ""
echo "── Compilando v4 -O3 schedule(static) + atomic ──"
gcc -O3 -fopenmp -Wall -o v4_static v4.c -lm \
    || { echo "ERROR compilando v4_static"; exit 1; }

echo "── Ejecutando v4_static ──"
for N in $SIZES; do
    for T in $THREADS; do
        for i in $(seq 1 10); do
            ./v4_static "$N" "$T" >> resultados/v4_O3_static.txt
        done
    done
done

# ══════════════════════════════════════════════════════════════
# VARIANTE 2: schedule(dynamic,16)
# Se genera v4_dyn.c sustituyendo schedule(static) por schedule(dynamic,16)
# ══════════════════════════════════════════════════════════════
echo ""
echo "── Generando y compilando v4_dyn (schedule dynamic) ──"
sed 's/schedule(static)/schedule(dynamic,16)/g' v4.c > v4_dyn.c
gcc -O3 -fopenmp -Wall -o v4_dyn v4_dyn.c -lm \
    || { echo "ERROR compilando v4_dyn"; exit 1; }

echo "── Ejecutando v4_dyn ──"
for N in $SIZES; do
    for T in $THREADS; do
        for i in $(seq 1 10); do
            ./v4_dyn "$N" "$T" >> resultados/v4_O3_dynamic.txt
        done
    done
done

# ══════════════════════════════════════════════════════════════
# VARIANTE 3: schedule(guided)
# ══════════════════════════════════════════════════════════════
echo ""
echo "── Generando y compilando v4_gui (schedule guided) ──"
sed 's/schedule(static)/schedule(guided)/g' v4.c > v4_gui.c
gcc -O3 -fopenmp -Wall -o v4_gui v4_gui.c -lm \
    || { echo "ERROR compilando v4_gui"; exit 1; }

echo "── Ejecutando v4_gui ──"
for N in $SIZES; do
    for T in $THREADS; do
        for i in $(seq 1 10); do
            ./v4_gui "$N" "$T" >> resultados/v4_O3_guided.txt
        done
    done
done

# ══════════════════════════════════════════════════════════════
# VARIANTE 4: critical en lugar de atomic
# En v4.c la reducción tiene este aspecto:
#   #pragma omp atomic
#   norm2 += local_norm2;
# La sustituimos por:
#   #pragma omp critical
#   { norm2 += local_norm2; }
# usando un bloque de sed multilínea.
# ══════════════════════════════════════════════════════════════
echo ""
echo "── Generando y compilando v4_crit (reducción critical) ──"
# Reemplaza las dos líneas "atomic + norm2 +=" por el bloque critical
sed '/^.*#pragma omp atomic/{
    N
    s/.*#pragma omp atomic\n.*norm2 += local_norm2;/            #pragma omp critical\n            { norm2 += local_norm2; }/
}' v4.c > v4_crit.c

gcc -O3 -fopenmp -Wall -o v4_crit v4_crit.c -lm \
    || { echo "ERROR compilando v4_crit — revisar v4_crit.c manualmente"; }

if [ -x ./v4_crit ]; then
    echo "── Ejecutando v4_crit ──"
    for N in $SIZES; do
        for T in $THREADS; do
            for i in $(seq 1 10); do
                ./v4_crit "$N" "$T" >> resultados/v4_O3_critical.txt
            done
        done
    done
else
    echo "AVISO: v4_crit no se ejecutó (fallo de compilación)"
fi

# ── Limpiar ficheros temporales ────────────────────────────────
rm -f v4_dyn v4_dyn.c v4_gui v4_gui.c v4_crit v4_crit.c v4_static

echo ""
echo "===== v4 finalizado ====="
for f in resultados/v4_O3_static.txt  resultados/v4_O3_dynamic.txt \
          resultados/v4_O3_guided.txt  resultados/v4_O3_critical.txt; do
    lines=$(wc -l < "$f" 2>/dev/null || echo 0)
    echo "  $f  ($lines líneas)"
done