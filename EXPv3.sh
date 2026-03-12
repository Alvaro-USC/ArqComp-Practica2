#!/bin/bash
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=512M
#SBATCH -t 04:00:00
#SBATCH --job-name jacobi_v3
#SBATCH --output=logs/v3_%j.out
#SBATCH --error=logs/v3_%j.err

# ==============================================================
# Experimentos v3 (SIMD AVX256 + FMA)
#
# Tamaños : n=1250, 2000, 3200
# Flags   : -O0 y -O3  (el makefile añade -mavx2 -mfma)
# Reps    : 10 por configuración
#
# Salida  : resultados/v3_O0.txt  v3_O3.txt
# Formato : <version> <n> <iter> <norm2> <ciclos>
# ==============================================================

echo "===== Jacobi v3 (AVX256+FMA): experimentos SIMD ====="
echo "Nodo : $(hostname)"
echo "Fecha: $(date)"

WORKDIR="$HOME/ArqComp-Practica2/"
cd "$WORKDIR" || { echo "ERROR: no existe $WORKDIR"; exit 1; }

mkdir -p resultados logs

SIZES="1250 2000 3200"

# ── -O0 ────────────────────────────────────────────────────────
echo ""
echo "── Compilando v3 con -O0 -mavx2 -mfma ──"
gcc -O0 -mavx2 -mfma -Wall -o v3 v3.c -lm || { echo "ERROR compilando v3 -O0"; exit 1; }

echo "── Ejecutando v3 -O0 ──"
for N in $SIZES; do
    for i in $(seq 1 10); do
        ./v3 "$N" >> resultados/v3_O0.txt
    done
done

# ── -O3 ────────────────────────────────────────────────────────
echo ""
echo "── Compilando v3 con -O3 -mavx2 -mfma ──"
gcc -O3 -mavx2 -mfma -Wall -o v3 v3.c -lm || { echo "ERROR compilando v3 -O3"; exit 1; }

echo "── Ejecutando v3 -O3 ──"
for N in $SIZES; do
    for i in $(seq 1 10); do
        ./v3 "$N" >> resultados/v3_O3.txt
    done
done

echo ""
echo "===== v3 finalizado ====="
for f in resultados/v3_O0.txt resultados/v3_O3.txt; do
    lines=$(wc -l < "$f" 2>/dev/null || echo 0)
    echo "  $f  ($lines líneas)"
done