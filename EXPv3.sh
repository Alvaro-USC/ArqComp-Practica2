#!/bin/bash
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 00:30:00
#SBATCH --job-name jacobi_v3
#SBATCH --output=logs/v3_%j.out
#SBATCH --error=logs/v3_%j.err

# ==============================================================
# Script de experimentación para v3 (SIMD AVX256 + FMA)
#
# Experimentos:
#   · v3 compilado con -O0 y -O3
#     (el makefile añade -mavx2 -mfma automáticamente)
#   · Tamaños: n=1250, 2000, 3200
#   · 10 repeticiones por configuración
#
# Salida: resultados/v3_O0.txt, v3_O3.txt
#   Formato de cada línea: <version> <n> <iter> <norm2> <ciclos>
# ==============================================================

echo "===== Jacobi v3 (AVX256+FMA): experimentos SIMD ====="
echo "Nodo: $(hostname)"
echo "Fecha: $(date)"

mkdir -p resultados logs

SIZES="1250 2000 3200"

# ---------------------------------------------------------------
# Compilar v3 con -O0
# ---------------------------------------------------------------
echo ""
echo "--- Compilando v3 con -O0 ---"
make clean
make v3 CFLAGS="-O0"

echo ""
echo "--- Ejecutando v3 con -O0 ---"
for N in $SIZES; do
    for i in $(seq 1 10); do
        ./v3 $N >> resultados/v3_O0.txt
    done
done

# ---------------------------------------------------------------
# Compilar v3 con -O3
# ---------------------------------------------------------------
echo ""
echo "--- Compilando v3 con -O3 ---"
make clean
make v3 CFLAGS="-O3"

echo ""
echo "--- Ejecutando v3 con -O3 ---"
for N in $SIZES; do
    for i in $(seq 1 10); do
        ./v3 $N >> resultados/v3_O3.txt
    done
done

echo ""
echo "===== Experimentos v3 finalizados ====="
echo "Resultados en: resultados/v3_O0.txt  v3_O3.txt"