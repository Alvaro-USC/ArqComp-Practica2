#!/bin/bash
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 01:00:00
#SBATCH --job-name jacobi_v1v2
#SBATCH --output=logs/v1v2_%j.out
#SBATCH --error=logs/v1v2_%j.err

# ==============================================================
# Script de experimentación para v1 y v2 (versiones secuenciales)
#
# Experimentos:
#   · v1 compilado con -O0 y -O3  (referencia base)
#   · v2 compilado con -O0 y -O3  (optimizaciones caché)
#   · Tamaños: n=1250 (pequeño), 2000 (mediano), 3200 (grande)
#   · 10 repeticiones por configuración → se toma la mediana en Python
#
# Salida: resultados/v1_O0.txt, v1_O3.txt, v2_O0.txt, v2_O3.txt
#   Formato de cada línea: <version> <n> <iter> <norm2> <ciclos>
# ==============================================================

echo "===== Jacobi v1 y v2: experimentos secuenciales ====="
echo "Nodo: $(hostname)"
echo "Fecha: $(date)"

# Crear directorios de salida
mkdir -p resultados logs

# Tamaños de problema
SIZES="1250 2000 3200"

# ---------------------------------------------------------------
# Compilar v1 y v2 con -O0
# ---------------------------------------------------------------
echo ""
echo "--- Compilando con -O0 ---"
make clean
make v1 v2 CFLAGS="-O0"

echo ""
echo "--- Ejecutando v1 con -O0 (10 repeticiones x 3 tamaños) ---"
for N in $SIZES; do
    for i in $(seq 1 10); do
        ./v1 $N >> resultados/v1_O0.txt
    done
done

echo ""
echo "--- Ejecutando v2 con -O0 (10 repeticiones x 3 tamaños) ---"
for N in $SIZES; do
    for i in $(seq 1 10); do
        ./v2 $N >> resultados/v2_O0.txt
    done
done

# ---------------------------------------------------------------
# Compilar v1 y v2 con -O3
# ---------------------------------------------------------------
echo ""
echo "--- Compilando con -O3 ---"
make clean
make v1 v2 CFLAGS="-O3"

echo ""
echo "--- Ejecutando v1 con -O3 (10 repeticiones x 3 tamaños) ---"
for N in $SIZES; do
    for i in $(seq 1 10); do
        ./v1 $N >> resultados/v1_O3.txt
    done
done

echo ""
echo "--- Ejecutando v2 con -O3 (10 repeticiones x 3 tamaños) ---"
for N in $SIZES; do
    for i in $(seq 1 10); do
        ./v2 $N >> resultados/v2_O3.txt
    done
done

echo ""
echo "===== Experimentos v1/v2 finalizados ====="
echo "Resultados en: resultados/v1_O0.txt  v1_O3.txt  v2_O0.txt  v2_O3.txt"