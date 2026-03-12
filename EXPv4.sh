#!/bin/bash
#SBATCH -n 1
#SBATCH -c 64
#SBATCH --mem=8G
#SBATCH -t 01:30:00
#SBATCH --job-name jacobi_v4
#SBATCH --output=logs/v4_%j.out
#SBATCH --error=logs/v4_%j.err

# ==============================================================
# Script de experimentación para v4 (OpenMP)
#
# Experimentos (requisitos del enunciado):
#   1. Variación del número de hilos: 1, 2, 4, 8, 16, 32 hilos
#   2. Schedulings: static, dynamic, guided
#      → Para cambiar el scheduling, recompilar con la macro OMP_SCHED
#        o editar v4.c y recompilar.  Aquí se hacen 3 compilaciones.
#   3. Comparación critical vs atomic (reducción de norm2)
#      → Se proporcionan dos versiones: v4_crit y v4_atm (ver abajo)
#   4. Búsqueda de la mejor combinación
#
#   Tamaños: n=1250, 2000, 3200
#   10 repeticiones por configuración
#
# Salida en resultados/v4_*.txt
#   Formato: v4 <n> <threads> <iter> <norm2> <ciclos>
# ==============================================================

echo "===== Jacobi v4 (OpenMP): experimentos paralelos ====="
echo "Nodo: $(hostname)"
echo "Fecha: $(date)"
echo "CPUs disponibles: $(nproc)"

mkdir -p resultados logs

SIZES="1250 2000 3200"
THREADS="1 2 4 8 16 32"

# ---------------------------------------------------------------
# EXPERIMENTO 1 y 2: variación de hilos y scheduling
# Compilamos v4 con -O3 y schedule(static) (por defecto en v4.c)
# ---------------------------------------------------------------
echo ""
echo "--- Compilando v4 con -O3 (schedule static, atomic) ---"
make clean
make v4 CFLAGS="-O3"

echo ""
echo "--- Experimento: hilos + schedule static ---"
for N in $SIZES; do
    for T in $THREADS; do
        for i in $(seq 1 10); do
            ./v4 $N $T >> resultados/v4_O3_static.txt
        done
    done
done

# ---------------------------------------------------------------
# EXPERIMENTO 2b: schedule dynamic
# Requiere editar v4.c para cambiar schedule(static) por schedule(dynamic,16)
# Hacemos una copia con sed para no modificar v4.c original
# ---------------------------------------------------------------
echo ""
echo "--- Compilando v4 con -O3 (schedule dynamic) ---"
sed 's/schedule(static)/schedule(dynamic,16)/g' v4.c > v4_dyn.c
$(CC) -O3 -fopenmp -Wall -o v4_dyn v4_dyn.c -lm

echo ""
echo "--- Experimento: hilos + schedule dynamic ---"
for N in $SIZES; do
    for T in $THREADS; do
        for i in $(seq 1 10); do
            ./v4_dyn $N $T >> resultados/v4_O3_dynamic.txt
        done
    done
done

# ---------------------------------------------------------------
# EXPERIMENTO 2c: schedule guided
# ---------------------------------------------------------------
echo ""
echo "--- Compilando v4 con -O3 (schedule guided) ---"
sed 's/schedule(static)/schedule(guided)/g' v4.c > v4_gui.c
$(CC) -O3 -fopenmp -Wall -o v4_gui v4_gui.c -lm

echo ""
echo "--- Experimento: hilos + schedule guided ---"
for N in $SIZES; do
    for T in $THREADS; do
        for i in $(seq 1 10); do
            ./v4_gui $N $T >> resultados/v4_O3_guided.txt
        done
    done
done

# ---------------------------------------------------------------
# EXPERIMENTO 3: critical vs atomic (reducción de norm2)
# Versión critical: descomentar el bloque critical en v4.c
# ---------------------------------------------------------------
echo ""
echo "--- Compilando v4 con -O3 (reducción critical) ---"
# Activar critical y desactivar atomic mediante sed
sed -e 's|/\* #pragma omp critical|#pragma omp critical|g' \
    -e 's|#pragma omp atomic|/* #pragma omp atomic (desactivado)|g' \
    v4.c > v4_crit.c
$(CC) -O3 -fopenmp -Wall -o v4_crit v4_crit.c -lm 2>/dev/null || \
    echo "AVISO: v4_crit no compiló (editar manualmente v4.c para el experimento critical)"

if [ -f ./v4_crit ]; then
    echo ""
    echo "--- Experimento: reducción critical ---"
    for N in $SIZES; do
        for T in $THREADS; do
            for i in $(seq 1 10); do
                ./v4_crit $N $T >> resultados/v4_O3_critical.txt
            done
        done
    done
fi

# ---------------------------------------------------------------
# Limpiar binarios temporales
# ---------------------------------------------------------------
rm -f v4_dyn v4_dyn.c v4_gui v4_gui.c v4_crit v4_crit.c

echo ""
echo "===== Experimentos v4 finalizados ====="
echo "Resultados en:"
echo "  resultados/v4_O3_static.txt"
echo "  resultados/v4_O3_dynamic.txt"
echo "  resultados/v4_O3_guided.txt"
echo "  resultados/v4_O3_critical.txt  (si compiló)"