#!/bin/bash
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=512M
#SBATCH -t 01:30:00
#SBATCH --job-name jacobi_v1v2
#SBATCH --output=logs/v1v2_%j.out
#SBATCH --error=logs/v1v2_%j.err

# ==============================================================
# Experimentos v1 y v2 (versiones secuenciales)
#
# Tamaños : n=1250, 2000, 3200
# Flags   : -O0 y -O3
# Reps    : 10 por configuración → mediana calculada en Python
#
# Salida  : resultados/v1_O0.txt  v1_O3.txt  v2_O0.txt  v2_O3.txt
# Formato : <version> <n> <iter> <norm2> <ciclos>
# ==============================================================

echo "===== Jacobi v1/v2: experimentos secuenciales ====="
echo "Nodo : $(hostname)"
echo "Fecha: $(date)"

# ── Directorio de trabajo: donde están los .c y el makefile ────
WORKDIR="$HOME/ArqComp-Practica2/"
cd "$WORKDIR" || { echo "ERROR: no existe $WORKDIR"; exit 1; }

mkdir -p resultados logs

SIZES="1250 2000 3200"

# ══════════════════════════════════════════════════════════════
# BLOQUE -O0
# ══════════════════════════════════════════════════════════════
echo ""
echo "── Compilando v1 y v2 con -O0 ──"
gcc -O0 -Wall -o v1 v1.c -lm || { echo "ERROR compilando v1 -O0"; exit 1; }
gcc -O0 -Wall -o v2 v2.c -lm || { echo "ERROR compilando v2 -O0"; exit 1; }

echo "── Ejecutando v1 -O0 ──"
for N in $SIZES; do
    for i in $(seq 1 10); do
        ./v1 "$N" >> resultados/v1_O0.txt
    done
done

echo "── Ejecutando v2 -O0 ──"
for N in $SIZES; do
    for i in $(seq 1 10); do
        ./v2 "$N" >> resultados/v2_O0.txt
    done
done

# ══════════════════════════════════════════════════════════════
# BLOQUE -O3
# Recompilamos encima de los mismos binarios (no hace falta
# borrarlos antes; simplemente se sobreescriben).
# ══════════════════════════════════════════════════════════════
echo ""
echo "── Compilando v1 y v2 con -O3 ──"
gcc -O3 -Wall -o v1 v1.c -lm || { echo "ERROR compilando v1 -O3"; exit 1; }
gcc -O3 -Wall -o v2 v2.c -lm || { echo "ERROR compilando v2 -O3"; exit 1; }

echo "── Ejecutando v1 -O3 ──"
for N in $SIZES; do
    for i in $(seq 1 10); do
        ./v1 "$N" >> resultados/v1_O3.txt
    done
done

echo "── Ejecutando v2 -O3 ──"
for N in $SIZES; do
    for i in $(seq 1 10); do
        ./v2 "$N" >> resultados/v2_O3.txt
    done
done

echo ""
echo "===== v1/v2 finalizados ====="
echo "Ficheros generados:"
for f in resultados/v1_O0.txt resultados/v1_O3.txt \
          resultados/v2_O0.txt resultados/v2_O3.txt; do
    lines=$(wc -l < "$f" 2>/dev/null || echo 0)
    echo "  $f  ($lines líneas)"
done