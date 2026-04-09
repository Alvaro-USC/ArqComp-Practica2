#!/bin/bash
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=1G
#SBATCH -t 08:00:00
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
#
# Líneas esperadas por fichero: 30  (3 tamaños × 10 reps)
# Tiempo estimado worst-case  : ~10 h  (v1 -O0 n=3200 × 10 reps)
# ==============================================================

echo "===== Jacobi v1/v2: experimentos secuenciales ====="
echo "Nodo : $(hostname)"
echo "Fecha: $(date)"
echo "CPUs : $(nproc)"

# ── Directorio de trabajo: donde están los .c, makefile y counter.h ──
WORKDIR="$HOME/ArqComp-Practica2/"
cd "$WORKDIR" || { echo "ERROR: no existe $WORKDIR"; exit 1; }

mkdir -p resultados logs

SIZES="1250 2000 3200"
REPS=10
EXPECTED=$((3 * REPS))   # 30 líneas por fichero

# ── Inicializar ficheros de resultados a vacío ─────────────────
# Si el job se relanza, los ficheros se sobreescriben en lugar
# de acumular resultados de ejecuciones anteriores.
for f in resultados/v1_O0.txt resultados/v1_O3.txt \
          resultados/v2_O0.txt resultados/v2_O3.txt; do
    > "$f"
done

# ══════════════════════════════════════════════════════════════
# BLOQUE -O0
# ══════════════════════════════════════════════════════════════
echo ""
echo "── Compilando v1 y v2 con -O0 ──"

# Se borran los binarios anteriores antes de compilar para evitar
# que un fallo silencioso deje el binario equivocado en disco.
rm -f v1 v2

gcc -O0 -Wall -I. -o v1 v1.c -lm \
    || { echo "ERROR compilando v1 -O0"; exit 1; }
gcc -O0 -Wall -I. -o v2 v2.c -lm \
    || { echo "ERROR compilando v2 -O0"; exit 1; }

echo "── Ejecutando v1 -O0 ──"
for N in $SIZES; do
    echo "   n=$N"
    for i in $(seq 1 $REPS); do
        ./v1 "$N" >> resultados/v1_O0.txt
    done
done

echo "── Ejecutando v2 -O0 ──"
for N in $SIZES; do
    echo "   n=$N"
    for i in $(seq 1 $REPS); do
        ./v2 "$N" >> resultados/v2_O0.txt
    done
done

# ══════════════════════════════════════════════════════════════
# BLOQUE -O3
# Borramos los binarios antes de recompilar para garantizar que
# las ejecuciones usan exactamente el binario recién generado.
# ══════════════════════════════════════════════════════════════
echo ""
echo "── Compilando v1 y v2 con -O3 ──"

rm -f v1 v2

gcc -O3 -Wall -I. -o v1 v1.c -lm \
    || { echo "ERROR compilando v1 -O3"; exit 1; }
gcc -O3 -Wall -I. -o v2 v2.c -lm \
    || { echo "ERROR compilando v2 -O3"; exit 1; }

echo "── Ejecutando v1 -O3 ──"
for N in $SIZES; do
    echo "   n=$N"
    for i in $(seq 1 $REPS); do
        ./v1 "$N" >> resultados/v1_O3.txt
    done
done

echo "── Ejecutando v2 -O3 ──"
for N in $SIZES; do
    echo "   n=$N"
    for i in $(seq 1 $REPS); do
        ./v2 "$N" >> resultados/v2_O3.txt
    done
done

# ── Limpiar binarios ───────────────────────────────────────────
rm -f v1 v2

# ── Resumen con verificación de líneas esperadas ───────────────
echo ""
echo "===== v1/v2 finalizados  ($(date)) ====="
echo "Líneas esperadas por fichero: $EXPECTED  (${#SIZES[@]} tamaños × $REPS reps)"
all_ok=1
for f in resultados/v1_O0.txt resultados/v1_O3.txt \
          resultados/v2_O0.txt resultados/v2_O3.txt; do
    lines=$(wc -l < "$f" 2>/dev/null || echo 0)
    if [ "$lines" -lt "$EXPECTED" ]; then
        echo "  AVISO: $f — $lines líneas (esperadas $EXPECTED)"
        all_ok=0
    else
        echo "  OK    $f — $lines líneas"
    fi
done

if [ "$all_ok" -eq 1 ]; then
    echo ""
    echo "Todos los ficheros completos. Listo para análisis."
else
    echo ""
    echo "AVISO: algún fichero tiene menos líneas de las esperadas."
    echo "Revisar logs/v1v2_${SLURM_JOB_ID}.err para errores de ejecución."
fi