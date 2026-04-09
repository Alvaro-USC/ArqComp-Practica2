#!/bin/bash
#SBATCH -n 1
#SBATCH -c 64
#SBATCH --mem=1G
#SBATCH -t 08:00:00
#SBATCH --job-name jacobi_v4
#SBATCH --output=logs/v4_%j.out
#SBATCH --error=logs/v4_%j.err

# ==============================================================
# Experimentos v4 (SIMD AVX256 + FMA)
#
# Tamaños : n=1250, 2000, 3200
# Flags   : -O0 y -O3  (junto con -mavx2 -mfma)
# Reps    : 10 por configuración → mediana calculada en Python
#
# Salida  : resultados/v4_O0.txt  v4_O3.txt
# Formato : v4 <n> <iter> <norm2> <ciclos>
#
# Líneas esperadas por fichero: 30  (3 tamaños × 10 reps)
# Tiempo estimado worst-case  : ~6 h  (v4 -O0 n=3200 × 10 reps)
# ==============================================================

echo "===== Jacobi v4 (AVX256+FMA): experimentos SIMD ====="
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
for f in resultados/v4_O0.txt resultados/v4_O3.txt; do
    > "$f"
done

# ══════════════════════════════════════════════════════════════
# BLOQUE -O0
# A -O0 los intrínsecos AVX se emiten igualmente (son llamadas a
# funciones intrínsecas, no dependen del nivel de optimización),
# pero el compilador no vectorizará automáticamente nada extra.
# ══════════════════════════════════════════════════════════════
echo ""
echo "── Compilando v4 con -O0 -mavx2 -mfma ──"

# Se borra el binario anterior para evitar que un fallo de compilación
# silencioso deje el binario equivocado ejecutándose en el bloque -O3.
rm -f v4

gcc -O0 -mavx2 -mfma -Wall -I. -o v4 v4.c -lm \
    || { echo "ERROR compilando v4 -O0"; exit 1; }

echo "── Ejecutando v4 -O0 ──"
for N in $SIZES; do
    echo "   n=$N"
    for i in $(seq 1 $REPS); do
        ./v4 "$N" >> resultados/v4_O0.txt
    done
done

# ══════════════════════════════════════════════════════════════
# BLOQUE -O3
# Con -O3 el compilador puede reorganizar y vectorizar el código
# escalar adicional (prólogos/epílogos). Los intrínsecos AVX
# explícitos se mantienen intactos.
# ══════════════════════════════════════════════════════════════
echo ""
echo "── Compilando v4 con -O3 -mavx2 -mfma ──"

rm -f v4

gcc -O3 -mavx2 -mfma -Wall -I. -o v4 v4.c -lm \
    || { echo "ERROR compilando v4 -O3"; exit 1; }

echo "── Ejecutando v4 -O3 ──"
for N in $SIZES; do
    echo "   n=$N"
    for i in $(seq 1 $REPS); do
        ./v4 "$N" >> resultados/v4_O3.txt
    done
done

# ── Limpiar binario ────────────────────────────────────────────
rm -f v4

# ── Resumen con verificación de líneas esperadas ───────────────
echo ""
echo "===== v4 finalizado  ($(date)) ====="
echo "Líneas esperadas por fichero: $EXPECTED  (3 tamaños × $REPS reps)"
all_ok=1
for f in resultados/v4_O0.txt resultados/v4_O3.txt; do
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
    echo "Revisar logs/v4_${SLURM_JOB_ID}.err para errores de ejecución."
fi
