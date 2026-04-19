#!/bin/bash
#SBATCH -n 1
#SBATCH -c 64
#SBATCH --mem=1G
#SBATCH -t 08:00:00
#SBATCH --job-name jacobi_v3
#SBATCH --output=logs/v3_%j.out
#SBATCH --error=logs/v3_%j.err

set -euo pipefail

# ==============================================================
# Experimentos v3 (OpenMP)
#
# Tamaños : n=1250, 2000, 3200
# Hilos   : 1, 2, 4, 8, 16, 32
# Reps    : 10 por configuracion
#
# Variantes:
#   - static   -> reduction + schedule(static)
#   - dynamic  -> reduction + schedule(dynamic,16)
#   - guided   -> reduction + schedule(guided)
#   - critical -> critical  + schedule(static)
#
# Salida  : resultados/v3_O3_static.txt
#           resultados/v3_O3_dynamic.txt
#           resultados/v3_O3_guided.txt
#           resultados/v3_O3_critical.txt
# Formato : v3 <n> <threads> <iter> <norm2> <ciclos>
# ==============================================================

SCRIPT_DIR="$SLURM_SUBMIT_DIR"
RESULTS_DIR="$SCRIPT_DIR/resultados"
LOGS_DIR="$SCRIPT_DIR/logs"
CC_BIN="${CC:-gcc}"
MAKE_BIN="${MAKE:-make}"

SIZES=(1250 2000 3200)
THREADS=(1 2 4 8 16 32)
REPS=10
EXPECTED=$(( ${#SIZES[@]} * ${#THREADS[@]} * REPS ))

mkdir -p "$RESULTS_DIR" "$LOGS_DIR"
cd "$SCRIPT_DIR"

echo "===== Jacobi v3 (OpenMP) ====="
echo "Directorio : $SCRIPT_DIR"
echo "Nodo       : $(hostname)"
echo "Fecha      : $(date)"
echo "CPUs       : $(nproc)"

compile_variant() {
    local output="$1"
    shift
    "$MAKE_BIN" clean >/dev/null 2>&1 || true
    "$MAKE_BIN" -B v3 CFLAGS="-O3 $*"
    mv v3 "$output"
}

run_variant() {
    local binary="$1"
    local outfile="$2"

    : > "$outfile"

    for n in "${SIZES[@]}"; do
        for t in "${THREADS[@]}"; do
            echo "   $binary  n=$n  hilos=$t"
            for _ in $(seq 1 "$REPS"); do
                "./$binary" "$n" "$t" >> "$outfile"
            done
        done
    done
}

summarize_file() {
    local file="$1"
    local lines
    lines=$(wc -l < "$file")
    if [ "$lines" -eq "$EXPECTED" ]; then
        echo "  OK    $file — $lines lineas"
    else
        echo "  AVISO $file — $lines lineas (esperadas $EXPECTED)"
        return 1
    fi
}

echo
echo "── Compilando v3_static [reduction + schedule(static)] ──"
compile_variant v3_static
echo "── Ejecutando v3_static ──"
run_variant v3_static "$RESULTS_DIR/v3_O3_static.txt"

echo
echo "── Compilando v3_dynamic [reduction + schedule(dynamic,16)] ──"
compile_variant v3_dynamic -DV3_SCHEDULE_DYNAMIC
echo "── Ejecutando v3_dynamic ──"
run_variant v3_dynamic "$RESULTS_DIR/v3_O3_dynamic.txt"

echo
echo "── Compilando v3_guided [reduction + schedule(guided)] ──"
compile_variant v3_guided -DV3_SCHEDULE_GUIDED
echo "── Ejecutando v3_guided ──"
run_variant v3_guided "$RESULTS_DIR/v3_O3_guided.txt"

echo
echo "── Compilando v3_critical [critical + schedule(static)] ──"
compile_variant v3_critical -DV3_USE_CRITICAL
echo "── Ejecutando v3_critical ──"
run_variant v3_critical "$RESULTS_DIR/v3_O3_critical.txt"

rm -f v3_static v3_dynamic v3_guided v3_critical

echo
echo "===== v3 finalizado ($(date)) ====="
echo "Lineas esperadas por fichero: $EXPECTED"

all_ok=1
for f in \
    "$RESULTS_DIR/v3_O3_static.txt" \
    "$RESULTS_DIR/v3_O3_dynamic.txt" \
    "$RESULTS_DIR/v3_O3_guided.txt" \
    "$RESULTS_DIR/v3_O3_critical.txt"; do
    if ! summarize_file "$f"; then
        all_ok=0
    fi
done

if [ "$all_ok" -eq 1 ]; then
    echo
    echo "Todos los ficheros de v3 se generaron correctamente."
else
    echo
    echo "Hay ficheros incompletos; revisa los logs del trabajo."
fi
