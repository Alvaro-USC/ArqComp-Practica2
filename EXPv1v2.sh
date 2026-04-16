#!/bin/bash
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=1G
#SBATCH -t 08:00:00
#SBATCH --job-name jacobi_v1v2
#SBATCH --output=logs/v1v2_%j.out
#SBATCH --error=logs/v1v2_%j.err

set -euo pipefail

# ==============================================================
# Experimentos v1 y v2 (versiones secuenciales)
#
# Tamaños : n=1250, 2000, 3200
# Flags   : -O0 y -O3
# Reps    : 10 por configuracion
#
# Salida  : resultados/v1_O0.txt, resultados/v1_O3.txt
#           resultados/v2_O0.txt, resultados/v2_O3.txt
# Formato : <version> <n> <iter> <norm2> <ciclos>
# ==============================================================

SCRIPT_DIR="$SLURM_SUBMIT_DIR"
RESULTS_DIR="$SCRIPT_DIR/resultados"
LOGS_DIR="$SCRIPT_DIR/logs"
MAKE_BIN="${MAKE:-make}"

SIZES=(1250 2000 3200)
REPS=10
EXPECTED=$(( ${#SIZES[@]} * REPS ))

mkdir -p "$RESULTS_DIR" "$LOGS_DIR"
cd "$SCRIPT_DIR"

echo "===== Jacobi v1/v2: experimentos secuenciales ====="
echo "Directorio : $SCRIPT_DIR"
echo "Nodo       : $(hostname)"
echo "Fecha      : $(date)"
echo "CPUs       : $(nproc)"

run_binary() {
    local binary="$1"
    local outfile="$2"

    : > "$outfile"

    for n in "${SIZES[@]}"; do
        echo "   $binary  n=$n"
        for _ in $(seq 1 "$REPS"); do
            "./$binary" "$n" >> "$outfile"
        done
    done
}

compile_pair() {
    local opt="$1"
    "$MAKE_BIN" clean >/dev/null 2>&1 || true
    "$MAKE_BIN" -B v1 v2 CFLAGS="$opt"
}

echo
echo "── Compilando v1 y v2 con -O0 ──"
compile_pair "-O0"

echo "── Ejecutando v1 -O0 ──"
run_binary v1 "$RESULTS_DIR/v1_O0.txt"

echo "── Ejecutando v2 -O0 ──"
run_binary v2 "$RESULTS_DIR/v2_O0.txt"

echo
echo "── Compilando v1 y v2 con -O3 ──"
compile_pair "-O3"

echo "── Ejecutando v1 -O3 ──"
run_binary v1 "$RESULTS_DIR/v1_O3.txt"

echo "── Ejecutando v2 -O3 ──"
run_binary v2 "$RESULTS_DIR/v2_O3.txt"

"$MAKE_BIN" clean >/dev/null 2>&1 || true

echo
echo "===== v1/v2 finalizados ($(date)) ====="
echo "Lineas esperadas por fichero: $EXPECTED"

all_ok=1
for f in \
    "$RESULTS_DIR/v1_O0.txt" \
    "$RESULTS_DIR/v1_O3.txt" \
    "$RESULTS_DIR/v2_O0.txt" \
    "$RESULTS_DIR/v2_O3.txt"; do
    lines=$(wc -l < "$f")
    if [ "$lines" -eq "$EXPECTED" ]; then
        echo "  OK    $f — $lines lineas"
    else
        echo "  AVISO $f — $lines lineas (esperadas $EXPECTED)"
        all_ok=0
    fi
done

if [ "$all_ok" -eq 1 ]; then
    echo
    echo "Todos los ficheros de v1/v2 se generaron correctamente."
else
    echo
    echo "Hay ficheros incompletos; revisa los logs del trabajo."
fi
