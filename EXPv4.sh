#!/bin/bash
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=1G
#SBATCH -t 08:00:00
#SBATCH --job-name jacobi_v4
#SBATCH --output=logs/v4_%j.out
#SBATCH --error=logs/v4_%j.err

set -euo pipefail

# ==============================================================
# Experimentos v4 (SIMD AVX256 + FMA)
#
# Tamaños : n=1250, 2000, 3200
# Flags   : -O0 y -O3
# Reps    : 10 por configuracion
#
# Salida  : resultados/v4_O0.txt, resultados/v4_O3.txt
# Formato : v4 <n> <iter> <norm2> <ciclos>
# ==============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/resultados"
LOGS_DIR="$SCRIPT_DIR/logs"
MAKE_BIN="${MAKE:-make}"

SIZES=(1250 2000 3200)
REPS=10
EXPECTED=$(( ${#SIZES[@]} * REPS ))

mkdir -p "$RESULTS_DIR" "$LOGS_DIR"
cd "$SCRIPT_DIR"

echo "===== Jacobi v4 (SIMD AVX256 + FMA) ====="
echo "Directorio : $SCRIPT_DIR"
echo "Nodo       : $(hostname)"
echo "Fecha      : $(date)"
echo "CPUs       : $(nproc)"

run_binary() {
    local outfile="$1"

    : > "$outfile"

    for n in "${SIZES[@]}"; do
        echo "   v4  n=$n"
        for _ in $(seq 1 "$REPS"); do
            ./v4 "$n" >> "$outfile"
        done
    done
}

compile_v4() {
    local opt="$1"
    "$MAKE_BIN" clean >/dev/null 2>&1 || true
    "$MAKE_BIN" -B v4 CFLAGS="$opt"
}

echo
echo "── Compilando v4 con -O0 ──"
compile_v4 "-O0"

echo "── Ejecutando v4 -O0 ──"
run_binary "$RESULTS_DIR/v4_O0.txt"

echo
echo "── Compilando v4 con -O3 ──"
compile_v4 "-O3"

echo "── Ejecutando v4 -O3 ──"
run_binary "$RESULTS_DIR/v4_O3.txt"

"$MAKE_BIN" clean >/dev/null 2>&1 || true

echo
echo "===== v4 finalizado ($(date)) ====="
echo "Lineas esperadas por fichero: $EXPECTED"

all_ok=1
for f in "$RESULTS_DIR/v4_O0.txt" "$RESULTS_DIR/v4_O3.txt"; do
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
    echo "Todos los ficheros de v4 se generaron correctamente."
else
    echo
    echo "Hay ficheros incompletos; revisa los logs del trabajo."
fi
