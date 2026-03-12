/*
 * v3.c  –  Método de Jacobi: vectorización SIMD AVX256 + FMA
 *
 * Compilado por el makefile con:  -mavx2 -mfma
 *
 * Estrategia:
 *  · __m256d: 4 doubles por registro (256 bits).
 *  · _mm256_fmadd_pd: FMA acumula sigma en 4 carriles en paralelo.
 *  · División de lazos [0,i) y (i,n] sin rama condicional.
 *  · Prólogo/epílogo escalar para los elementos no alineados.
 *  · Reducción horizontal con _mm256_hadd_pd.
 *  · Memoria alineada a 32 B con aligned_alloc (C11/C17 estándar).
 *    Stride de fila = n_pad (múltiplo de 4) para garantizar alineación
 *    en cada acceso _mm256_load_pd.
 *  · NO se usan intrínsecos "_u" (no alineados) — restricción del enunciado.
 *  · NO se usa __attribute__((aligned)) ni _mm_malloc — restricción del enunciado.
 *
 * Salida: v3 <n> <iter> <norm2> <ciclos>
 *
 * Uso: ./v3 <n> [c]
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <immintrin.h>
#include "counter.h"

#define AVX_W 4   /* doubles por registro AVX256 */

int main(int argc, char *argv[]) {

    /* ---------- Comprobación del argumento obligatorio n ---------- */
    if (argc < 2) {
        fprintf(stderr, "Uso: %s <n> [c]\n", argv[0]);
        return 1;
    }
    int n = atoi(argv[1]);
    if (n <= 0) {
        fprintf(stderr, "Error: n debe ser un entero positivo.\n");
        return 1;
    }

    const double tol      = 1e-5;
    const int    max_iter = 15000;

    /* ---------- Stride con padding ----------
     * n_pad: múltiplo de AVX_W >= n.
     * Cada fila de la matriz ocupa n_pad doubles, de modo que con la
     * base alineada a 32 B, cada inicio de fila también lo está.
     */
    int n_pad = ((n + AVX_W - 1) / AVX_W) * AVX_W;

    /* ---------- Reserva dinámica alineada a 32 B (C11 aligned_alloc) ----------
     * aligned_alloc exige que el tamaño sea múltiplo de la alineación.
     */
    const size_t align = 32;
    size_t sz_a = ((size_t)n * n_pad * sizeof(double) + align - 1) / align * align;
    size_t sz_x = ((size_t)n_pad     * sizeof(double) + align - 1) / align * align;
    size_t sz_b = ((size_t)n         * sizeof(double) + align - 1) / align * align;

    double *a     = (double *)aligned_alloc(align, sz_a);
    double *b     = (double *)aligned_alloc(align, sz_b);
    double *x     = (double *)aligned_alloc(align, sz_x);
    double *x_new = (double *)aligned_alloc(align, sz_x);

    if (!a || !b || !x || !x_new) {
        fprintf(stderr, "Error: fallo en la reserva de memoria.\n");
        free(a); free(b); free(x); free(x_new);
        return 1;
    }

    /* Inicializar todo a 0: los elementos de padding no contaminan sumas */
    memset(a,     0, sz_a);
    memset(b,     0, sz_b);
    memset(x,     0, sz_x);
    memset(x_new, 0, sz_x);

    /* ---------- Inicialización con valores aleatorios (misma semilla) ---------- */
    srand(42);
    for (int i = 0; i < n; i++) {
        double row_sum = 0.0;
        for (int j = 0; j < n; j++) {
            a[i * n_pad + j] = (double)rand() / RAND_MAX;
            row_sum += a[i * n_pad + j];
        }
        a[i * n_pad + i] += row_sum;      /* dominancia diagonal */

        b[i] = (double)rand() / RAND_MAX;
        x[i] = 0.0;
    }

    /* ================================================================
     * INICIO MEDIDA DE CICLOS
     * Se incluye todo el cómputo AVX para contabilizar su overhead.
     * ================================================================ */
    start_counter();

    double norm2 = 0.0;
    int    iter  = 0;

    for (iter = 0; iter < max_iter; iter++) {

        norm2 = 0.0;

        for (int i = 0; i < n; i++) {

            const double *row = &a[i * n_pad];

            /* Acumulador vectorial: 4 sumas parciales de sigma */
            __m256d vsigma      = _mm256_setzero_pd();
            double  scalar_sigma = 0.0;

            /* ============================================================
             * PARTE IZQUIERDA: j en [0, i)
             * ============================================================ */
            int j = 0;

            /* Prólogo escalar: avanzar hasta primer múltiplo de AVX_W <= i */
            int left_vec = (i / AVX_W) * AVX_W;
            for (; j < left_vec && j < i; j++) {
                scalar_sigma += row[j] * x[j];
            }

            /* Bucle vectorial (acceso alineado garantizado) */
            for (; j + AVX_W <= i; j += AVX_W) {
                __m256d va = _mm256_load_pd(&row[j]);
                __m256d vx = _mm256_load_pd(&x[j]);
                vsigma = _mm256_fmadd_pd(va, vx, vsigma);
            }

            /* Epílogo escalar parte izquierda */
            for (; j < i; j++) {
                scalar_sigma += row[j] * x[j];
            }

            /* ============================================================
             * PARTE DERECHA: j en (i, n)
             * ============================================================ */
            j = i + 1;

            /* Prólogo escalar: avanzar hasta siguiente múltiplo de AVX_W */
            int right_vec = ((j + AVX_W - 1) / AVX_W) * AVX_W;
            for (; j < right_vec && j < n; j++) {
                scalar_sigma += row[j] * x[j];
            }

            /* Bucle vectorial hasta n_pad (padding=0, no contamina suma) */
            for (; j + AVX_W <= n_pad; j += AVX_W) {
                __m256d va = _mm256_load_pd(&row[j]);
                __m256d vx = _mm256_load_pd(&x[j]);
                vsigma = _mm256_fmadd_pd(va, vx, vsigma);
            }

            /* Epílogo escalar parte derecha (si quedaron elementos entre n y n_pad) */
            for (; j < n; j++) {
                scalar_sigma += row[j] * x[j];
            }

            /* ============================================================
             * REDUCCIÓN HORIZONTAL de vsigma → un solo double
             * hadd: [v0+v1, v2+v3 | v0+v1, v2+v3]
             * Suma lane baja + lane alta para obtener v0+v1+v2+v3.
             * ============================================================ */
            __m256d vhadd = _mm256_hadd_pd(vsigma, vsigma);
            __m128d vlow  = _mm256_castpd256_pd128(vhadd);
            __m128d vhigh = _mm256_extractf128_pd(vhadd, 1);
            __m128d vsum  = _mm_add_pd(vlow, vhigh);
            double  sigma = _mm_cvtsd_f64(vsum) + scalar_sigma;

            x_new[i] = (b[i] - sigma) / row[i];

            double diff = x_new[i] - x[i];
            norm2 += diff * diff;
        }

        /* x = x_new (solo los n elementos reales) */
        memcpy(x, x_new, (size_t)n * sizeof(double));

        if (sqrt(norm2) < tol) {
            break;
        }
    }

    /* ================================================================
     * FIN MEDIDA DE CICLOS
     * ================================================================ */
    double ciclos = get_counter();

    printf("v3 %d %d %.6e %.0f\n", n, iter, norm2, ciclos);

    free(a);
    free(b);
    free(x);
    free(x_new);

    return 0;
}