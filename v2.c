/*
 * v2.c  –  Método de Jacobi: optimizaciones de CACHÉ
 *
 * Misma semilla y resultado numérico que v1.
 * Mejoras aplicadas:
 *  1. Reducción de instrucciones: se elimina if(i!=j) pre-restando el diagonal.
 *  2. División de lazos: bucle j dividido en [0,i) y (i,n].
 *  3. Desenrollamiento de lazos x4 en ambas mitades.
 *  4. Blocking: filas procesadas en bloques de BLOCK_SIZE para
 *     mantener x[] caliente en caché L1/L2.
 *
 * Salida: v2 <n> <iter> <norm2> <ciclos>
 *
 * Uso: ./v2 <n> [c]
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "counter.h"

/* Tamaño de bloque para cache blocking (ajustar experimentalmente) */
#define BLOCK_SIZE 64
/* Factor de desenrollamiento del bucle j */
#define UNROLL     4

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

    /* ---------- Reserva dinámica de memoria ---------- */
    double *a     = (double *)malloc((size_t)n * n * sizeof(double));
    double *b     = (double *)malloc((size_t)n     * sizeof(double));
    double *x     = (double *)malloc((size_t)n     * sizeof(double));
    double *x_new = (double *)malloc((size_t)n     * sizeof(double));

    if (!a || !b || !x || !x_new) {
        fprintf(stderr, "Error: fallo en la reserva de memoria.\n");
        free(a); free(b); free(x); free(x_new);
        return 1;
    }

    /* ---------- Inicialización idéntica a v1 ---------- */
    srand(42);
    for (int i = 0; i < n; i++) {
        double row_sum = 0.0;
        for (int j = 0; j < n; j++) {
            a[i * n + j] = (double)rand() / RAND_MAX;
            row_sum += a[i * n + j];
        }
        a[i * n + i] += row_sum;

        b[i] = (double)rand() / RAND_MAX;
        x[i] = 0.0;
    }

    /* ================================================================
     * INICIO MEDIDA DE CICLOS
     * ================================================================ */
    start_counter();

    double norm2 = 0.0;
    int    iter  = 0;

    for (iter = 0; iter < max_iter; iter++) {

        norm2 = 0.0;

        /* ==== Mejora 4: Blocking por bloques de BLOCK_SIZE filas ====
         * Dividimos las n filas en bloques. Dentro de cada bloque el
         * vector x[] permanece en caché, reduciendo fallos de caché.
         */
        for (int ii = 0; ii < n; ii += BLOCK_SIZE) {

            int i_end = (ii + BLOCK_SIZE < n) ? ii + BLOCK_SIZE : n;

            for (int i = ii; i < i_end; i++) {

                /* Puntero directo a la fila i: evita recalcular i*n
                 * en cada acceso (Mejora 1) */
                const double *row = &a[i * n];
                double sigma = 0.0;

                /* ==== Mejoras 2 y 3: división + desenrollamiento x4 ====
                 *
                 * PARTE IZQUIERDA: j en [0, i)  — sin diagonal, sin rama
                 */
                int j = 0;
                int left_main = (i / UNROLL) * UNROLL;
                for (; j < left_main; j += UNROLL) {
                    sigma += row[j]     * x[j];
                    sigma += row[j + 1] * x[j + 1];
                    sigma += row[j + 2] * x[j + 2];
                    sigma += row[j + 3] * x[j + 3];
                }
                /* Epílogo izquierdo */
                for (; j < i; j++) {
                    sigma += row[j] * x[j];
                }

                /* PARTE DERECHA: j en (i, n)  — sin diagonal, sin rama */
                j = i + 1;
                int right_start = ((j + UNROLL - 1) / UNROLL) * UNROLL;
                /* Prólogo derecho: alinear a múltiplo de UNROLL */
                for (; j < right_start && j < n; j++) {
                    sigma += row[j] * x[j];
                }
                /* Bucle principal derecho desenrollado */
                int right_main = n - ((n - j) % UNROLL);
                for (; j < right_main; j += UNROLL) {
                    sigma += row[j]     * x[j];
                    sigma += row[j + 1] * x[j + 1];
                    sigma += row[j + 2] * x[j + 2];
                    sigma += row[j + 3] * x[j + 3];
                }
                /* Epílogo derecho */
                for (; j < n; j++) {
                    sigma += row[j] * x[j];
                }

                /* Acceso directo al diagonal con el puntero row */
                x_new[i] = (b[i] - sigma) / row[i];

                double diff = x_new[i] - x[i];
                norm2 += diff * diff;
            }
        }

        /* x = x_new */
        memcpy(x, x_new, (size_t)n * sizeof(double));

        if (sqrt(norm2) < tol) {
            break;
        }

        /* ETA cada 1000 iteraciones (visible con: tail -f logs/*.err) */
        if (iter > 0 && iter % 1000 == 0) {
            double ciclos_ahora    = get_counter();
            double ciclos_por_iter = ciclos_ahora / iter;
            double eta_seg         = ciclos_por_iter * (max_iter - iter) / 2.2e9;
            fprintf(stderr, "[v2] n=%d | iter=%d/%d | norm2=%.3e | ETA ~%.1f s\n",
                    n, iter, max_iter, norm2, eta_seg);
            fflush(stderr);
        }
    }

    /* ================================================================
     * FIN MEDIDA DE CICLOS
     * ================================================================ */
    double ciclos = get_counter();

    printf("v2 %d %d %.6e %.0f\n", n, iter, norm2, ciclos);

    free(a);
    free(b);
    free(x);
    free(x_new);

    return 0;
}