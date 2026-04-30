/*
 * v2.c  –  Método de Jacobi: optimizaciones de CACHÉ (Bloqueo 2D / Tiling)
 *
 * Mejoras aplicadas (acumuladas):
 * 1. Reducción de instrucciones (puntero row).
 * 2. División de lazos (eliminar if de la diagonal).
 * 3. Desenrollamiento de lazos x4.
 * 4. Blocking 2D (Tiling): filas Y columnas procesadas en bloques de
 * BLOCK_SIZE para maximizar el uso de caché L1 para el vector x[].
 * 5. Array temporal sigma_arr[] para acumular sumas parciales.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "counter.h"

/* Tamaño de bloque para cache blocking (ajustar experimentalmente) */
#define BLOCK_SIZE 64
#define UNROLL     4

int main(int argc, char *argv[]) {

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

    double *a         = (double *)malloc((size_t)n * n * sizeof(double));
    double *b         = (double *)malloc((size_t)n     * sizeof(double));
    double *x         = (double *)malloc((size_t)n     * sizeof(double));
    double *x_new     = (double *)malloc((size_t)n     * sizeof(double));
    /* Array para guardar las sumas parciales de cada fila */
    double *sigma_arr = (double *)malloc((size_t)n     * sizeof(double));

    if (!a || !b || !x || !x_new || !sigma_arr) {
        fprintf(stderr, "Error: fallo en la reserva de memoria.\n");
        free(a); free(b); free(x); free(x_new); free(sigma_arr);
        return 1;
    }

    /* Inicialización de la matriz */
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

    start_counter();

    double norm2 = 0.0;
    int    iter  = 0;

    for (iter = 0; iter < max_iter; iter++) {
        norm2 = 0.0;
        
        /* Limpiamos el array de sumas parciales en cada iteración */
        memset(sigma_arr, 0, (size_t)n * sizeof(double));

        /* Bucle externo: Bloques de FILAS (ii) */
        for (int ii = 0; ii < n; ii += BLOCK_SIZE) {
            int i_end = (ii + BLOCK_SIZE < n) ? ii + BLOCK_SIZE : n;

            /* Bucle medio: Bloques de COLUMNAS (jj) -> BLOQUEO 2D */
            for (int jj = 0; jj < n; jj += BLOCK_SIZE) {
                int j_end = (jj + BLOCK_SIZE < n) ? jj + BLOCK_SIZE : n;

                /* Bucle interno: Cálculo dentro del bloque 2D */
                for (int i = ii; i < i_end; i++) {
                    const double *row = &a[i * n];
                    double sigma_parcial = 0.0;

                    /* Comprobamos si la diagonal cae DENTRO de este bloque 2D */
                    if (i >= jj && i < j_end) {
                        
                        /* --- EL BLOQUE CONTIENE LA DIAGONAL --- */
                        
                        /* Parte Izquierda: [jj, i) */
                        int j = jj;
                        int left_main = jj + ((i - jj) / UNROLL) * UNROLL;
                        for (; j < left_main; j += UNROLL) {
                            sigma_parcial += row[j]     * x[j];
                            sigma_parcial += row[j + 1] * x[j + 1];
                            sigma_parcial += row[j + 2] * x[j + 2];
                            sigma_parcial += row[j + 3] * x[j + 3];
                        }
                        for (; j < i; j++) {
                            sigma_parcial += row[j] * x[j];
                        }

                        /* Parte Derecha: (i, j_end) */
                        j = i + 1;
                        int right_start = ((j + UNROLL - 1) / UNROLL) * UNROLL;
                        for (; j < right_start && j < j_end; j++) {
                            sigma_parcial += row[j] * x[j];
                        }
                        int right_main = j_end - ((j_end - j) % UNROLL);
                        for (; j < right_main; j += UNROLL) {
                            sigma_parcial += row[j]     * x[j];
                            sigma_parcial += row[j + 1] * x[j + 1];
                            sigma_parcial += row[j + 2] * x[j + 2];
                            sigma_parcial += row[j + 3] * x[j + 3];
                        }
                        for (; j < j_end; j++) {
                            sigma_parcial += row[j] * x[j];
                        }

                    } else {
                        
                        /* EL BLOQUE NO TIENE DIAGONAL (El 99% de los casos)
                         * Podemos procesar todo el bloque de corrido a máxima velocidad
                         */
                        int j = jj;
                        int main_end = jj + ((j_end - jj) / UNROLL) * UNROLL;
                        for (; j < main_end; j += UNROLL) {
                            sigma_parcial += row[j]     * x[j];
                            sigma_parcial += row[j + 1] * x[j + 1];
                            sigma_parcial += row[j + 2] * x[j + 2];
                            sigma_parcial += row[j + 3] * x[j + 3];
                        }
                        /* Epílogo */
                        for (; j < j_end; j++) {
                            sigma_parcial += row[j] * x[j];
                        }
                    }

                    /* Acumulamos el resultado de este bloque en el array global */
                    sigma_arr[i] += sigma_parcial;
                }
            }

            /* Una vez sumados TODOS los bloques jj para este bloque de filas ii,
             * ya tenemos la ecuación completa. Calculamos x_new y el error.
             */
            for (int i = ii; i < i_end; i++) {
                x_new[i] = (b[i] - sigma_arr[i]) / a[i * n + i];
                double diff = x_new[i] - x[i];
                norm2 += diff * diff;
            }
        }

        memcpy(x, x_new, (size_t)n * sizeof(double));

        if (sqrt(norm2) < tol) {
            break;
        }

        if (iter > 0 && iter % 1000 == 0) {
            double ciclos_ahora    = get_counter();
            double ciclos_por_iter = ciclos_ahora / iter;
            double eta_seg         = ciclos_por_iter * (max_iter - iter) / 2.2e9;
            fprintf(stderr, "[v2] n=%d | iter=%d/%d | norm2=%.3e | ETA ~%.1f s\n",
                    n, iter, max_iter, norm2, eta_seg);
            fflush(stderr);
        }
    }

    double ciclos = get_counter();
    printf("v2 %d %d %.6e %.0f\n", n, iter, norm2, ciclos);

    free(a);
    free(b);
    free(x);
    free(x_new);
    free(sigma_arr);

    return 0;
}