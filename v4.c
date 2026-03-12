/*
 * v4.c  –  Método de Jacobi: paralelización con OpenMP
 *
 * Compilado por el makefile con:  -fopenmp
 * Solo se usan directivas de OpenMP 3.0.
 *
 * Características:
 *  1. Región parallel EXTERIOR: los hilos se crean una sola vez y
 *     se mantienen vivos entre iteraciones (evita fork/join repetido).
 *  2. División de lazos sin if(i!=j).
 *  3. Reducción de norm2 con variable local por hilo + atomic
 *     (más eficiente que critical; alternativa comentada incluida).
 *  4. schedule(static) por defecto; comentarios para dynamic/guided.
 *  5. omp single para copia x=x_new y control de convergencia.
 *
 * Salida: v4 <n> <threads> <iter> <norm2> <ciclos>
 *
 * Uso: ./v4 <n> [c]
 *   n  tamaño de la matriz   (obligatorio)
 *   c  número de hilos OMP   (opcional, por defecto 1)
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include "counter.h"

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

    /* ---------- Argumento c: número de hilos (obligatorio comprobar en v4) --- */
    int num_threads = 1;
    if (argc >= 3) {
        num_threads = atoi(argv[2]);
        if (num_threads <= 0) num_threads = 1;
    }
    omp_set_num_threads(num_threads);

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

    /* ---------- Inicialización con valores aleatorios (misma semilla) ---------- */
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
     * Incluye TODA la región parallel para medir el overhead OpenMP
     * frente a v2 secuencial.
     * ================================================================ */
    start_counter();

    double norm2    = 0.0;
    int    iter     = 0;
    int    converged = 0;

    /* ================================================================
     * REGIÓN PARALLEL EXTERIOR  (característica 1)
     *
     * Los hilos se crean aquí una sola vez y no se destruyen hasta
     * cerrar la llave de esta región.
     *
     * shared : a, b, x, x_new, n, max_iter, tol, norm2, iter, converged
     * private: i, j, sigma, diff, local_norm2, row
     *          (declaradas dentro del bloque → privadas automáticamente)
     * ================================================================ */
    #pragma omp parallel default(none) \
        shared(a, b, x, x_new, n, max_iter, tol, norm2, iter, converged, num_threads)
    {
        while (iter < max_iter && !converged) {

            /* Variable local por hilo: acumula norm2 sin contención */
            double local_norm2 = 0.0;

            /* --------------------------------------------------------
             * omp for: reparte las filas i entre hilos.
             *
             * schedule(static)   : reparto fijo, mínimo overhead.
             *   Para probar otros modos, sustituir por:
             *     schedule(dynamic, 16)   bloque dinámico de 16 filas
             *     schedule(guided)        bloques decrecientes
             *
             * nowait: no hay barrera implícita al salir del for;
             *   la sincronización la gestiona omp barrier más adelante.
             * -------------------------------------------------------- */
            #pragma omp for schedule(static) nowait
            for (int i = 0; i < n; i++) {

                const double *row = &a[i * n];
                double sigma = 0.0;

                /* División de lazos: parte izquierda [0, i) — sin if */
                for (int j = 0; j < i; j++) {
                    sigma += row[j] * x[j];
                }
                /* División de lazos: parte derecha (i, n) — sin if */
                for (int j = i + 1; j < n; j++) {
                    sigma += row[j] * x[j];
                }

                x_new[i] = (b[i] - sigma) / row[i];

                double diff = x_new[i] - x[i];
                local_norm2 += diff * diff;
            }

            /* --------------------------------------------------------
             * Reducción de norm2 con ATOMIC  (más eficiente que critical)
             *
             * Alternativa con CRITICAL (descomentar para comparar):
             *   #pragma omp critical
             *   { norm2 += local_norm2; }
             * -------------------------------------------------------- */
            #pragma omp atomic
            norm2 += local_norm2;

            /* Barrera: todos los hilos han escrito x_new[] y acumulado
             * su local_norm2 antes de que single copie y evalúe. */
            #pragma omp barrier

            /* --------------------------------------------------------
             * single: un solo hilo copia x=x_new y comprueba convergencia.
             * Barrera implícita al salir garantiza coherencia para la
             * siguiente iteración.
             * -------------------------------------------------------- */
            #pragma omp single
            {
                memcpy(x, x_new, (size_t)n * sizeof(double));

                if (sqrt(norm2) < tol) {
                    converged = 1;
                }

                norm2 = 0.0;   /* reiniciar para la siguiente iteracion */
                iter++;
            }
            /* barrera implicita de omp single */

           
        }
    } /* fin region parallel */

    /* ================================================================
     * FIN MEDIDA DE CICLOS
     * ================================================================ */
    double ciclos = get_counter();

    /* norm2 se resetea dentro de single; calculamos la norma final
     * directamente sobre el vector solución x[] para la salida. */
    double norm2_final = 0.0;
    for (int i = 0; i < n; i++) {
        norm2_final += x[i] * x[i];
    }

    printf("v4 %d %d %d %.6e %.0f\n",
           n, num_threads, iter, norm2_final, ciclos);

    free(a);
    free(b);
    free(x);
    free(x_new);

    return 0;

}

