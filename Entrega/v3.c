/*
 * v3.c  –  Metodo de Jacobi: paralelizacion con OpenMP
 *
 * Version base de este fichero:
 *   - schedule(static)
 *   - reduction(+:norm2)
 *
 * Variantes activables por compilacion:
 *   -DV3_SCHEDULE_DYNAMIC   -> schedule(dynamic,16)
 *   -DV3_SCHEDULE_GUIDED    -> schedule(guided)
 *   -DV3_USE_CRITICAL       -> acumulacion manual con critical
 *
 * Todas las variantes mantienen la region parallel por fuera del bucle
 * iterativo para conservar los hilos entre iteraciones.
 *
 * Salida:
 *   v3 <n> <threads> <iter> <norm2> <ciclos>
 *
 * Uso: ./v3 <n> [c]
 *   n  tamano de la matriz   (obligatorio)
 *   c  numero de hilos OMP   (opcional, por defecto 1)
 */

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include "counter.h"

#ifndef V3_CHUNK
#define V3_CHUNK 16
#endif

#if defined(V3_SCHEDULE_DYNAMIC) && defined(V3_SCHEDULE_GUIDED)
#error "Solo se puede seleccionar un tipo de scheduling por compilacion."
#endif

#if defined(V3_SCHEDULE_DYNAMIC)
#define V3_SCHEDULE_NAME "dynamic,16"
#elif defined(V3_SCHEDULE_GUIDED)
#define V3_SCHEDULE_NAME "guided"
#else
#define V3_SCHEDULE_NAME "static"
#endif

#ifdef V3_USE_CRITICAL
#define V3_REDUCTION_NAME "critical"
#else
#define V3_REDUCTION_NAME "reduction"
#endif

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

    int num_threads = 1;
    if (argc >= 3) {
        num_threads = atoi(argv[2]);
        if (num_threads <= 0) {
            num_threads = 1;
        }
    }
    omp_set_num_threads(num_threads);

    const double tol = 1e-5;
    const int max_iter = 15000;

    double *a = (double *)malloc((size_t)n * n * sizeof(double));
    double *b = (double *)malloc((size_t)n * sizeof(double));
    double *x = (double *)malloc((size_t)n * sizeof(double));
    double *x_new = (double *)malloc((size_t)n * sizeof(double));

    if (!a || !b || !x || !x_new) {
        fprintf(stderr, "Error: fallo en la reserva de memoria.\n");
        free(a);
        free(b);
        free(x);
        free(x_new);
        return 1;
    }

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

    FILE *log_stream = stderr;
    double norm2 = 0.0;
    int iter = 0;
    int converged = 0;
    int stop = 0;

#pragma omp parallel default(none) shared(a, b, x, x_new, n, tol, max_iter, norm2, iter, converged, stop, log_stream)
    {
#ifdef V3_USE_CRITICAL
        double local_norm2 = 0.0;
#endif

        while (1) {
#pragma omp single
            {
                stop = (iter >= max_iter || converged);
                if (!stop) {
                    norm2 = 0.0;
                }
            }

            if (stop) {
                break;
            }

#ifdef V3_USE_CRITICAL
            local_norm2 = 0.0;

#if defined(V3_SCHEDULE_DYNAMIC)
#pragma omp for schedule(dynamic, V3_CHUNK)
#elif defined(V3_SCHEDULE_GUIDED)
#pragma omp for schedule(guided)
#else
#pragma omp for schedule(static)
#endif
            for (int i = 0; i < n; i++) {
                const double *row = &a[i * n];
                double sigma = 0.0;

                for (int j = 0; j < i; j++) {
                    sigma += row[j] * x[j];
                }
                for (int j = i + 1; j < n; j++) {
                    sigma += row[j] * x[j];
                }

                x_new[i] = (b[i] - sigma) / row[i];

                double diff = x_new[i] - x[i];
                local_norm2 += diff * diff;
            }

#pragma omp critical
            {
                norm2 += local_norm2;
            }
#else
#if defined(V3_SCHEDULE_DYNAMIC)
#pragma omp for schedule(dynamic, V3_CHUNK) reduction(+:norm2)
#elif defined(V3_SCHEDULE_GUIDED)
#pragma omp for schedule(guided) reduction(+:norm2)
#else
#pragma omp for schedule(static) reduction(+:norm2)
#endif
            for (int i = 0; i < n; i++) {
                const double *row = &a[i * n];
                double sigma = 0.0;

                for (int j = 0; j < i; j++) {
                    sigma += row[j] * x[j];
                }
                for (int j = i + 1; j < n; j++) {
                    sigma += row[j] * x[j];
                }

                x_new[i] = (b[i] - sigma) / row[i];

                double diff = x_new[i] - x[i];
                norm2 += diff * diff;
            }
#endif

#if defined(V3_SCHEDULE_DYNAMIC)
#pragma omp for schedule(dynamic, V3_CHUNK)
#elif defined(V3_SCHEDULE_GUIDED)
#pragma omp for schedule(guided)
#else
#pragma omp for schedule(static)
#endif
            for (int i = 0; i < n; i++) {
                x[i] = x_new[i];
            }

#pragma omp single
            {
                if (sqrt(norm2) < tol) {
                    converged = 1;
                } else {
                    if (iter > 0 && iter % 1000 == 0) {
                        double ciclos_ahora = get_counter();
                        double ciclos_por_iter = ciclos_ahora / iter;
                        double eta_seg = ciclos_por_iter * (max_iter - iter) / 2.2e9;
                        fprintf(log_stream,
                                "[v3:%s:%s] n=%d | iter=%d/%d | norm2=%.3e | ETA ~%.1f s\n",
                                V3_SCHEDULE_NAME, V3_REDUCTION_NAME, n, iter, max_iter, norm2, eta_seg);
                        fflush(log_stream);
                    }
                    iter++;
                }
            }
        }
    }

    double ciclos = get_counter();

    printf("v3 %d %d %d %.6e %.0f\n", n, num_threads, iter, norm2, ciclos);

    free(a);
    free(b);
    free(x);
    free(x_new);

    return 0;
}
