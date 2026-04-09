/*
 * v3.c  –  Método de Jacobi: paralelización con OpenMP
 *
 * Compilado por el makefile con:  -fopenmp
 * Solo se usan directivas de OpenMP 3.0.
 *
 * Características:
 *  1. Región parallel EXTERIOR: los hilos se crean una sola vez y
 *     se mantienen vivos entre iteraciones (evita fork/join repetido).
 *  2. División de lazos sin if(i!=j).
 *  3. Reducción de norm2 con cláusula reduction nativa de OpenMP
 *     (el compilador genera copias privadas por hilo y las combina
 *     al final sin contención; más limpio que atomic o critical).
 *     Para comparar con critical, ver comentario al final del fichero.
 *  4. schedule(static) por defecto; comentarios para dynamic/guided.
 *  5. omp single para control de convergencia y reset de norm2.
 *  6. Copia x=x_new paralelizada con omp for (escala con hilos).
 *
 * Salida: v3 <n> <threads> <iter> <norm2> <ciclos>
 *
 * Uso: ./v3 <n> [c]
 *   n  tamaño de la matriz   (obligatorio)
 *   c  número de hilos OMP   (opcional, por defecto 1)
 */
// TODO: Probar diferentes tipos de schedule, y diferentes cantidades de hilos
// schedule: static, dynamic, guided, auto, runtime

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include "counter.h"

int main(int argc, char *argv[]) {

    /* Comprobación del argumento obligatorio n */
    if (argc < 2) {
        fprintf(stderr, "Uso: %s <n> [c]\n", argv[0]);
        return 1;
    }
    int n = atoi(argv[1]);
    if (n <= 0) {
        fprintf(stderr, "Error: n debe ser un entero positivo.\n");
        return 1;
    }

    /* Argumento c: número de hilos */
    int num_threads = 1;
    if (argc >= 3) {
        num_threads = atoi(argv[2]);
        if (num_threads <= 0) num_threads = 1;
    }
    omp_set_num_threads(num_threads);

    const double tol      = 1e-5;
    const int    max_iter = 15000;

    /* Reserva dinámica de memoria */
    double *a     = (double *)malloc((size_t)n * n * sizeof(double));
    double *b     = (double *)malloc((size_t)n     * sizeof(double));
    double *x     = (double *)malloc((size_t)n     * sizeof(double));
    double *x_new = (double *)malloc((size_t)n     * sizeof(double));

    if (!a || !b || !x || !x_new) {
        fprintf(stderr, "Error: fallo en la reserva de memoria.\n");
        free(a); free(b); free(x); free(x_new);
        return 1;
    }

    /* Inicialización con valores aleatorios.
     * Para garantizar convergencia del método de Jacobi, la matriz
     * debe ser diagonalmente dominante: se suma el total de cada fila
     * al elemento diagonal. */
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

    /* INICIO MEDIDA DE CICLOS
     * Incluye TODA la región parallel para medir el overhead OpenMP
     * frente a v2 secuencial. */
    start_counter();

    double norm2     = 0.0;
    int    iter      = 0;
    int    converged = 0;

    /* REGIÓN PARALLEL EXTERIOR  (característica 1)
     * Los hilos se crean aquí una sola vez y no se destruyen hasta
     * cerrar la llave de esta región: se evita el overhead de
     * fork/join en cada iteración (a diferencia de poner el pragma
     * parallel dentro del while).
     *
     * shared : a, b, x, x_new, n, max_iter, tol, norm2, iter, converged
     * private: i, j, sigma, diff, row
     *          (declaradas dentro del bloque son privadas automáticamente) */
    #pragma omp parallel default(none) \
        shared(a, b, x, x_new, n, max_iter, tol, norm2, iter, converged, num_threads)
    {
        while (iter < max_iter && !converged) {

            /* omp for con reduction nativa (característica 3):
             *
             * reduction(+:norm2) crea una copia privada de norm2 por hilo,
             * cada hilo acumula su parte sin contención, y al salir del for
             * OpenMP las combina automáticamente con una suma.
             * Es equivalente al patrón manual local_norm2 + atomic, pero
             * expresado directamente en la directiva — más limpio y portable.
             *
             * Comparación con critical (ver bloque comentado al final):
             *   critical  → serializa: solo un hilo puede sumar a la vez.
             *               Con muchos hilos genera mucha contención.
             *   reduction → sin contención durante el bucle; combinación
             *               eficiente al final. Siempre preferible a critical
             *               para reducciones aritméticas.
             *
             * schedule(static): reparto fijo de filas entre hilos.
             *   Para probar otros modos, sustituir por:
             *     schedule(dynamic, 16)   bloque dinámico de 16 filas
             *     schedule(guided)        bloques decrecientes
             *
             * nowait: no hay barrera implícita al salir del for;
             *   la sincronización necesaria la gestiona omp barrier
             *   explícita más adelante (todos deben haber escrito
             *   x_new[] antes de que single lo copie a x[]). */
            #pragma omp for schedule(static) nowait reduction(+:norm2)
            for (int i = 0; i < n; i++) {

                const double *row = &a[i * n];
                double sigma = 0.0;

                /* División de lazos (característica 2):
                 * Se evita el if(i!=j) del pseudocódigo dividiendo el
                 * recorrido en parte izquierda [0, i) y parte derecha
                 * (i, n). Menos instrucciones de control por iteración. */

                /* Parte izquierda: j en [0, i) */
                for (int j = 0; j < i; j++) {
                    sigma += row[j] * x[j];
                }
                /* Parte derecha: j en (i, n) */
                for (int j = i + 1; j < n; j++) {
                    sigma += row[j] * x[j];
                }

                x_new[i] = (b[i] - sigma) / row[i];

                double diff = x_new[i] - x[i];
                norm2 += diff * diff;   /* reduction acumula por hilo */
            }

            /* Barrera explícita: garantiza que TODOS los hilos han
             * terminado de escribir x_new[] y que reduction ha combinado
             * norm2 antes de que single copie y evalúe convergencia.
             * (La barrera implícita del omp for fue cancelada con nowait.) */
            #pragma omp barrier

            /* single: un solo hilo evalúa convergencia y resetea norm2.
             * Barrera implícita al salir de single sincroniza a todos
             * antes de la siguiente iteración del while. */
            #pragma omp single
            {
                if (sqrt(norm2) < tol) {
                    converged = 1;
                }
                norm2 = 0.0;   /* reiniciar para la siguiente iteración */
                iter++;
            }
            /* barrera implícita de omp single */

            /* Copia paralela x = x_new (característica 6):
             * En lugar de que un solo hilo copie todo el vector (memcpy
             * secuencial, O(n)), cada hilo copia su bloque de filas.
             * Con schedule(static) el reparto es el mismo que en el for
             * de cómputo, favoreciendo la localidad de caché.
             * nowait: la barrera del omp single ya sincronizó antes de
             * llegar aquí; la próxima barrera necesaria es al inicio del
             * siguiente omp for (implícita). */
            #pragma omp for schedule(static) nowait
            for (int i = 0; i < n; i++) {
                x[i] = x_new[i];
            }
            /* La barrera implícita del omp for (sin nowait aquí) garantiza
             * que todos los hilos han terminado la copia antes de que el
             * siguiente omp for lea x[]. */

        }
    } /* fin región parallel */

    double ciclos = get_counter();

    /* norm2 se resetea dentro de single al final de cada iteración;
     * calculamos la norma final sobre el vector solución x[] para la salida. */
    double norm2_final = 0.0;
    for (int i = 0; i < n; i++) {
        norm2_final += x[i] * x[i];
    }

    printf("v3 %d %d %d %.6e %.0f\n",
           n, num_threads, iter, norm2_final, ciclos);

    free(a);
    free(b);
    free(x);
    free(x_new);

    return 0;
}

/*
 * VERSIÓN CON CRITICAL (para comparar rendimiento, documentar en memoria)
 * Sustituir el bloque omp for + reduction por este fragmento y medir
 * ciclos con el mismo n y número de hilos para ver el impacto de la
 * serialización en la actualización de norm2.
 *
 *   #pragma omp for schedule(static) nowait
 *   for (int i = 0; i < n; i++) {
 *       const double *row = &a[i * n];
 *       double sigma = 0.0;
 *       for (int j = 0; j < i; j++)       sigma += row[j] * x[j];
 *       for (int j = i + 1; j < n; j++)   sigma += row[j] * x[j];
 *       x_new[i] = (b[i] - sigma) / row[i];
 *       double diff = x_new[i] - x[i];
 *       double local = diff * diff;
 *       #pragma omp critical        // un hilo a la vez: serializa
 *       { norm2 += local; }
 *   }
 *
 * Con pocos hilos (1-2) la diferencia es pequeña.
 * Con muchos hilos (8-16) critical genera alta contención y degrada
 * el speedup. reduction no tiene este problema.
 */