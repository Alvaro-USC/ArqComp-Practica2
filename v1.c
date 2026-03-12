/*
 * v1.c  –  Método de Jacobi: versión secuencial BASE
 *
 * Traducción directa del pseudocódigo del enunciado, sin ninguna
 * optimización manual. Se usa como referencia para medir speedup.
 *
 * API de counter.h:
 *   start_counter()  → void, guarda el timestamp TSC actual
 *   get_counter()    → double, devuelve los ciclos desde start_counter()
 *
 * Salida (una línea por ejecución, legible por Python):
 *   v1 <n> <iter> <norm2> <ciclos>
 *
 * Uso: ./v1 <n> [c]
 *   n  tamaño de la matriz  (obligatorio)
 *   c  número de hilos      (opcional, ignorado en esta versión)
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
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
    /* Argumento c aceptado pero ignorado (solo obligatorio en v4 OpenMP) */

    /* ---------- Parámetros del algoritmo ---------- */
    const double tol      = 1e-5;
    const int    max_iter = 15000;

    /* ---------- Reserva dinámica de memoria ----------
     * Matriz a en formato fila-mayor: a[i][j] == a[i*n + j]
     * Todos los arrays dependen de n → reserva dinámica obligatoria.
     */
    double *a     = (double *)malloc((size_t)n * n * sizeof(double));
    double *b     = (double *)malloc((size_t)n     * sizeof(double));
    double *x     = (double *)malloc((size_t)n     * sizeof(double));
    double *x_new = (double *)malloc((size_t)n     * sizeof(double));

    if (!a || !b || !x || !x_new) {
        fprintf(stderr, "Error: fallo en la reserva de memoria.\n");
        free(a); free(b); free(x); free(x_new);
        return 1;
    }

    /* ---------- Inicialización con valores aleatorios ----------
     * Para garantizar convergencia la matriz debe ser diagonalmente
     * dominante: se suma el total de cada fila al elemento diagonal.
     */
    srand(42);
    for (int i = 0; i < n; i++) {
        double row_sum = 0.0;
        for (int j = 0; j < n; j++) {
            a[i * n + j] = (double)rand() / RAND_MAX;
            row_sum += a[i * n + j];
        }
        a[i * n + i] += row_sum;          /* dominancia diagonal */

        b[i] = (double)rand() / RAND_MAX;
        x[i] = 0.0;                        /* estimación inicial: vector nulo */
    }

    /* ================================================================
     * INICIO MEDIDA DE CICLOS
     * start_counter() guarda el valor TSC actual (void).
     * get_counter()   devuelve los ciclos transcurridos (double).
     * Se mide solo el cómputo del pseudocódigo, sin init ni printf.
     * ================================================================ */
    start_counter();

    double norm2 = 0.0;
    int    iter  = 0;

    /* ---------- Bucle principal de Jacobi ---------- */
    for (iter = 0; iter < max_iter; iter++) {

        norm2 = 0.0;

        /* Calcular x_new[i] para cada componente i */
        for (int i = 0; i < n; i++) {

            double sigma = 0.0;

            /* Sumar a[i][j]*x[j] para todos los j != i
             * (traducción directa del pseudocódigo) */
            for (int j = 0; j < n; j++) {
                if (i != j) {
                    sigma += a[i * n + j] * x[j];
                }
            }

            x_new[i] = (b[i] - sigma) / a[i * n + i];

            /* Norma cuadrática de la diferencia entre iteraciones */
            double diff = x_new[i] - x[i];
            norm2 += diff * diff;
        }

        /* x = x_new */
        memcpy(x, x_new, (size_t)n * sizeof(double));

        /* Criterio de convergencia */
        if (sqrt(norm2) < tol) {
            break;
        }
    }

    /* ================================================================
     * FIN MEDIDA DE CICLOS
     * ================================================================ */
    double ciclos = get_counter();

    /* ---------- Salida: una línea parseable por Python ----------
     * Formato: v1 <n> <iter> <norm2> <ciclos>
     */
    printf("v1 %d %d %.6e %.0f\n", n, iter, norm2, ciclos);

    /* ---------- Liberación de memoria ---------- */
    free(a);
    free(b);
    free(x);
    free(x_new);

    return 0;
}