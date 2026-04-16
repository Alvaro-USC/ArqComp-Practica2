"""
analisis_resultados_2.py
========================

Variante ligera del analisis principal.
Genera las mismas graficas que analisis_resultados.py, pero en el
directorio `graficas2/` y sin recompilar la memoria LaTeX.
"""

from analisis_resultados import run_analysis


if __name__ == "__main__":
    run_analysis(results_dir="resultados", plots_dir="graficas2", compile_latex=False, latex_file="Memoria.tex")
