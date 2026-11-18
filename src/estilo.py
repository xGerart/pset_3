"""Estilo común de los gráficos (validado para daltonismo con el validador de paleta).

Orden categórico fijo: el color sigue a la entidad, no al ranking. El ámbar queda por
debajo de 3:1 de contraste, por eso toda serie lleva etiqueta directa o leyenda.
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from src.datos import RAIZ

INDIGO, TEAL, AMBAR, GRIS = "#544ddb", "#1fa396", "#e9820c", "#8c8c8c"
TINTA, TINTA_2, REJILLA = "#1f1f1f", "#55534e", "#e6e5e1"

SECUENCIAL = LinearSegmentedColormap.from_list("indigo", ["#f4f3fd", "#a9a4ee", INDIGO, "#231d7a"])
DIVERGENTE = LinearSegmentedColormap.from_list("ambar_indigo", [AMBAR, "#f4f4f2", INDIGO])

DIR_FIGURAS = RAIZ / "docs" / "figuras"


def aplicar():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
        "font.size": 9.5, "axes.titlesize": 10.5, "axes.titleweight": "bold",
        "axes.edgecolor": REJILLA, "axes.labelcolor": TINTA_2, "text.color": TINTA,
        "xtick.color": TINTA_2, "ytick.color": TINTA_2,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": REJILLA, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "lines.linewidth": 2.0, "legend.frameon": False,
        "axes.prop_cycle": plt.cycler(color=[INDIGO, TEAL, AMBAR, GRIS]),
    })


def guardar(fig, nombre: str) -> Path:
    DIR_FIGURAS.mkdir(parents=True, exist_ok=True)
    ruta = DIR_FIGURAS / f"{nombre}.png"
    fig.savefig(ruta)
    return ruta
