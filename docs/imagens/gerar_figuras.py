"""Gera figuras conceituais, sem treinar modelos nem usar dados da competição."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyBboxPatch
import numpy as np

OUT = Path(__file__).resolve().parent
NAVY = "#172F46"
TEAL = "#007F83"
CORAL = "#D56845"
BG = "#F7F8F5"
MUTED = "#526474"
LINE = "#CDD8DC"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "text.color": NAVY, "axes.labelcolor": NAVY,
    "axes.titlecolor": NAVY, "xtick.color": MUTED, "ytick.color": MUTED,
    "savefig.facecolor": BG, "figure.facecolor": BG,
})


def save(fig, name):
    fig.savefig(OUT / name, dpi=170, bbox_inches="tight", pad_inches=0.24)
    plt.close(fig)


def header(fig, title, subtitle):
    fig.text(0.05, 0.95, title, fontsize=22, weight="bold", va="top")
    fig.text(0.05, 0.89, subtitle, fontsize=11, color=MUTED, va="top")


def box(ax, x, y, w, h, title, body, accent=TEAL):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.015",
        linewidth=1.2, edgecolor=LINE, facecolor="white",
    ))
    ax.text(x + 0.017, y + h - 0.035, title, color=accent,
            fontsize=12, weight="bold", va="top")
    ax.text(x + 0.017, y + h - 0.089, body, fontsize=10.8,
            linespacing=1.7, va="top")


def arrow(ax, start, end, color=TEAL):
    ax.annotate("", xy=end, xytext=start,
                arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.8,
                            "mutation_scale": 14, "shrinkA": 2, "shrinkB": 2})


def fronteiras():
    rng = np.random.default_rng(12)
    points = rng.uniform(-2.65, 2.65, (80, 2))
    labels = (points[:, 1] > 0.72 * points[:, 0] + 0.12).astype(int)
    xx, yy = np.meshgrid(np.linspace(-3, 3, 600), np.linspace(-3, 3, 600))
    edges = np.array([-1.8, -0.6, 0.6, 1.8])
    levels = 0.72 * np.array([-2.4, -1.2, 0, 1.2, 2.4]) + 0.12
    staircase = levels[np.searchsorted(edges, xx)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.8))
    fig.subplots_adjust(left=0.07, right=0.96, top=0.75, bottom=0.18, wspace=0.22)
    header(fig, "Uma fronteira inclinada, duas formas de decidir",
           "Geometria ilustrativa • mesmos pontos nos dois painéis • sem treinamento")
    cmap = ListedColormap(["#E0F0ED", "#FAE7DD"])
    for ax, boundary, title in zip(
        axes, (staircase, 0.72 * xx + 0.12),
        ("Cortes ortogonais", "Corte oblíquo"),
    ):
        ax.contourf(xx, yy, (yy > boundary).astype(int), levels=[-0.5, 0.5, 1.5], cmap=cmap)
        for label, color, marker in ((0, TEAL, "o"), (1, CORAL, "^")):
            selected = points[labels == label]
            ax.scatter(selected[:, 0], selected[:, 1], s=32, marker=marker,
                       c=color, edgecolors="white", linewidths=0.6,
                       label=f"Classe {label}", zorder=3)
        ax.set(xlim=(-3, 3), ylim=(-3, 3), xlabel="Atributo 1", ylabel="Atributo 2")
        ax.set_title(title, fontsize=15, weight="bold", pad=12)
        ax.set_aspect("equal")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(LINE)
    # Fronteira em degraus realizável por cortes ortogonais sucessivos.
    xline = np.linspace(-3, 3, 1500)
    axes[0].plot(xline, levels[np.searchsorted(edges, xline)], color=NAVY, lw=2)
    axes[1].plot(xline, 0.72 * xline + 0.12, color=NAVY, lw=2)
    axes[0].text(0.5, -0.23, r"Vários testes do tipo $x_j \leq t$",
                 transform=axes[0].transAxes, ha="center", fontsize=12)
    axes[1].text(0.5, -0.23, r"Um teste do tipo $w^\top x \leq t$",
                 transform=axes[1].transAxes, ha="center", fontsize=12)
    handles, names = axes[1].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", bbox_to_anchor=(0.5, 0.005),
               ncol=2, frameon=False)
    save(fig, "fronteiras.png")


def no_obliquo():
    fig = plt.figure(figsize=(14, 8.2))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    header(fig, "Um novo hiperplano em cada nó",
           "Fluxo de treino para um nó que ainda pode ser dividido")
    xs, width, top, bottom, height = [0.05, 0.365, 0.68], 0.27, 0.57, 0.26, 0.22
    box(ax, xs[0], top, width, height, "1  Dados locais",
        "Amostras que chegaram ao nó.\nAtributos selecionados para a árvore.\nRótulos usados na avaliação do corte.")
    box(ax, xs[1], top, width, height, "2  Padronização",
        "Média e desvio por coluna.\nDesvio zero é substituído por 1.\nA = (X − média) / desvio")
    box(ax, xs[2], top, width, height, "3  SVD reduzida",
        "A = U Σ Vᵀ\nDireções nas linhas de Vᵀ.\nExaminar até três direções.")
    box(ax, xs[2], bottom, width, height, "4  Cortes candidatos",
        "Projetar: z = A w.\nTestar os valores únicos de z.\nExigir dois filhos não vazios.")
    box(ax, xs[1], bottom, width, height, "5  Ganho de informação",
        "Medir a redução de entropia.\nUsar os rótulos de cada filho.\nEscolher a direção e o limiar.")
    box(ax, xs[0], bottom, width, height, "6  Nó aprendido",
        "Salvar média, desvio, w e limiar.\nEsquerda: z ≤ limiar.\nDireita: z > limiar.")
    for left, right in zip(xs[:-1], xs[1:]):
        arrow(ax, (left + width + 0.006, top + height / 2),
              (right - 0.012, top + height / 2))
        arrow(ax, (right - 0.012, bottom + height / 2),
              (left + width + 0.006, bottom + height / 2))
    arrow(ax, (xs[2] + width / 2, top - 0.014),
          (xs[2] + width / 2, bottom + height + 0.015))
    arrow(ax, (xs[0] + width / 2, bottom - 0.016), (xs[0] + width / 2, 0.18))
    ax.add_patch(FancyBboxPatch((0.05, 0.07), 0.9, 0.105,
        boxstyle="round,pad=0.008,rounding_size=0.015", facecolor="#E0F0ED", edgecolor="none"))
    ax.text(0.07, 0.14, "Repetir nos filhos, com suas próprias amostras e estatísticas",
            fontsize=13, weight="bold", va="center", color=TEAL)
    ax.text(0.07, 0.095, "Os filhos recebem as linhas originais do subespaço da árvore. Na predição, reutilizar os parâmetros salvos.",
            fontsize=10.8, va="center")
    save(fig, "no-obliquo.png")


def floresta():
    fig = plt.figure(figsize=(14, 8.2))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    header(fig, "Muitas árvores, uma decisão por votação",
           "Exemplo fictício com cinco árvores • votos de uma única amostra")
    box(ax, 0.05, 0.42, 0.24, 0.33, "Dados de treinamento",
        "Para cada árvore:\n\n⌊p × n⌋ linhas com reposição.\n⌊s × d⌋ atributos sem reposição.\n\nAtributos fixos naquela árvore.")
    ys = [0.75, 0.61, 0.47, 0.33, 0.19]
    votes = [1, 0, 1, 1, 0]
    features = ["{1, 3, 4}", "{2, 4, 5}", "{1, 2, 5}", "{1, 4, 5}", "{2, 3, 4}"]
    for index, (y, vote, attrs) in enumerate(zip(ys, votes, features), start=1):
        ax.add_patch(FancyBboxPatch((0.39, y - 0.045), 0.29, 0.09,
            boxstyle="round,pad=0.008,rounding_size=0.012", facecolor="white", edgecolor=LINE))
        ax.text(0.41, y + 0.012, f"Árvore {index}", fontsize=12, weight="bold", color=TEAL)
        ax.text(0.41, y - 0.022, f"Bootstrap próprio • atributos {attrs}", fontsize=9.8)
        arrow(ax, (0.3, 0.585), (0.38, y))
        ax.text(0.725, y, str(vote), ha="center", va="center", fontsize=13,
                weight="bold", color="white", bbox={"boxstyle": "circle,pad=0.4",
                    "facecolor": CORAL if vote else TEAL, "edgecolor": "none"})
        arrow(ax, (0.69, y), (0.702, y))
    ax.text(0.725, 0.82, "Voto", ha="center", color=MUTED, fontsize=11)
    ax.plot([0.77, 0.79, 0.79, 0.77], [0.75, 0.75, 0.19, 0.19], color=LINE, lw=2)
    arrow(ax, (0.79, 0.47), (0.82, 0.47))
    ax.text(0.89, 0.53, "CLASSE 1", ha="center", fontsize=17, weight="bold", color=CORAL)
    ax.text(0.89, 0.465, "3 de 5 votos", ha="center", fontsize=12)
    ax.text(0.89, 0.415, "Moda dos rótulos", ha="center", fontsize=10, color=MUTED)
    ax.text(0.05, 0.065, "Na inferência: selecionar os mesmos atributos de cada árvore → percorrer seus nós → agregar os votos.",
            fontsize=11, color=MUTED)
    save(fig, "floresta.png")


if __name__ == "__main__":
    fronteiras()
    no_obliquo()
    floresta()
    print(f"Três figuras geradas em {OUT}")
