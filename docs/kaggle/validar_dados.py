"""Valida o contrato estrutural do NPZ usado pelo gerador original."""

import argparse
from pathlib import Path

import numpy as np


def validar(path):
    with np.load(path, allow_pickle=False) as data:
        required = {"X_train", "y_train", "X_test"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"Chaves ausentes: {', '.join(sorted(missing))}")
        arrays = {key: data[key] for key in sorted(required)}

    for key, array in arrays.items():
        if array.dtype.kind not in "iuf":
            raise ValueError(f"{key}: use valores numéricos reais, não objetos/texto.")
        if not np.isfinite(array).all():
            raise ValueError(f"{key}: contém NaN ou infinito.")
    X_train, y_train, X_test = (arrays[k] for k in ("X_train", "y_train", "X_test"))
    if X_train.ndim != 2 or X_test.ndim != 2 or y_train.ndim != 1:
        raise ValueError("Formas esperadas: X_train e X_test 2D; y_train 1D.")
    if min(*X_train.shape, *X_test.shape) <= 0:
        raise ValueError("Treino e teste devem ter linhas e colunas não vazias.")
    if len(y_train) != len(X_train):
        raise ValueError("X_train e y_train têm números de amostras diferentes.")
    if X_test.shape[1] != X_train.shape[1]:
        raise ValueError("Treino e teste têm números de atributos diferentes.")
    n_bootstrap = int(0.63 * len(X_train))
    n_features = int(0.27 * X_train.shape[1])
    if n_bootstrap < 1 or n_features < 1:
        raise ValueError("A configuração padrão seleciona zero linhas ou atributos.")
    print(f"Contrato estrutural válido: {path}")
    for key, array in arrays.items():
        print(f"{key}: shape={array.shape}; dtype={array.dtype}")
    print(f"Padrão oRF: {n_bootstrap} sorteios de linhas e {n_features} atributos/árvore.")
    if n_bootstrap < 10:
        print("Atenção: cada árvore terá menos de dez amostras e será uma folha.")
    print("Confira separadamente as classes, os IDs e as regras oficiais do Kaggle.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arquivo", nargs="?", type=Path, default=Path("data/data.npz"))
    args = parser.parse_args()
    try:
        validar(args.arquivo)
    except (OSError, ValueError, TypeError) as exc:
        parser.exit(1, f"Dados inválidos: {exc}\n")


if __name__ == "__main__":
    main()
