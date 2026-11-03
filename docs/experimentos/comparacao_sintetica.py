"""Medição exploratória; profundidades de RF e oRF permanecem diferentes."""

from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.oRF import RF, oRF


def main():
    X, y = make_classification(
        n_samples=100, n_features=10, n_informative=5, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    print("Dados sintéticos; uma semente; k=10, p=0.7, s=0.5.")
    print("A profundidade NÃO está controlada: RF=100; oRF=10.")
    for name, cls in (("RF", RF), ("oRF", oRF)):
        np.random.seed(42)
        model = cls(k=10, p=0.7, s=0.5)
        start = perf_counter()
        model.fit(X_train, y_train)
        fit_seconds = perf_counter() - start
        start = perf_counter()
        y_pred = model.predict(X_test)
        predict_seconds = perf_counter() - start
        print(
            f"{name}: acurácia={accuracy_score(y_test, y_pred):.4f}; "
            f"treino={fit_seconds:.4f}s; predição={predict_seconds:.4f}s"
        )


if __name__ == "__main__":
    main()
