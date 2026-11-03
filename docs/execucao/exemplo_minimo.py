"""Demonstração sintética reproduzível; não é uma avaliação do Kaggle."""

from pathlib import Path
import sys

import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.oRF import oRF


def main():
    X, y = make_classification(
        n_samples=100, n_features=10, n_informative=5, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    np.random.seed(42)
    model = oRF(k=15, p=0.7, s=0.5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Demonstração sintética — acurácia: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Árvores: {len(model.trees)}; predições: {len(y_pred)}")


if __name__ == "__main__":
    main()
