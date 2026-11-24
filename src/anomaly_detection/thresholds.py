"""Évaluation d'un détecteur d'inconnus par seuillage de la confiance du modèle.

Convention des étiquettes : ``0`` = connu, ``1`` = inconnu (anomalie).
Un échantillon est déclaré inconnu lorsque la confiance du modèle est
strictement inférieure au seuil.
"""
from __future__ import annotations

from typing import Any, Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score


def eval_with_thresholds(model: Any, X_test: np.ndarray, y_test: np.ndarray,
                         thresholds: Sequence[float]) -> Dict[float, Dict[str, float]]:
    """Évalue la détection d'inconnus pour chaque seuil de confiance.

    Args:
        model: modèle Keras (sortie softmax multi-classes ou score binaire).
        X_test: images normalisées.
        y_test: étiquettes (0 = connu, 1 = inconnu).
        thresholds: seuils de confiance à tester.

    Returns:
        Dictionnaire ``seuil -> {connus_detectes, inconnus_detectes, accuracy}``.
    """
    probs = model.predict(X_test, verbose=0)
    conf = np.max(probs, axis=1) if probs.ndim > 1 else probs

    results: Dict[float, Dict[str, float]] = {}
    for t in thresholds:
        y_pred = (conf < t).astype(int)
        results[t] = {
            "connus_detectes": int(np.sum((y_test == 0) & (y_pred == 0))),
            "inconnus_detectes": int(np.sum((y_test == 1) & (y_pred == 1))),
            "accuracy": accuracy_score(y_test, y_pred),
        }
    return results


def plot_threshold_results(results: Dict[float, Dict[str, float]]) -> float:
    """Trace les détections selon le seuil et renvoie le seuil de meilleure accuracy."""
    seuils = list(results.keys())
    connus = [r["connus_detectes"] for r in results.values()]
    inconnus = [r["inconnus_detectes"] for r in results.values()]
    accs = [r["accuracy"] for r in results.values()]

    _, ax = plt.subplots(figsize=(10, 6))
    ax.plot(seuils, connus, label="Connus détectés", marker="o")
    ax.plot(seuils, inconnus, label="Inconnus détectés", marker="s")
    ax.set_xlabel("Seuil de confiance")
    ax.set_ylabel("Nombre d'images détectées")
    ax.grid(True)
    ax.legend(loc="best")
    plt.title("Évolution des détections selon le seuil")
    plt.show()

    best_seuil = seuils[int(np.argmax(accs))]
    print(f"Meilleur seuil : {best_seuil:.3f} (accuracy = {max(accs):.4f})")
    return best_seuil
