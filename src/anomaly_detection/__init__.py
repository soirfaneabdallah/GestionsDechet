"""Détection d'anomalies (objets de déchets inconnus).

Les modèles génératifs (VAE résiduel, auto-encodeur) et le banc de mémoire
JEPA vivent encore dans les notebooks ; ce paquet accueille le code partagé
extrait au fil de l'eau (voir ``thresholds``).
"""
from .thresholds import eval_with_thresholds, plot_threshold_results

__all__ = ["eval_with_thresholds", "plot_threshold_results"]
