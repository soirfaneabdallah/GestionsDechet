"""Utilitaires transverses : chemins, évaluation des modèles, inférence.

Les modules ``model_eval`` et ``inference`` dépendent de TensorFlow : ils ne
sont donc pas importés ici afin que ``src.utils.paths`` reste léger.
"""
from .paths import *  # noqa: F401,F403
