"""Chemins centralisés du projet.

Tous les notebooks et modules passent par ces constantes : aucun chemin
absolu (``C:/Users/...``) ni chemin relatif au dossier courant n'est
nécessaire, quel que soit l'endroit d'où le code est lancé.
"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

# --- Données -------------------------------------------------------------
DATA_DIR: Path = PROJECT_ROOT / "data"
TRAIN_DIR: Path = DATA_DIR / "train"
TEST_DIR: Path = DATA_DIR / "test"
ANOMALIES_DIR: Path = DATA_DIR / "anomalies"

# Jeux de données sérialisés (objets ``Load_data`` compressés en .zip)
DATASET_PUBLIC: Path = TRAIN_DIR / "dataset2.zip"           # jeu public (split train/test fait au chargement)
DATASET_VALIDATION: Path = TEST_DIR / "validation.zip"      # validation externe (données locales)
DATASET_LOCAL: Path = TEST_DIR / "data_local.zip"           # données locales (évaluation comparative)
DATASET_ANOMALIES: Path = ANOMALIES_DIR / "detect_anomalie.zip"  # connus (0) / inconnus (1)

# --- Modèles -------------------------------------------------------------
MODELS_DIR: Path = PROJECT_ROOT / "models"
CLS_MODELS_DIR: Path = MODELS_DIR / "classification"
ANOMALY_MODELS_DIR: Path = MODELS_DIR / "anomaly_detection"
CHECKPOINTS_DIR: Path = ANOMALY_MODELS_DIR / "checkpoints"

# --- Résultats -----------------------------------------------------------
RESULTS_DIR: Path = PROJECT_ROOT / "results"
FIGURES_DIR: Path = RESULTS_DIR / "figures"
METRICS_DIR: Path = RESULTS_DIR / "metrics"


def ensure_dirs() -> None:
    """Crée les dossiers de sortie s'ils n'existent pas encore."""
    for directory in (CLS_MODELS_DIR, ANOMALY_MODELS_DIR, CHECKPOINTS_DIR, FIGURES_DIR, METRICS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
