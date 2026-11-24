"""Projet GestionsDechet : classification et détection d'anomalies de déchets.

Façade d'accès rapide (``src.Load_data``, ``src.evaluation``, ...). Les imports
sont paresseux : TensorFlow n'est chargé que si une fonction qui en dépend est
réellement utilisée.
"""
from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, str] = {
    "Load_data": "src.preprocessing.data_loader",
    "select_data": "src.preprocessing.sampling",
    "concat_datasets": "src.preprocessing.sampling",
    "evaluation": "src.utils.model_eval",
    "plot_result": "src.utils.model_eval",
    "compare_models": "src.utils.model_eval",
    "compare_models_full": "src.utils.model_eval",
    "prepare_image": "src.utils.inference",
    "predict": "src.utils.inference",
    "JEPAMaskingConfig": "src.classification.ijepa",
    "JEPAMaskGenerator": "src.classification.ijepa",
    "IJEPADataGenerator": "src.classification.ijepa",
    "eval_with_thresholds": "src.anomaly_detection.thresholds",
    "plot_threshold_results": "src.anomaly_detection.thresholds",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        return getattr(import_module(_EXPORTS[name]), name)
    raise AttributeError(f"module 'src' has no attribute {name!r}")
