# GestionsDechet — Classification et détection d'anomalies de déchets

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

Système de vision par ordinateur pour le tri des déchets, fondé sur le Deep Learning. Il combine deux fonctions :

1. **Classification** d'une image parmi cinq catégories connues : `Canette`, `Organique`, `Plastique`, `Textile`, `Verre`.
2. **Détection d'anomalies** : signaler les images qui n'appartiennent à aucune de ces catégories (objets inconnus).

Trois familles d'approches sont explorées dans les notebooks :

| Fonction | Approche | Notebook |
|---|---|---|
| Classification | CNN Keras (jeu public + fine-tuning sur données locales) | `01_classification_cnn` |
| Classification | Pré-entraînement auto-supervisé I-JEPA + tête de classification | `02_classification_jepa` |
| Anomalies | VAE à blocs résiduels (score de reconstruction) | `03_anomalies_vae_residuel` |
| Anomalies | Banc de mémoire sur les *features* JEPA, PCA, Isolation Forest, One-Class SVM | `02_classification_jepa` (fin du notebook) |
| Anomalies | Seuil de confiance du classifieur softmax | `04_validation_modeles` |

---

## Structure du projet

```text
GestionsDechet/
│
├── data/                          # Données (contenu non versionné)
│   ├── train/                     #   dataset2.zip : jeu public (split 80/20 au chargement)
│   ├── test/                      #   validation.zip, data_local.zip : données locales
│   └── anomalies/                 #   detect_anomalie.zip : connus (0) / inconnus (1)
│
├── models/                        # Modèles entraînés (contenu non versionné)
│   ├── classification/            #   model_cls_*.keras, model_jepa_cls_*.keras
│   └── anomaly_detection/         #   model_dct_*.keras, checkpoints/
│
├── notebooks/
│   ├── 01_classification_cnn.ipynb
│   ├── 02_classification_jepa.ipynb
│   ├── 03_anomalies_vae_residuel.ipynb
│   ├── 04_validation_modeles.ipynb
│   ├── archive/                   # Expériences abandonnées ou redondantes (voir ci-dessous)
│   └── assets/udc1.png            # Logo utilisé dans les notebooks
│
├── src/
│   ├── __init__.py                # Façade : src.Load_data, src.evaluation, ... (imports paresseux)
│   ├── preprocessing/
│   │   ├── data_loader.py         #   Load_data : chargement, encodage, split, (dé)sérialisation .zip
│   │   └── sampling.py            #   select_data, concat_datasets
│   ├── classification/
│   │   └── ijepa.py               #   JEPAMaskingConfig, JEPAMaskGenerator, IJEPADataGenerator
│   ├── anomaly_detection/
│   │   └── thresholds.py          #   eval_with_thresholds, plot_threshold_results
│   └── utils/
│       ├── paths.py               #   Chemins centralisés (DATA_DIR, MODELS_DIR, FIGURES_DIR, ...)
│       ├── model_eval.py          #   evaluation, plot_result, compare_models, compare_models_full
│       └── inference.py           #   prepare_image, predict
│
├── results/
│   ├── figures/                   # Courbes, matrices de confusion (.png)
│   └── metrics/                   # Historiques d'entraînement (.json, .csv)
│
├── requirements.txt
├── README.md
└── LICENSE
```

### Rôle des notebooks

| Notebook | Rôle | Statut |
|---|---|---|
| `01_classification_cnn` | Chargement, rééquilibrage (`select_data`), augmentation, entraînement et évaluation du CNN | Actif |
| `02_classification_jepa` | Classifieur I-JEPA (64×64) et détection d'anomalies sur ses *features* | Actif |
| `03_anomalies_vae_residuel` | VAE résiduel 240×240, seuils de détection, matrices de confusion connu/inconnu | Actif |
| `04_validation_modeles` | Comparaison de modèles sur jeu public et données locales, seuil de confiance | Actif |
| `archive/autoencoder_baseline` | Auto-encodeur 64×64 initial, supplanté par le VAE | Abandonné |
| `archive/vae_predicteur_jepa_doublon` | Copie du VAE avec un prédicteur défini mais jamais utilisé | Redondant |
| `archive/gan_augmentation` | GAN 64×64 pour générer des images | Abandonné |
| `archive/pretraitement_hdf5_gan` | Export HDF5 des images générées par le GAN | Abandonné |

> Les notebooks portent leurs sorties d'exécution (figures, métriques) : ils sont volumineux (jusqu'à 21 Mo).

---

## Installation

Python 3.11 ou supérieur est requis.

```bash
git clone <URL_DU_DEPOT>
cd GestionsDechet

python -m venv venv
source venv/bin/activate        # Windows : venv\Scripts\activate

pip install -r requirements.txt
```

## Utilisation

### 1. Placer les données

Les jeux d'images sont des objets `Load_data` sérialisés en `.zip`. Déposer les archives aux emplacements attendus par `src/utils/paths.py` :

```text
data/train/dataset2.zip
data/test/validation.zip
data/test/data_local.zip
data/anomalies/detect_anomalie.zip
```

Les anciennes archives (créées avant la réorganisation, avec l'ancien paquet `outils`) restent lisibles.

Un dossier d'images classé par sous-dossiers de classes peut aussi être converti :

```python
from src.preprocessing import Load_data
from src.utils.paths import TRAIN_DIR

loader = Load_data(root_path=TRAIN_DIR / "images", image_shape=(240, 240, 3))
loader.load()
loader.save(TRAIN_DIR / "dataset2.zip")
```

### 2. Lancer les notebooks

```bash
jupyter lab notebooks/
```

Chaque notebook démarre par une cellule d'initialisation qui ajoute la racine du projet au `sys.path` : il peut être lancé depuis n'importe quel dossier. Ordre conseillé : `01` → `02` → `03` → `04`.

Les modèles sont écrits dans `models/`, les figures dans `results/figures/` et les historiques dans `results/metrics/`.

### 3. Réutiliser le code

```python
import src
from src.utils.paths import DATASET_PUBLIC, CLS_MODELS_DIR

df = src.Load_data.load_object(DATASET_PUBLIC)
df.encodage()
(X_train, y_train), (X_test, y_test) = df.create_data(test_size=0.2)

X_bal, y_bal = src.select_data(X_train, y_train, n=200, random_state=40)
```

L'import de `src` ne charge pas TensorFlow tant qu'une fonction qui en dépend n'est pas appelée.

### 4. Prédire sur une image

```python
from tensorflow.keras.models import load_model
from src.utils.inference import predict

discriminateur = load_model(CLS_MODELS_DIR / "<modèle_binaire>.keras")
classificateur = load_model(CLS_MODELS_DIR / "model_cls_n_13.keras")
predict("chemin/image.jpg", discriminateur, classificateur, target_size=240,
        list_name=["Can", "Organic", "Plastic", "Textile", "Glass"])
```

`predict` suppose un discriminateur binaire connu/inconnu (sortie sigmoïde) ; la détection par VAE est, pour l'instant, utilisée dans les notebooks.

---

## Périmètre actuel

- `src/anomaly_detection/` ne contient que l'évaluation par seuil de confiance. Le VAE résiduel, l'auto-encodeur et le banc de mémoire JEPA sont définis dans les notebooks `02` et `03`.
- `notebooks/02_classification_jepa` embarque une ancienne copie des classes de `src/classification/ijepa.py`.
- Aucun script d'entraînement en ligne de commande : l'entraînement se fait dans les notebooks.

## Technologies

| Technologie | Utilisation |
|---|---|
| TensorFlow / Keras | Modèles (CNN, VAE, GAN, I-JEPA) |
| scikit-learn | Métriques, PCA, Isolation Forest, One-Class SVM |
| NumPy, pandas, Pillow | Données et images |
| Matplotlib, seaborn | Visualisation |
| tqdm, h5py | Suivi de progression ; export HDF5 (archive) |

## Licence

Projet distribué sous licence **MIT** — voir [LICENSE](LICENSE).
