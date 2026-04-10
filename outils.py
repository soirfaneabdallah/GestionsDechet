from sklearn.metrics import confusion_matrix, classification_report, f1_score
import time
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tqdm import tqdm
from PIL import Image
import os
import copy
import random
import pandas as pd
from sklearn.model_selection import train_test_split
sns.set_theme()
import pickle, zipfile, io
from datetime import datetime
import json
import os
import random
import copy
import io
import zipfile
import pickle
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
import math
class Load_data:
    """
    Classe pour charger, préparer et gérer des ensembles d'images pour l'entraînement de modèles ML/DL.
    Parcourt récursivement tous les sous-dossiers pour trouver les images.
    """

    def __init__(self, 
                 root_path: str = None, 
                 path_list: list[str] = None, 
                 extension: tuple[str, ...] = ('.png', '.jpg', '.jpeg', '.JPG', '.bmp', '.tiff'),
                 image_shape: tuple[int, int, int] = (240, 240, 3)) -> None:
        """
        Initialise la classe.
        
        :param root_path: dossier racine à parcourir récursivement
        :param path_list: liste alternative de dossiers spécifiques
        :param extension: extensions des fichiers supportés
        :param image_shape: tuple (H, W, C) -> hauteur, largeur, canaux (1 = grayscale, 3 = RGB)
        """
        self.root_path_ = root_path
        self.path_list_ = path_list
        self.extension_ = extension
        self.name_label_ = []
        self.data_ = None
        self.data_label_ = None
        self.original_data_ = None
        self.image_shape_ = image_shape  # (H, W, C)
        self.class_mapping_ = {}  # mapping entre classes et labels encodés

    # ---------------------- Parcours récursif des dossiers ----------------------
    def _get_image_folders(self, root_path: str) -> list:
        """
        Parcourt récursivement tous les sous-dossiers pour trouver ceux qui contiennent des images.
        Retourne une liste de dossiers qui contiennent directement des images.
        """
        image_folders = []
        
        for dirpath, dirnames, filenames in os.walk(root_path):
            # Vérifier si ce dossier contient au moins une image
            has_images = any(f.lower().endswith(self.extension_) for f in filenames)
            
            if has_images:
                # Le dossier contient des images directement
                image_folders.append(dirpath)
            else:
                # Vérifier si les sous-dossiers contiennent des images
                for dirname in dirnames:
                    subdir_path = os.path.join(dirpath, dirname)
                    if any(f.lower().endswith(self.extension_) for f in os.listdir(subdir_path) if os.path.isfile(os.path.join(subdir_path, f))):
                        image_folders.append(subdir_path)
        
        # Supprimer les doublons (au cas où un dossier parent serait aussi compté)
        image_folders = list(set(image_folders))
        
        return image_folders

    def _get_label_from_path(self, image_path: str, root_path: str) -> str:
        """
        Extrait le label à partir du chemin de l'image.
        Le label est le nom du dossier parent immédiat de l'image.
        """
        # Obtenir le dossier contenant l'image
        image_dir = os.path.dirname(image_path)
        # Le label est le nom de ce dossier
        label = os.path.basename(image_dir)
        return label

    # ---------------------- Chargement ----------------------
    def load(self, recursive: bool = True) -> None:
        """
        Charge et prétraite les images.
        
        :param recursive: si True, parcourt récursivement tous les sous-dossiers.
                         si False, charge seulement les dossiers explicitement listés.
        """
        image_data, labels = [], []
        H, W, C = self.image_shape_
        
        # Déterminer la liste des dossiers à parcourir
        folders_to_scan = []
        
        if self.root_path_ and recursive:
            # Parcours récursif du dossier racine
            folders_to_scan = self._get_image_folders(self.root_path_)
            print(f"Trouvé {len(folders_to_scan)} dossiers contenant des images")
        elif self.path_list_:
            # Utilisation de la liste explicite
            folders_to_scan = self.path_list_
        
        if not folders_to_scan:
            print("Aucun dossier à scanner. Vérifiez root_path_ ou path_list_.")
            return
        
        # Chargement des images
        for folder in tqdm(folders_to_scan, desc="Chargement des dossiers"):
            # Le label est le nom du dossier courant (celui qui contient directement les images)
            label = os.path.basename(os.path.normpath(folder))
            
            if label not in self.name_label_:
                self.name_label_.append(label)
            
            for fichier in os.listdir(folder):
                if fichier.lower().endswith(self.extension_):
                    try:
                        path_image = os.path.join(folder, fichier)
                        img = Image.open(path_image)
                        
                        # Convertir selon le nombre de canaux
                        if C == 1:
                            img = img.convert("L")  # grayscale
                        elif C == 3:
                            img = img.convert("RGB")
                        else:
                            raise ValueError("Le nombre de canaux doit être 1 (gris) ou 3 (RGB).")
                        
                        # Redimensionnement
                        img = img.resize((W, H))
                        img = np.array(img)
                        
                        # Ajout d'une dimension pour grayscale
                        if C == 1:
                            img = np.expand_dims(img, axis=-1)
                        
                        image_data.append(img)
                        labels.append(label)
                        
                    except (OSError, ValueError, Exception) as e:
                        print(f"Erreur lors du chargement de {fichier}: {e}")
                        continue
        
        print(f"Images chargées : {len(image_data)}")
        
        # Création du DataFrame
        self.data_ = pd.DataFrame({'Image': image_data, 'Label': labels})
        self.data_label_ = pd.DataFrame({'Label': labels})
        self.copy()
        
        # Affichage du nombre de classes trouvées
        unique_labels = self.data_['Label'].unique()
        print(f"Classes trouvées : {len(unique_labels)}")
        for lbl in unique_labels:
            count = len(self.data_[self.data_['Label'] == lbl])
            print(f"  - {lbl}: {count} images")

    def load_from_multiple_roots(self, root_paths: list[str], recursive: bool = True) -> None:
        """
        Charge des images depuis plusieurs dossiers racines.
        
        :param root_paths: liste des dossiers racines à parcourir
        :param recursive: si True, parcourt récursivement les sous-dossiers
        """
        all_folders = []
        
        for root in root_paths:
            if recursive:
                folders = self._get_image_folders(root)
                all_folders.extend(folders)
            else:
                all_folders.append(root)
        
        # Supprimer les doublons
        all_folders = list(set(all_folders))
        
        self.path_list_ = all_folders
        self.root_path_ = None
        self.load(recursive=False)

    # ---------------------- Mélange ----------------------
    def shuffle(self) -> None:
        """ Mélange aléatoirement les images et labels. """
        if self.data_ is not None:
            self.data_ = self.data_.sample(frac=1, random_state=42).reset_index(drop=True)
            self.data_label_ = pd.DataFrame({'Label': self.data_["Label"].tolist()})

    # ---------------------- Sauvegarde état ----------------------
    def copy(self) -> None:
        """ Sauvegarde une copie des données originales. """
        if self.data_ is not None:
            self.original_data_ = copy.deepcopy(self.data_)

    def restore_data(self) -> None:
        """ Restaure les données originales. """
        if self.original_data_ is not None:
            self.data_ = copy.deepcopy(self.original_data_)
            self.data_label_ = pd.DataFrame({'Label': self.data_["Label"].tolist()})

    # ---------------------- Ajout de nouvelles données ----------------------
    def add_data(self, folder: str, recursive: bool = True) -> None:
        """
        Ajoute des images depuis un nouveau dossier ou une nouvelle racine.
        
        :param folder: chemin vers le dossier ou la racine
        :param recursive: si True, parcourt récursivement les sous-dossiers
        """
        H, W, C = self.image_shape_
        
        # Déterminer les dossiers à scanner
        if recursive and os.path.isdir(folder):
            folders_to_scan = self._get_image_folders(folder)
        else:
            folders_to_scan = [folder]
        
        image_data, labels = [], []
        
        for scan_folder in folders_to_scan:
            label = os.path.basename(os.path.normpath(scan_folder))
            
            # Si la classe est nouvelle, on l'ajoute
            if label not in self.name_label_:
                self.name_label_.append(label)
            
            for fichier in tqdm(os.listdir(scan_folder), desc=f"Ajout: {label}"):
                if fichier.lower().endswith(self.extension_):
                    try:
                        path_image = os.path.join(scan_folder, fichier)
                        img = Image.open(path_image)
                        
                        # Conversion selon shape
                        if C == 1:
                            img = img.convert("L")
                        elif C == 3:
                            img = img.convert("RGB")
                        else:
                            raise ValueError("Le nombre de canaux doit être 1 (gris) ou 3 (RGB).")
                        
                        # Redimensionnement
                        img = img.resize((W, H))
                        img = np.array(img)
                        
                        if C == 1:
                            img = np.expand_dims(img, axis=-1)
                        
                        image_data.append(img)
                        labels.append(label)
                        
                    except (OSError, ValueError, Exception) as e:
                        print(f"Erreur: {e}")
                        continue
        
        # Fusion avec le dataset existant
        if self.data_ is None:
            self.data_ = pd.DataFrame({'Image': image_data, 'Label': labels})
        else:
            new_df = pd.DataFrame({'Image': image_data, 'Label': labels})
            self.data_ = pd.concat([self.data_, new_df], ignore_index=True)
        
        # Mise à jour des labels
        self.data_label_ = pd.DataFrame({'Label': self.data_["Label"].tolist()})
        self.copy()
        
        print(f"Ajouté : {len(image_data)} images")
        print(f"Total : {len(self.data_)} images, {len(self.name_label_)} classes")

    # ---------------------- Visualisation ----------------------
    def plot(self, view_code: bool = False, name_fig: str = "fig", register: bool = False, n_samples: int = 20) -> None:
        """
        Affiche un échantillon aléatoire d'images avec leurs labels.
        
        :param view_code: afficher les codes numériques plutôt que les noms
        :param name_fig: nom du fichier pour la sauvegarde
        :param register: sauvegarder la figure
        :param n_samples: nombre d'images à afficher (doit être un multiple de 4 ou 5)
        """
        if self.data_ is None:
            print("Aucune donnée chargée.")
            return
        
        n_samples = min(n_samples, len(self.data_))
        n_cols = 5
        n_rows = (n_samples + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes.flatten()
        
        indices = np.random.choice(len(self.data_), n_samples, replace=False)
        
        title_source = self.data_["Label"] if not view_code else self.data_label_["Label"]
        
        for i, idx in enumerate(indices):
            axes[i].imshow(self.data_["Image"][idx])
            axes[i].set_title(str(title_source[idx]))
            axes[i].axis("off")
        
        # Cacher les axes inutilisés
        for i in range(len(indices), len(axes)):
            axes[i].axis("off")
        
        plt.tight_layout()
        if register:
            plt.savefig(f"{name_fig}.png", dpi=300, bbox_inches='tight')
        plt.show()

    # ---------------------- Traitement des labels ----------------------
    def encodage(self) -> None:
        """
        Encode les labels en entiers.
        """
        if self.data_ is None:
            return
        
        if isinstance(self.data_["Label"].iloc[0], str):
            self.class_mapping_ = {name: i for i, name in enumerate(self.name_label_)}
            self.data_["Label"] = self.data_["Label"].map(self.class_mapping_)
            self.data_label_ = pd.DataFrame({'Label': self.data_["Label"].tolist()})
            print(f"Encodage effectué. Mapping: {self.class_mapping_}")

    def decode_labels(self, encoded_labels: list) -> list:
        """
        Décode des labels encodés en noms de classes.
        
        :param encoded_labels: liste des labels encodés
        :return: liste des noms de classes
        """
        reverse_mapping = {v: k for k, v in self.class_mapping_.items()}
        return [reverse_mapping.get(label, "unknown") for label in encoded_labels]

    # ---------------------- Compression ----------------------
    def compress(self, image: np.ndarray, test: bool = True, k: int = 100, threshold_kb: float = 200.0) -> np.ndarray:
        """
        Compresse une image via décomposition SVD si elle dépasse un seuil en Ko.
        """
        image_size_kb = image.nbytes / 1024
        if image_size_kb > threshold_kb:
            compressed_channels = []
            H, W = image.shape[:2]
            for i in range(image.shape[2] if image.ndim == 3 else 1):
                channel = image[:, :, i] if image.ndim == 3 else image
                U, S, Vt = np.linalg.svd(channel, full_matrices=False)
                Sk = np.diag(S[:k])
                compressed = np.dot(U[:, :k], np.dot(Sk, Vt[:k, :]))
                compressed_channels.append(compressed)
            
            if len(compressed_channels) > 1:
                compressed_img = np.stack(compressed_channels, axis=2)
            else:
                compressed_img = compressed_channels[0]
            
            compressed_img = np.clip(compressed_img, 0, 255).astype(np.uint8)
            resized_img = Image.fromarray(compressed_img.astype(np.uint8)).resize((W, H))
            
            if test:
                fig, ax = plt.subplots(1, 2, figsize=(12, 6))
                ax[0].imshow(resized_img)
                ax[0].set_title("Image compressée")
                ax[1].imshow(image.astype(np.uint8))
                ax[1].set_title("Image originale")
                plt.show()
            
            return np.array(resized_img) / 255.0
        
        return np.array(image) / 255.0

    def compress_data(self, threshold_kb: float = 200.0) -> None:
        """
        Compresse toutes les images de l'ensemble de données.
        """
        if self.data_ is None:
            return
        
        compressed_images = []
        for img in tqdm(self.data_["Image"], desc="Compression"):
            compressed_images.append(self.compress(img, test=False, threshold_kb=threshold_kb))
        
        self.data_["Image"] = compressed_images

    # ---------------------- Sauvegarde / Chargement ----------------------
    def save(self, zip_path: str = "dataset.zip", internal_name: str = "loader.pkl") -> None:
        """
        Sauvegarde l'objet dans une archive zip.
        """
        backup = self.data_
        self.data_ = None
        buffer = io.BytesIO()
        pickle.dump(self, buffer)
        self.data_ = backup
        buffer.seek(0)
        
        with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(internal_name, buffer.read())
        
        print(f"Dataset sauvegardé dans {zip_path}")

    @staticmethod
    def load_object(zip_path: str = "dataset.zip", internal_name: str = "loader.pkl") -> "Load_data":
        """
        Charge un objet Load_data sauvegardé dans une archive zip.
        """
        with zipfile.ZipFile(zip_path, 'r') as zf:
            with zf.open(internal_name) as f:
                obj = pickle.load(f)
                
                # Restaurer les données originales si disponibles
                if obj.original_data_ is not None:
                    obj.data_ = copy.deepcopy(obj.original_data_)
                
                # Recréer data_label_ si nécessaire
                if obj.data_ is not None and obj.data_label_ is None:
                    obj.data_label_ = pd.DataFrame({'Label': obj.data_["Label"].tolist()})
                
                # S'assurer que name_label_ existe
                if not hasattr(obj, "name_label_"):
                    obj.name_label_ = []
                
                return obj

    # ---------------------- Split train/test ----------------------
    def create_data(self, test_size: float = 0.2, shuffle: bool = True):
        """
        Crée les ensembles d'entraînement et de test.
        
        :param test_size: proportion du jeu de test
        :param shuffle: mélanger les données avant le split
        :return: tuple ((X_train, Y_train), (X_test, Y_test))
        """
        if self.data_ is None:
            raise ValueError("Aucune donnée chargée. Appelez load() d'abord.")
        
        # S'assurer que les labels sont encodés
        if isinstance(self.data_["Label"].iloc[0], str):
            self.encodage()
        
        data_train, data_test = train_test_split(
            self.data_, test_size=test_size, random_state=42, shuffle=shuffle
        )
        
        X_train = np.array(data_train['Image'].tolist(), dtype=np.float32)
        Y_train = np.array(data_train['Label'].tolist(), dtype=np.int32)
        X_test = np.array(data_test['Image'].tolist(), dtype=np.float32)
        Y_test = np.array(data_test['Label'].tolist(), dtype=np.int32)
        
        return (X_train, Y_train), (X_test, Y_test)

    # ---------------------- Informations ----------------------
    def info(self) -> None:
        """Affiche les informations sur le dataset."""
        if self.data_ is None:
            print("Aucune donnée chargée.")
            return
        
        print("\n" + "="*50)
        print("INFORMATIONS SUR LE DATASET")
        print("="*50)
        print(f"Nombre total d'images : {len(self.data_)}")
        print(f"Nombre de classes : {len(self.name_label_)}")
        print(f"Shape des images : {self.image_shape_}")
        print("\nRépartition par classe :")
        
        label_counts = self.data_['Label'].value_counts()
        for label, count in label_counts.items():
            if isinstance(label, int):
                label_name = self.decode_labels([label])[0] if self.class_mapping_ else str(label)
                print(f"  - {label_name}: {count} images")
            else:
                print(f"  - {label}: {count} images")
        
        print("="*50 + "\n")


    def reshape(self, target_shape: tuple[int, int, int], batch_size: int = 32) -> None:
        
        """
        Redimensionne les images par batch vers target_shape.
       
        - Si l'image est plus grande → resize direct
        - Si plus petite → interpolation
        - Utilise tqdm pour suivi
        - Modifie self.data_["Image"] directement
        """
    
        if self.data_ is None:
            raise ValueError("Aucune donnée chargée.")
    
        H_target, W_target, C_target = target_shape
        total_images = len(self.data_)
    
        new_images = []
    
        for start in tqdm(range(0, total_images, batch_size), desc="Reshape en batch"):
            batch = self.data_["Image"].iloc[start:start + batch_size]
    
            for img in batch:
                img = np.array(img)
                H_init, W_init, C_init = img.shape
    
                # Vérification des canaux
                if C_init != C_target:
                    if C_target == 3:
                        img = Image.fromarray(img).convert("RGB")
                    elif C_target == 1:
                        img = Image.fromarray(img).convert("L")
                    img = np.array(img)
    
                # Resize nécessaire ?
                if (H_init != H_target) or (W_init != W_target):
    
                    pil_img = Image.fromarray(img)
    
                    # Si image plus petite → interpolation plus douce
                    if H_init < H_target or W_init < W_target:
                        resized = pil_img.resize((W_target, H_target), Image.BICUBIC)
                    else:
                        resized = pil_img.resize((W_target, H_target), Image.BILINEAR)
    
                    img = np.array(resized)
    
                    if C_target == 1:
                        img = np.expand_dims(img, axis=-1)
    
                new_images.append(img)
    
        self.data_["Image"] = new_images
        self.image_shape_ = target_shape



def compare_models(y_true_test: np.ndarray,
                   y_pred_test_model1: np.ndarray,
                   y_pred_test_model2: np.ndarray,
                   y_true_local: np.ndarray,
                   y_pred_local_model1: np.ndarray,
                   y_pred_local_model2: np.ndarray) -> Dict[str, Tuple[float, float]]:
    """
    Compare les performances de deux modèles sur un jeu de test et un jeu local.
    
    Calcule les pourcentages de données correctement et incorrectement classifiées
    pour chaque modèle, puis affiche un graphe comparatif.

    Parameters
    ----------
    y_true_test : np.ndarray
        Labels réels du jeu de test.
    y_pred_test_model1 : np.ndarray
        Prédictions du modèle 1 sur le jeu de test.
    y_pred_test_model2 : np.ndarray
        Prédictions du modèle 2 sur le jeu de test.
    y_true_local : np.ndarray
        Labels réels du jeu local.
    y_pred_local_model1 : np.ndarray
        Prédictions du modèle 1 sur le jeu local.
    y_pred_local_model2 : np.ndarray
        Prédictions du modèle 2 sur le jeu local.

    Returns
    -------
    results : Dict[str, Tuple[float, float]]
        Dictionnaire contenant, pour chaque modèle et chaque dataset,
        un tuple (pourcentage correct, pourcentage incorrect).
    """

    results: Dict[str, Tuple[float, float]] = {}

    # --- Jeu de test ---
    acc_test_m1 = np.mean(y_true_test == y_pred_test_model1) * 100
    acc_test_m2 = np.mean(y_true_test == y_pred_test_model2) * 100
    results["Test_Model1"] = (acc_test_m1, 100 - acc_test_m1)
    results["Test_Model2"] = (acc_test_m2, 100 - acc_test_m2)

    # --- Données locales ---
    acc_local_m1 = np.mean(y_true_local == y_pred_local_model1) * 100
    acc_local_m2 = np.mean(y_true_local == y_pred_local_model2) * 100
    results["Local_Model1"] = (acc_local_m1, 100 - acc_local_m1)
    results["Local_Model2"] = (acc_local_m2, 100 - acc_local_m2)

    # --- Tracé comparatif ---
    labels = list(results.keys())
    correct = [val[0] for val in results.values()]
    incorrect = [val[1] for val in results.values()]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, correct, width, label="Bien classé (%)")
    bars2 = ax.bar(x + width/2, incorrect, width, label="Mal classé (%)")

    ax.set_ylabel("Pourcentage (%)")
    ax.set_title("Comparaison des performances des modèles")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Affichage des valeurs sur les barres
    for bars in (bars1, bars2):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.1f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.show()

    return results

def evaluation(model, X_test, y_test, return_conf_mat=True, return_clss=True, labels=[]):
    # Prédictions
    predict_proba = model.predict(X_test)
    y_pred = np.argmax(predict_proba, axis=1)

    # Matrice de confusion
    conf_matrice = confusion_matrix(y_test, y_pred)

    if return_conf_mat:
        print("========================================== Confusion Matrix =====================================================")
        print(conf_matrice)
        f, ax = plt.subplots(figsize=(6, 5))
        conf_matrice_nor = conf_matrice.astype('float') / conf_matrice.sum(axis=1)[:, np.newaxis]
        sns.heatmap(conf_matrice_nor, annot=True, fmt=".2%", linewidths=.5, ax=ax,
                    xticklabels=labels, yticklabels=labels, cbar=False)
        plt.ylabel("True class")
        plt.xlabel("Predicted class")
        plt.title("Normalized Confusion Matrix")
        plt.show()

    if return_clss:
        print("========================================= Detailed Metrics =====================================================")
        # Rapport complet (inclut précision, rappel et f1-score par classe)
        report = classification_report(y_test, y_pred, target_names=labels, digits=4)
        print(report)

        # F1 macro et weighted
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')

        print(f"Macro F1-score (unweighted): {f1_macro:.4f}")
        print(f"Weighted F1-score: {f1_weighted:.4f}")
    return y_pred

def plot_result(historique, name_fig="fig", register_plot=False, register_history=True, save_dir="training_results"):
    """
    Affiche et sauvegarde les courbes d'entraînement pour TOUTES les métriques disponibles
    
    Args:
        historique : objet History retourné par model.fit()
        name_fig : nom du fichier pour la figure (sans extension)
        register_plot : bool, sauvegarde la figure si True
        register_history : bool, sauvegarde l'historique complet si True
        save_dir : répertoire de sauvegarde (créé automatiquement)
    
    Returns:
        dict: dictionnaire contenant toutes les métriques
    """
    
    # Créer le répertoire de sauvegarde si nécessaire
    if register_plot or register_history:
        os.makedirs(save_dir, exist_ok=True)
    
    # Récupérer TOUTES les métriques disponibles dans l'historique
    all_metrics = {}
    for key, value in historique.history.items():
        all_metrics[key] = value
    
    # Séparer les métriques d'entraînement et de validation
    train_metrics = {}
    val_metrics = {}
    
    for key, values in all_metrics.items():
        if key.startswith('val_'):
            val_metrics[key[4:]] = values  # enlève 'val_' du nom
        else:
            train_metrics[key] = values
    
    
    # Déterminer le nombre de graphiques
    n_plots = len(train_metrics)
    if n_plots == 0:
        print("Aucune métrique trouvée dans l'historique.")
        return {}
    
    # Créer une grille adaptative
    ncols = 2
    nrows = (n_plots + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6*ncols, 5*nrows))
    
    # Aplatir axes pour un accès facile
    if nrows == 1 and ncols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    epochs = range(1, len(train_metrics[list(train_metrics.keys())[0]]) + 1)
    
    # Pour chaque métrique, créer un graphique
    for idx, (metric_name, train_values) in enumerate(train_metrics.items()):
        ax = axes[idx]
        
        # Courbe d'entraînement
        ax.plot(epochs, train_values, 'b-', label=f'Train {metric_name}', linewidth=2)
        
        # Courbe de validation si disponible
        if metric_name in val_metrics:
            ax.plot(epochs, val_metrics[metric_name], 'r-', label=f'Val {metric_name}', linewidth=2)
        
        
        if metric_name in val_metrics:
            best_val_idx = np.argmax(val_metrics[metric_name]) if 'acc' in metric_name.lower() or 'f1' in metric_name.lower() else np.argmin(val_metrics[metric_name])
            best_val_val = val_metrics[metric_name][best_val_idx]
            ax.axvline(x=best_val_idx + 1, color='red', linestyle='--', alpha=0.5, label=f'Best val: {best_val_val:.4f}')
        
        ax.set_xlabel('Epochs')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name.capitalize()} : Train vs Validation')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    # Cacher les axes inutilisés
    for idx in range(len(train_metrics), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    # Sauvegarde de la figure
    if register_plot:
        fig_path = os.path.join(save_dir, f"{name_fig}.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Figure sauvegardée : {fig_path}")
    
    plt.show()
    
    # ============================================
    # Sauvegarde de l'historique complet
    # ============================================
    if register_history:
        # Préparer les données pour la sauvegarde
        history_dict = {
            'all_metrics': {},
            'summary': {},
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Sauvegarder toutes les métriques
        for key, values in all_metrics.items():
            history_dict['all_metrics'][key] = [float(x) for x in values]
            
            # Résumé
            if 'acc' in key.lower() or 'f1' in key.lower():
                best_idx = np.argmax(values)
                best_val = values[best_idx]
                final_val = values[-1]
            else:
                best_idx = np.argmin(values)
                best_val = values[best_idx]
                final_val = values[-1]
            
            history_dict['summary'][key] = {
                'best_value': float(best_val),
                'best_epoch': int(best_idx) + 1,
                'final_value': float(final_val)
            }
        
        # Sauvegarde en JSON
        json_path = os.path.join(save_dir, f"{name_fig}_history.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(history_dict, f, indent=4, ensure_ascii=False)
        print(f"Historique sauvegardé : {json_path}")
        
        # Sauvegarde en CSV
        csv_path = os.path.join(save_dir, f"{name_fig}_history.csv")
        with open(csv_path, 'w', encoding='utf-8') as f:
            # En-tête
            headers = ['epoch'] + list(all_metrics.keys())
            f.write(','.join(headers) + '\n')
            
            # Données
            max_len = max(len(v) for v in all_metrics.values())
            for i in range(max_len):
                row = [str(i + 1)]
                for key in all_metrics.keys():
                    val = all_metrics[key][i] if i < len(all_metrics[key]) else ''
                    row.append(str(val))
                f.write(','.join(row) + '\n')
        print(f"CSV sauvegardé : {csv_path}")

def prepare_image(img_path, target_size):
    img = image.load_img(img_path, target_size=(target_size,target_size))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  
    img_array /= 255.0  
    return img_array

def predict(img_path, discriminateur,classificateur, target_size=225, list_name = None):
    img_array, img= prepare_image(img_path , target_size)
    est_connue = discriminateur.predict(img_array)
    if est_connue[0][0] >=0.5:
        predict_proba = classificateur.predict(img_array)
        y_pred = np.argmax(predict_proba, axis=1)
        result = np.max(predict_proba[0]*100)
        plt.imshow(img)
        plt.title(f"{list_name[y_pred[0]]} avec une précision de {result}%")
    else:
        result = est_connue[0][0]*100
        plt.imshow(img)
        plt.title(f"Classe inconnue {result}%")
        
def load_model_(path_model):
    list_name = ['Can', 'Organic', 'Plastic', 'Textile', 'Glass']
    model = load_model(path_model)
    return model, list_name






def compare_models_full(model1, model2, X_test, y_test, class_names):
    def evaluate_model(model, X, y_true):
        start = time.time()
        y_prob = model.predict(X)
        end = time.time()
        y_pred = np.argmax(y_prob, axis=1)
        total_time = end - start
        fps = len(X) / total_time

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
        conf_matrix = confusion_matrix(y_true, y_pred)

        return {
            'acc': acc,
            'f1': f1,
            'fps': fps,
            'time': total_time,
            'y_pred': y_pred,
            'report': report,
            'conf_matrix': conf_matrix,
            'params': model.count_params()
        }

    res1 = evaluate_model(model1, X_test, y_test)
    res2 = evaluate_model(model2, X_test, y_test)

    # --- Impression détaillée
    print("==========  Détails - Modèle 1 ==========")
    print(res1['report'])
    print(f"Total params : {res1['params']}")
    print(f"Accuracy : {res1['acc']:.4f} | F1-macro : {res1['f1']:.4f}")
    print(f"Prediction time : {res1['time']:.2f}s | FPS : {res1['fps']:.2f}")

    print("\n==========  Détails - Modèle 2 ==========")
    print(res2['report'])
    print(f"Total params : {res2['params']}")
    print(f"Accuracy : {res2['acc']:.4f} | F1-macro : {res2['f1']:.4f}")
    print(f"Prediction time : {res2['time']:.2f}s | FPS : {res2['fps']:.2f}")

    # --- Matrices de confusion
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(res1['conf_matrix'].astype('float') / res1['conf_matrix'].sum(axis=1)[:, np.newaxis],
                annot=True, fmt=".2%", xticklabels=class_names, yticklabels=class_names,
                ax=axs[0], cmap="Blues")
    axs[0].set_title("Confusion Matrix - Model 1")

    sns.heatmap(res2['conf_matrix'].astype('float') / res2['conf_matrix'].sum(axis=1)[:, np.newaxis],
                annot=True, fmt=".2%", xticklabels=class_names, yticklabels=class_names,
                ax=axs[1], cmap="Greens")
    axs[1].set_title("Confusion Matrix - Model 2")
    plt.tight_layout()
    plt.show()

    # --- Résumé comparatif
    print("\n==========  Résumé comparatif ==========")
    print(f"{'':<18} | {'Model 1':<12} | {'Model 2':<12}")
    print(f"{'Accuracy':<18} | {res1['acc']:<12.4f} | {res2['acc']:<12.4f}")
    print(f"{'Macro F1-score':<18} | {res1['f1']:<12.4f} | {res2['f1']:<12.4f}")
    print(f"{'Params':<18} | {res1['params']:<12} | {res2['params']:<12}")
    print(f"{'Time (s)':<18} | {res1['time']:<12.2f} | {res2['time']:<12.2f}")
    print(f"{'FPS':<18} | {res1['fps']:<12.2f} | {res2['fps']:<12.2f}")


def concat_datasets(x_train, y_train, x_test, y_test,
                    x_train1, y_train1, x_test1, y_test1):
    """
    Concatène deux jeux de données (train/test).
    Les arrays doivent avoir des dimensions compatibles.
    """
    # Concat train
    X_train_full = np.concatenate([x_train, x_train1], axis=0)
    y_train_full = np.concatenate([y_train, y_train1], axis=0)

    # Concat test
    X_test_full = np.concatenate([x_test, x_test1], axis=0)
    y_test_full = np.concatenate([y_test, y_test1], axis=0)

    return X_train_full, y_train_full, X_test_full, y_test_full




def select_data(X_train, y_train, n, shuffle=True, random_state=None):
   
    if random_state is not None:
        np.random.seed(random_state)

    classes = np.unique(y_train)
    X_selected, y_selected = [], []

    for cls in classes:
        idx = np.where(y_train == cls)[0]
        if len(idx) < n:
            raise ValueError(f"Classe {cls} n'a que {len(idx)} échantillons, inférieur à n={n}.")
        chosen_idx = np.random.choice(idx, n, replace=False)
        X_selected.append(X_train[chosen_idx])
        y_selected.append(y_train[chosen_idx])

    X_selected = np.concatenate(X_selected, axis=0)
    y_selected = np.concatenate(y_selected, axis=0)

    if shuffle:
        perm = np.random.permutation(len(y_selected))
        X_selected, y_selected = X_selected[perm], y_selected[perm]

    return X_selected, y_selected

class JEPAMaskingConfig:
    """
    Configuration du masquage selon l'article I-JEPA.
    
    Référence: Section 3 - Method, et Appendix C
    """
    
    def __init__(self, img_size=224, patch_size=16):
        """
        Args:
            img_size: taille de l'image en pixels (H=W)
            patch_size: taille de chaque patch en pixels
        """
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.grid_size = img_size // patch_size
        
        # Paramètres des cibles (target blocks) - Section 3
        self.target_scale_min = 0.15      # Échelle min des blocs cibles
        self.target_scale_max = 0.20      # Échelle max des blocs cibles
        self.target_aspect_min = 0.75     # Ratio d'aspect min
        self.target_aspect_max = 1.5      # Ratio d'aspect max
        self.num_targets = 4              # Nombre de blocs cibles (M=4)
        
        # Paramètres du contexte (context block) - Section 3
        self.context_scale_min = 0.85     # Échelle min du bloc contexte
        self.context_scale_max = 1.0      # Échelle max du bloc contexte
        self.context_aspect = 1.0         # Ratio d'aspect du contexte (carré)



class JEPAMaskGenerator:
    """
    Générateur de masques pour I-JEPA.
    Produit les indices des patches pour le contexte et les cibles.
    
    Référence: Section 3 - Method, "Masking strategy"
    """
    
    def __init__(self, config):
        self.config = config
        
    def _random_block(self, scale_min, scale_max, aspect_min=None, aspect_max=None, aspect_fixed=None):
       
        grid = self.config.grid_size
        
        # Échelle en nombre de patches
        scale = random.uniform(scale_min, scale_max)
        area = scale * scale * grid * grid  # surface en nombre de patches
        
        # Ratio d'aspect
        if aspect_fixed is not None:
            aspect = aspect_fixed
        else:
            aspect = random.uniform(aspect_min, aspect_max)
        
        # Dimensions en patches
        w_patches = int(round(math.sqrt(area * aspect)))
        h_patches = int(round(math.sqrt(area / aspect)))
        
        # Ajuster pour rester dans les limites
        w_patches = min(w_patches, grid - 1)
        h_patches = min(h_patches, grid - 1)
        
        if w_patches < 1:
            w_patches = 1
        if h_patches < 1:
            h_patches = 1
        
        # Position aléatoire
        x = random.randint(0, grid - w_patches)
        y = random.randint(0, grid - h_patches)
        
        return x, y, w_patches, h_patches
    
    def _block_to_indices(self, x, y, w, h):
        
        grid = self.config.grid_size
        indices = []
        for row in range(y, y + h):
            for col in range(x, x + w):
                idx = row * grid + col
                indices.append(idx)
        return indices
    
    def generate_masks(self, batch_size):
        
        context_masks = []
        target_masks_list = []
        
        for _ in range(batch_size):
            # 1. Générer les blocs cibles (M=4)
            target_blocks = []
            target_blocks_coords = []
            
            for _ in range(self.config.num_targets):
                x, y, w, h = self._random_block(
                    self.config.target_scale_min,
                    self.config.target_scale_max,
                    self.config.target_aspect_min,
                    self.config.target_aspect_max
                )
                target_blocks.append(self._block_to_indices(x, y, w, h))
                target_blocks_coords.append((x, y, w, h))
            
            # 2. Générer le bloc contexte
            ctx_x, ctx_y, ctx_w, ctx_h = self._random_block(
                self.config.context_scale_min,
                self.config.context_scale_max,
                aspect_fixed=self.config.context_aspect
            )
            context_idx = set(self._block_to_indices(ctx_x, ctx_y, ctx_w, ctx_h))
            
            # 3. Supprimer les chevauchements avec les cibles
            #    (comme décrit dans l'article: "remove any overlapping regions from the context block")
            for target_block in target_blocks:
                context_idx = context_idx - set(target_block)
            
            # Sauvegarder
            context_masks.append(list(context_idx))
            target_masks_list.append(target_blocks)
        
        return context_masks, target_masks_list
    
    def visualize_masks(self, image, context_mask, target_masks):
       
        grid = self.config.grid_size
        patch_size = self.config.patch_size
        h, w = image.shape[:2]
        
        # Créer une copie
        vis = image.copy().astype(np.uint8)
        
        # Couleurs pour les cibles (RGB)
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
        
        # Dessiner les cibles
        for i, target_mask in enumerate(target_masks):
            color = colors[i % len(colors)]
            for idx in target_mask:
                row = idx // grid
                col = idx % grid
                y1 = row * patch_size
                x1 = col * patch_size
                y2 = y1 + patch_size
                x2 = x1 + patch_size
                # Mélanger l'image originale avec la couleur
                vis[y1:y2, x1:x2] = vis[y1:y2, x1:x2] * 0.4 + np.array(color) * 0.6
        
        # Dessiner le contexte (en cyan)
        for idx in context_mask:
            row = idx // grid
            col = idx % grid
            y1 = row * patch_size
            x1 = col * patch_size
            y2 = y1 + patch_size
            x2 = x1 + patch_size
            vis[y1:y2, x1:x2] = vis[y1:y2, x1:x2] * 0.4 + np.array([0, 255, 255]) * 0.6
        
        return vis

class IJEPADataGenerator(tf.keras.utils.Sequence):
    """
    Générateur de données pour I-JEPA avec masquage.
    Retourne ((images_masquees, target_data), labels)
    """
    
    def __init__(self, images, labels=None, batch_size=32, img_size=64, patch_size=8, shuffle=True):
        self.images = images.astype(np.float32)
        self.labels = labels
        self.batch_size = batch_size
        self.img_size = img_size
        self.patch_size = patch_size
        self.shuffle = shuffle
        
        self.config = JEPAMaskingConfig(img_size, patch_size)
        self.mask_generator = JEPAMaskGenerator(self.config)
        
        self.indices = np.arange(len(images))
        if shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        return int(np.ceil(len(self.images) / self.batch_size))
    
    def _apply_mask_to_image(self, image, context_mask):
        grid = self.config.grid_size
        patch_size = self.config.patch_size
        masked_image = np.zeros_like(image, dtype=np.float32)
        
        for idx in context_mask:
            row = idx // grid
            col = idx % grid
            y1 = row * patch_size
            x1 = col * patch_size
            y2 = y1 + patch_size
            x2 = x1 + patch_size
            masked_image[y1:y2, x1:x2] = image[y1:y2, x1:x2]
        
        return masked_image
    
    def _pad_target_masks(self, target_masks_list):
        """
        Pad les target_masks pour qu'ils aient tous la même taille.
        """
        batch_size = len(target_masks_list)
        if batch_size == 0:
            return np.array([]), np.array([])
        
        num_targets = len(target_masks_list[0])
        
        #  Calcul correct de max_patches
        max_patches = 0
        for masks in target_masks_list:
            for m in masks:
                max_patches = max(max_patches, len(m))
        
        #  Si max_patches est trop petit, au moins 4
        max_patches = max(max_patches, 4)
        
        padded_masks = np.zeros((batch_size, num_targets, max_patches), dtype=np.int32)
        mask_lengths = np.zeros((batch_size, num_targets), dtype=np.int32)
        
        for i, masks in enumerate(target_masks_list):
            for j, m in enumerate(masks):
                length = len(m)
                mask_lengths[i, j] = length
                if length > 0:
                    padded_masks[i, j, :length] = m
        
        return padded_masks, mask_lengths
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_images = self.images[batch_indices]
        batch_images = batch_images / 255.0
        
        # Générer les masques
        context_masks, target_masks_list = self.mask_generator.generate_masks(len(batch_images))
        
        # Appliquer les masques
        context_images = []
        for i, img in enumerate(batch_images):
            masked = self._apply_mask_to_image(img, context_masks[i])
            context_images.append(masked)
        context_images = np.array(context_images, dtype=np.float32)
        
        # Pad les target_masks
        padded_target_masks, mask_lengths = self._pad_target_masks(target_masks_list)
        
        # Convertir en tenseurs
        padded_target_masks = tf.constant(padded_target_masks, dtype=tf.int32)
        mask_lengths = tf.constant(mask_lengths, dtype=tf.int32)
        target_data = (padded_target_masks, mask_lengths)
        
        if self.labels is not None:
            batch_labels = self.labels[batch_indices]
            batch_labels = tf.constant(batch_labels, dtype=tf.int32)
            return (context_images, target_data), batch_labels
        else:
            dummy_labels = tf.zeros(len(batch_indices), dtype=tf.int32)
            return (context_images, target_data), dummy_labels
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def visualize_batch(self, batch_idx=0, save_path=None):
        
        
        # Récupérer le batch
        (context_images, target_data), labels = self[batch_idx]
        padded_target_masks, mask_lengths = target_data
        
        # Récupérer les images originales
        batch_indices = self.indices[batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size]
        original_images = self.images[batch_indices].astype(np.uint8)
        
        #  Vérifier le type et convertir si nécessaire
        if hasattr(context_images, 'numpy'):
            context_images = context_images.numpy()
        if hasattr(padded_target_masks, 'numpy'):
            padded_target_masks = padded_target_masks.numpy()
        if hasattr(mask_lengths, 'numpy'):
            mask_lengths = mask_lengths.numpy()
        
        # Reconstruire les target_masks
        target_masks_list = []
        for i in range(padded_target_masks.shape[0]):
            masks = []
            for j in range(padded_target_masks.shape[1]):
                length = mask_lengths[i, j] if mask_lengths.ndim > 1 else mask_lengths[j]
                if length > 0:
                    masks.append(padded_target_masks[i, j, :length].tolist())
                else:
                    masks.append([])
            target_masks_list.append(masks)
        
        # Générer les masques de contexte pour la visualisation
        context_masks, _ = self.mask_generator.generate_masks(len(original_images))
        
        n_images = min(4, len(context_images))
        fig, axes = plt.subplots(n_images, 4, figsize=(10, 2 * n_images))
        
        if n_images == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_images):
            # 1. Image originale
            axes[i, 0].imshow(original_images[i])
            axes[i, 0].set_title(f"Originale {i+1}")
            axes[i, 0].axis('off')
            
            # 2. Image masquée (seul le contexte)
            axes[i, 1].imshow(context_images[i])
            axes[i, 1].set_title(f"Contexte masqué")
            axes[i, 1].axis('off')
            
            # 3. Visualisation des cibles
            vis_targets = self.mask_generator.visualize_masks(
                original_images[i],
                [],  # pas de contexte pour cette visualisation
                target_masks_list[i]
            )
            axes[i, 2].imshow(vis_targets)
            axes[i, 2].set_title(f"Cibles (4 blocs)")
            axes[i, 2].axis('off')
            
            # 4. Visualisation complète (contexte + cibles)
            vis_full = self.mask_generator.visualize_masks(
                original_images[i],
                context_masks[i],
                target_masks_list[i]
            )
            axes[i, 3].imshow(vis_full)
            axes[i, 3].set_title(f"Contexte + Cibles")
            axes[i, 3].axis('off')
        
        plt.tight_layout()
            
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure sauvegardée: {save_path}")
        
        plt.show()
        plt.close(fig)




