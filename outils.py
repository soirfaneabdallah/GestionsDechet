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
import pickle, zipfile, io
from tensorflow.keras.callbacks import Callback
import tensorflow as tf
sns.set_theme()
class Load_data:
    """
    Classe pour charger, préparer et gérer des ensembles d'images pour l'entraînement de modèles ML/DL.
    """

    def __init__(self, 
                 path_list: list[str] = None, 
                 extension: tuple[str, ...] = ('.png', '.jpg', '.jpeg', '.JPG'),
                 image_shape: tuple[int, int, int] = (240, 240, 3)) -> None:
        """
        Initialise la classe.
        
        :param path_list: liste des dossiers contenant les images
        :param extension: extensions des fichiers supportés
        :param image_shape: tuple (H, W, C) -> hauteur, largeur, canaux (1 = grayscale, 3 = RGB)
        """
        self.path_list_     = path_list
        self.extension_     = extension
        self.name_label_    = []
        self.data_          = None
        self.data_label_    = None
        self.original_data_ = None
        self.image_shape_   = image_shape  # (H, W, C)

    # ---------------------- Chargement ----------------------
    def load(self) -> None:
        """
        Charge et prétraite les images : redimensionnement + conversion en RGB/Grayscale.
        Stocke les données dans un DataFrame pandas.
        """
        image_data, labels = [], []
        H, W, C = self.image_shape_

        for folder in self.path_list_:
            label = os.path.basename(os.path.normpath(folder))
            self.name_label_.append(label)

            for fichier in tqdm(os.listdir(folder), desc=f"Chargement: {label}"):
                if fichier.lower().endswith(self.extension_):
                    try:
                        path_image = os.path.join(folder, fichier)
                        img = Image.open(path_image)

                        # Convertir selon le nombre de canaux atenu
                        if C == 1:
                            img = img.convert("L")  # grayscale
                        elif C == 3:
                            img = img.convert("RGB")
                        else:
                            raise ValueError("Le nombre de canaux doit être 1 (gris) ou 3 (RGB).")

                        # Redimensionnement automatique
                        img = img.resize((W, H))
                        img = np.array(img)

                        # Ajout d'une dimension pour grayscale
                        if C == 1:
                            img = np.expand_dims(img, axis=-1)

                        image_data.append(img)
                        labels.append(label)

                    except (OSError, ValueError):
                        continue

        self.data_ = pd.DataFrame({'Image': image_data, 'Label': labels})
        labels_ = self.shuffle(image_data, labels)
        self.data_label_ = pd.DataFrame({'Label': labels_})
        self.copy()

    # ---------------------- Mélange ----------------------
    def shuffle(self, images: list, labels: list) -> None:
        """ Mélange aléatoirement les images et labels. """
        pairs = list(zip(images, labels))
        random.seed(0)
        random.shuffle(pairs)
        image, label = zip(*pairs)
        self.data_["Image"], self.data_["Label"] = image, label
        return label

    # ---------------------- Sauvegarde état ----------------------
    def copy(self) -> None:
        """ Sauvegarde une copie des données originales. """
        self.original_data_ = copy.deepcopy(self.data_)

    def restore_data(self) -> None:
        """ Restaure les données originales. """
        if self.original_data_ is not None:
            self.data_ = copy.deepcopy(self.original_data_)
    # ---------------------- Ajout de nouvelles données ----------------------
    def add_data(self, folder: str) -> None:
        """
        Ajoute des images depuis un nouveau dossier au dataset existant.
        
        :param folder: chemin vers le dossier contenant les nouvelles images
        """
        H, W, C = self.image_shape_
        label = os.path.basename(os.path.normpath(folder))

        # Si la classe est nouvelle, on l'ajoute
        if label not in self.name_label_:
            self.name_label_.append(label)

        image_data, labels = [], []
        for fichier in tqdm(os.listdir(folder), desc=f"Ajout: {label}"):
            if fichier.lower().endswith(self.extension_):
                try:
                    path_image = os.path.join(folder, fichier)
                    img = Image.open(path_image)

                    # Conversion RGB ou grayscale selon shape
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

                except (OSError, ValueError):
                    continue

        # Fusion avec le dataset existant
        if self.data_ is None:
            self.data_ = pd.DataFrame({'Image': image_data, 'Label': labels})
        else:
            new_df = pd.DataFrame({'Image': image_data, 'Label': labels})
            self.data_ = pd.concat([self.data_, new_df], ignore_index=True)

        # Mise à jour des labels globaux
        self.data_label_ = pd.DataFrame({'Label': self.data_["Label"].tolist()})
        self.copy()

    # ---------------------- Visualisation ----------------------
    def plot(self, view_code: bool = False, name_fig: str = "fig", register: bool = False) -> None:
        """
        Affiche un échantillon aléatoire d'images avec leurs labels.
        """
        plt.figure(figsize=(12, 9))
        final = np.random.randint(0, self.data_.shape[0])
        indices = list(range(max(0, final - 10),final))
        title_source = self.data_["Label"] if view_code else self.data_label_["Label"]

        for i, idx in enumerate(indices, start=1):
            plt.subplot(4, 5, i)
            plt.imshow(self.data_["Image"][idx])
            plt.title(title_source[idx])
            plt.axis("off")

        plt.tight_layout()
        if register:
            plt.savefig(f"{name_fig}.png")
        plt.show()

    # ---------------------- Traitement des labels ----------------------
    def encodage(self) -> None:
        """
        Encode les labels en entiers.
        """
        if isinstance(self.data_["Label"].iloc[0], str):
            code = {name: i for i, name in enumerate(self.name_label_)}
            self.data_["Label"] = self.data_["Label"].map(code)

    # ---------------------- Compression ----------------------
    def compress(self, image: np.ndarray, test: bool = True, k: int = 100, threshold_kb: float = 200.0) -> np.ndarray:
        """
        Compresse une image via décomposition SVD si elle dépasse un seuil en Ko.
        """
        image_size_kb = image.nbytes / 1024
        if image_size_kb > threshold_kb:
            compressed_channels = []
            for i in range(3):  
                U, S, Vt = np.linalg.svd(image[:, :, i], full_matrices=False)
                Sk = np.diag(S[:k])
                compressed = np.dot(U[:, :k], np.dot(Sk, Vt[:k, :]))
                compressed_channels.append(compressed)

            compressed_img = np.stack(compressed_channels, axis=2)
            compressed_img = np.clip(compressed_img, 0, 255).astype(np.uint8)
            resized_img = Image.fromarray(compressed_img).resize((self.image_size_, self.image_size_))

            if test:
                fig, ax = plt.subplots(1, 2, figsize=(12, 6))
                ax[0].imshow(resized_img)
                ax[0].set_title("Image compressée")
                ax[1].imshow(image)
                ax[1].set_title("Image originale")
                plt.show()

            return np.array(resized_img) / 255.0

        return np.array(image) / 255.0

    def compress_data(self, threshold_kb: float = 200.0) -> None:
        """
        Compresse toutes les images de l'ensemble de données.
        """
        self.data_["Image"] = [
            self.compress(img, test=False, threshold_kb=threshold_kb)
            for img in tqdm(self.data_["Image"], desc="Compression")
        ]

       # ---------------------- Sauvegarde / Chargement ----------------------
    def save(self, zip_path: str = "dataset.zip", internal_name: str = "loader.pkl") -> None:
        """
        Sauvegarde l'objet dans une archive zip (y compris name_label_).
        """
        backup = self.data_
        self.data_ = None  # on évite de sauvegarder trop de données brutes dans le pickle
        buffer = io.BytesIO()
        pickle.dump(self, buffer)
        self.data_ = backup  # restauration avant écriture
        buffer.seek(0)

        with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(internal_name, buffer.read())

    @staticmethod
    def load_object(zip_path: str = "dataset.zip", internal_name: str = "loader.pkl") -> "Load_data":
        """
        Charge un objet LoadData sauvegardé dans une archive zip.
        """
        with zipfile.ZipFile(zip_path, 'r') as zf:
            with zf.open(internal_name) as f:
                obj = pickle.load(f)

                # Restaurer les données originales si dispo
                if obj.original_data_ is not None:
                    obj.data_ = copy.deepcopy(obj.original_data_)

                # Vérification que les structures sont prêtes pour add_data()
                if not hasattr(obj, "name_label_"):
                    obj.name_label_ = []
                if not hasattr(obj, "data_label_"):
                    obj.data_label_ = pd.DataFrame({'Label': obj.data_["Label"].tolist()}) if obj.data_ is not None else None

                return obj


    # ---------------------- Split train/test ----------------------
    def create_data(self, test_size: float = 0.2, shuffle: bool = True):
        """
        Crée les ensembles d'entraînement et de test.
        """
        data_train, data_test = train_test_split(self.data_, test_size=test_size, random_state=42, shuffle=shuffle)
        
        X_train = np.array(data_train['Image'].tolist(), dtype=np.int32)
        Y_train = np.array(data_train['Label'].tolist(), dtype=np.int32)
        X_test = np.array(data_test['Image'].tolist(), dtype=np.int32)
        Y_test = np.array(data_test['Label'].tolist(), dtype=np.int32)

        return (X_train, Y_train), (X_test, Y_test)



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

def plot_result(historique, name_fig = "fig", register = False):
    acc = historique.history['accuracy']
    val_acc = historique.history['val_accuracy']
    loss = historique.history['loss']
    val_loss = historique.history['val_loss']

    epochs = range(len(acc))
    fig,ax =plt.subplots(nrows=1,ncols=2,figsize=(16,6))
    ax[0].plot(epochs, acc, 'b', label='Train accuracy')
    ax[0].plot(epochs, val_acc, 'r', label='Test accuracy')
    ax[0].set_title("Accuracy: Train and Test")
    ax[0].legend(loc=0)
    ax[1].plot(epochs, loss, 'b', label='Training loss')
    ax[1].plot(epochs, val_loss, 'r', label='Test loss')
    ax[1].legend(loc=0)
    ax[1].set_title("Loss : Train and Test")
    if register:
      plt.savefig(f"{name_fig}.png")
    plt.show()

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
        
def load_modele(path_model):
    list_name = ['Canette', 'Organique','Plastique','Textile','Verre']
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
                annot=True, fmt=".2f", xticklabels=class_names, yticklabels=class_names,
                ax=axs[0], cmap="Blues")
    axs[0].set_title("Confusion Matrix - Model 1")

    sns.heatmap(res2['conf_matrix'].astype('float') / res2['conf_matrix'].sum(axis=1)[:, np.newaxis],
                annot=True, fmt=".2f", xticklabels=class_names, yticklabels=class_names,
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





class PlotGANCallback(Callback):
    def __init__(self, generator, latent_dim=128, n=5, every_n_epochs=1, rescale=True, show=True, save_dir=None):
        super().__init__()
        self.generator = generator
        self.latent_dim = latent_dim
        self.n = n
        self.every_n_epochs = every_n_epochs
        self.rescale = rescale
        self.show = show
        self.save_dir = save_dir

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.every_n_epochs != 0:
            return

        # Génération
        random_latent_vectors = tf.random.normal(shape=(self.n, self.latent_dim))
        generated_images = self.generator(random_latent_vectors, training=False).numpy()

        # Ajustement
        if generated_images.shape[-1] == 1:
            generated_images = generated_images.squeeze(-1)

        if self.rescale:
            generated_images = (generated_images + 1.0) / 2.0

        # Plot
        plt.figure(figsize=(15, 3))
        for i in range(self.n):
            plt.subplot(1, self.n, i + 1)
            plt.imshow(np.clip(generated_images[i], 0, 1))
            plt.axis("off")
        plt.suptitle(f"Images générées - Époque {epoch + 1}")

        # Sauvegarde et affichage
        if self.save_dir:
            plt.savefig(f"{self.save_dir}/generated_epoch_{epoch+1}.png")

        if self.show:
            plt.show(block=False)
            plt.pause(0.001)

        plt.close()


       
