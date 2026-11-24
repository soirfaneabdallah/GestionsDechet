"""Inférence sur une image isolée (pipeline discriminateur + classificateur)."""
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image


def prepare_image(img_path, target_size):
    img = image.load_img(img_path, target_size=(target_size,target_size))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  
    img_array /= 255.0  
    return img_array, img

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
