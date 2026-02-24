# Classification et Détection d'anomalie par un  Variational Autoencoder (VAE) avec Blocs Résiduels

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

##  Table des matières
- [Description](#description)
- [Architecture](#architecture)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Configuration](#configuration)
- [Entraînement](#entraînement)
- [Visualisation](#visualisation)
- [Dépannage](#dépannage)
- [Références](#références)

##  Description

Ce projet implémente un **Variational Autoencoder (VAE)** avec des **blocs résiduels** pour la reconstruction et la génération d'images. L'architecture utilise des connexions résiduelles pour faciliter l'apprentissage de représentations profondes tout en évitant les problèmes de gradient vanishing.

### Objectifs
- Reconstruction d'images de haute qualité
- Génération de nouvelles images à partir de l'espace latent
- Visualisation et exploration de l'espace latent
- Robustesse à l'overfitting via régularisation

##  Architecture

### Vue d'ensemble