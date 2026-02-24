# Classification et Détection d'anomalie (Déchet)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Table des matières
- [Description du projet](#description-du-projet)
- [Architecture du système](#architecture-du-système)
- [Installation](#installation)
- [Structure du projet](#structure-du-projet)
- [Utilisation](#utilisation)
- [Configuration](#configuration)
- [Résultats attendus](#résultats-attendus)
- [Contributions](#contributions)
- [Licence](#licence)

## 🎯 Description du projet

Ce projet implémente un **système profond de gestion des déchets** basé sur la vision par ordinateur. L'objectif est double :

1. **Classification** : Identifier automatiquement le type de déchet parmi plusieurs catégories (plastique, verre, papier, métal, carton, etc.)

2. **Détection d'anomalies** : Distinguer les images qui n'appartiennent à aucune classe connue du modèle. Ces "anomalies" peuvent représenter des déchets non conformes, des objets indésirables dans la chaîne de tri, ou de nouvelles catégories non encore rencontrées.

### Problématique

Dans un système réel de gestion des déchets, il est crucial de :
- ✅ Trier correctement les déchets connus
- 🔍 Identifier les objets inconnus qui pourraient perturber la chaîne de tri
- ♻️ Adapter le système à de nouvelles catégories au fil du temps

Notre approche combine un **Variational Autoencoder (VAE)** avec des **blocs résiduels** pour créer un espace latent structuré, permettant à la fois une classification précise et une détection fiable des outliers.

## 🏗 Architecture du système

### Vue d'ensemble
