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

1. **Classification** : Identifier automatiquement le type de déchet parmi plusieurs catégories (Canette, Organique, Plastique, Textile, Verre.)

2. **Détection d'anomalies** : Distinguer les images qui n'appartiennent à aucune classe connue du modèle. Ces "anomalies" peuvent représenter des déchets non conformes, des objets indésirables dans la chaîne de tri, ou de nouvelles catégories non encore rencontrées.

### Problématique

Dans un système réel de gestion des déchets, il est crucial de :
 ✅ Trier correctement les déchets connus
 🔍 Identifier les objets inconnus qui pourraient perturber la chaîne de tri
 ♻️ Adapter le système à de nouvelles catégories au fil du temps

Notre approche combine un **Variational Autoencoder (VAE)** avec des **blocs résiduels** pour créer un espace latent structuré, permettant à la fois une classification précise et une détection fiable des outliers.

## 🏗 Architecture du système

### Vue d'ensemble
```tikz
\begin{document}
\begin{tikzpicture}[
    block/.style={
        rectangle, rounded corners=5pt, draw=blue!70!black, fill=blue!5,
        minimum width=3cm, minimum height=1cm, text centered, font=\small\bfseries
    },
    class/.style={
        rectangle, rounded corners=3pt, draw=green!60!black, fill=green!5,
        minimum width=2.2cm, minimum height=0.8cm, text centered, font=\small
    },
    arrow/.style={-{Latex[length=2mm]}, thick},
    title/.style={font=\large\bfseries, text=blue!80!black}
]

% Titre
\node[title] at (0, 5) (titre) {SYSTÈME DE CLASSIFICATION ET DÉTECTION D'ANOMALIE};
\node[font=\small] at (0, 4.5) (sous titre) {(Canette, Organique, Plastique, Textile, Verre)};

% Entrée
\node[block] at (0, 3) (input) {Image de déchet};

% Encodeur
\node[block] at (-2, 1) (encoder) {Encodeur Résiduel};
\node[block] at (2, 1) (decoder) {Décodeur Résiduel};

% Espace latent
\node[ellipse, draw=purple!70!black, fill=purple!5,
      minimum width=4cm, minimum height=1.5cm] at (0, 1) (latent) 
      {Espace latent $z$};

% Branches
\node[class] at (-3, -1) (classif) {Classifieur};
\node[class] at (0, -1) (recon) {Reconstruction};
\node[class] at (3, -1) (anomaly) {Détection anomalie};

% Classes
\node[class, fill=blue!5] at (-3, -2.5) (classes) {
    \begin{tabular}{c}
        Canette\\ Organique\\ Plastique\\ Textile\\ Verre
    \end{tabular}
};

% Anomalie
\node[class, fill=red!5] at (3, -2.5) (result) {
    \begin{tabular}{c}
        Score $< \theta$: Normal\\
        Score $\geq \theta$: ANOMALIE
    \end{tabular}
};

% Flèches
\draw[arrow] (input) -- (encoder);
\draw[arrow] (input) -- (decoder);
\draw[arrow] (encoder) -- (latent);
\draw[arrow] (latent) -- (decoder);
\draw[arrow] (latent) -- (classif);
\draw[arrow] (latent) -- (recon);
\draw[arrow] (latent) -- (anomaly);
\draw[arrow] (classif) -- (classes);
\draw[arrow] (anomaly) -- (result);

\end{tikzpicture}
\end{document}
```