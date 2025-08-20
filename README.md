# Projet TP1 - MNIST avec TensorFlow

## 📌 Description
Ce projet est réalisé dans le cadre du cours **420-013-XX - Intelligence Artificielle 1**.
L’objectif est de développer une application simple en **Python/TensorFlow** capable de reconnaître les chiffres manuscrits (dataset **MNIST**).

Le projet comprend :
- Entraînement d’un modèle CNN sur MNIST
- Sauvegarde du modèle (`tp1.keras`)
- Génération de graphiques (`accuracy`, `loss`)
- Prédictions sur des exemples spécifiques
- Tests unitaires pour valider le modèle
- Rapport PDF documentant le processus

---

## 📂 Structure du projet
```bash
tp1-mnist/
├── data/
│ ├── input/ # Données brutes (optionnel) — MNIST est chargé automatiquement via TensorFlow/Keras; ce dossier n'est requis que pour des données personnalisées
│ ├── output/
│ │ ├── graphs/ # Graphiques (accuracy.png, loss.png)
│ │ ├── predictions/ # Prédictions générées
│ │ └── logs/ # Logs d’entraînement
│ └── tp1.keras # Modèle sauvegardé
│
├── src/
│ ├── init.py
│ ├── model.py # Architecture CNN
│ ├── utils.py # Fonctions utilitaires (plots, logs)
│ ├── train.py # Script d’entraînement
│ └── predict.py # Script de prédiction
│
├── tests/
│ ├── init.py
│ └── test_model.py # Tests unitaires (shape, range, accuracy)
│
├── requirements.txt # Dépendances du projet
└── README.md # Documentation
```

## ⚙️ Installation

1. **Cloner le projet :**
```bash
git clone https://github.com/votre-repo/tp1-mnist.git

cd tp1-mnist

python -m venv .venv

source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

2. **Installer les dépendances :**
```bash
pip install -r requirements.txt
```

## 🚀 Utilisation

1. **Entraînement :**
Lancer l’entraînement et générer le modèle + graphiques :

```bash
python src/train.py
```

Résultats générés :

* ```data/tp1.keras``` (modèle sauvegardé)
* ```data/output/graphs/accuracy.png```
* ```data/output/graphs/loss.png```
* ```data/output/logs/training_log.txt```

2. **Prédictions :**
Lancer des prédictions sur les indices demandés (1, 6, 3513, 10123, 43213) :

```bash
python src/predict.py
```
Résultats générés dans ``` data/output/predictions/. ```

3. **Tests unitaires :**

Exécuter la suite de tests :
 ```bash
python -m unittest discover tests
```

## 📊 Résultats attendus
* Accuracy du modèle ~99% sur le test set.
* Graphiques montrant une amélioration claire de la performance.
* Prédictions correctes sur la majorité des échantillons choisis.

## 👤 Auteurs

Projet réalisé par  **[ El hadi Felfoul & ..... ]**

Cours :  **Intelligence Artificielle 1 - Steve Lévesque**