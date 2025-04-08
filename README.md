# Application d'Analyse d'Équité en Apprentissage Automatique

Cette application Streamlit permet d'analyser et de corriger les biais dans les modèles d'apprentissage automatique.

## Fonctionnalités

- Upload de fichiers CSV
- Sélection de la variable cible et des caractéristiques sensibles
- Analyse des performances du modèle
- Mesures d'équité (Disparate Impact, Statistical Parity Difference, etc.)
- Techniques de correction des biais (Reweighing)
- Visualisations interactives des résultats

## Installation

1. Clonez ce dépôt
2. Installez les dépendances :
```bash
pip install -r requirements.txt
```

## Utilisation

1. Lancez l'application :
```bash
streamlit run app.py
```

2. Dans l'interface :
   - Uploader votre fichier CSV
   - Sélectionner la variable cible
   - Choisir les caractéristiques sensibles
   - Sélectionner le modèle et les techniques d'équité
   - Analyser les résultats

## Exemple de données

L'application fonctionne avec n'importe quel jeu de données CSV contenant :
- Une variable cible binaire (0/1)
- Des caractéristiques sensibles (genre, race, etc.)
- Des variables prédictives

## Métriques d'équité

- Disparate Impact
- Statistical Parity Difference
- Mean Difference
- Taux de sélection par groupe
- Matrices de confusion par groupe

## Techniques d'équité

- Reweighing : Rééquilibrage des poids des échantillons pour corriger les biais
- Plus de techniques seront ajoutées dans les futures versions 