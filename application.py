import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
import os
import io
import traceback

# Importer les bibliothèques fairlearn si elles sont disponibles
try:
    from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
    from fairlearn.metrics import MetricFrame, selection_rate
    from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds
    FAIRLEARN_AVAILABLE = True
except ImportError:
    FAIRLEARN_AVAILABLE = False
    st.warning("La bibliothèque fairlearn n'est pas installée. Certaines fonctionnalités d'équité avancées ne seront pas disponibles.")

# Configuration de la page
st.set_page_config(
    page_title="Analyse d'Équité Algorithmique",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stTitle {
        color: white;
        background-color: #4C4CFF;
        padding: 1rem;
        border-radius: 5px;
    }
    .stSubheader {
        color: #4C4CFF;
    }
    </style>
    """, unsafe_allow_html=True)

# Fonctions d'équité
def demographic_parity(y_pred, sensitive_attr, sensitive_encoder=None):
    """
    Calcule la parité démographique entre les groupes.
    Formule: P(Ŷ=1|A=a) = P(Ŷ=1|A=b)
    """
    groups = np.unique(sensitive_attr)
    results = {}
    
    for group in groups:
        group_mask = (sensitive_attr == group)
        positive_rate = np.mean(y_pred[group_mask])
        if sensitive_encoder is not None:
            group_name = sensitive_encoder.inverse_transform([group])[0]
        else:
            group_name = f"Groupe {group}"
        results[group_name] = positive_rate
    
    disparity = max(results.values()) - min(results.values())
    return results, disparity

def equalized_odds(y_true, y_pred, sensitive_attr, sensitive_encoder=None):
    """
    Calcule l'égalité des taux d'erreurs entre les groupes.
    Formule: P(Ŷ=1|A=a,Y=y) = P(Ŷ=1|A=b,Y=y) pour y ∈ {0,1}
    """
    groups = np.unique(sensitive_attr)
    results = {}
    
    for group in groups:
        group_mask = (sensitive_attr == group)
        
        # Taux de vrais positifs (TPR)
        tp_mask = (y_true == 1) & group_mask
        if np.sum(y_true[group_mask] == 1) > 0:
            tpr = np.sum(y_pred[tp_mask]) / np.sum(y_true[group_mask] == 1)
        else:
            tpr = np.nan
            
        # Taux de faux positifs (FPR)
        fp_mask = (y_true == 0) & group_mask
        if np.sum(y_true[group_mask] == 0) > 0:
            fpr = np.sum(y_pred[fp_mask]) / np.sum(y_true[group_mask] == 0)
        else:
            fpr = np.nan
            
        if sensitive_encoder is not None:
            group_name = sensitive_encoder.inverse_transform([group])[0]
        else:
            group_name = f"Groupe {group}"
        results[group_name] = {"TPR": tpr, "FPR": fpr}
    
    tpr_disparity = max([r["TPR"] for r in results.values()]) - min([r["TPR"] for r in results.values()]) if not np.isnan([r["TPR"] for r in results.values()]).any() else np.nan
    fpr_disparity = max([r["FPR"] for r in results.values()]) - min([r["FPR"] for r in results.values()]) if not np.isnan([r["FPR"] for r in results.values()]).any() else np.nan
    
    return results, tpr_disparity, fpr_disparity

def equal_opportunity(y_true, y_pred, sensitive_attr, sensitive_encoder=None):
    """
    Calcule l'égalité des chances entre les groupes.
    Formule: P(Ŷ=1|A=a,Y=1) = P(Ŷ=1|A=b,Y=1)
    """
    groups = np.unique(sensitive_attr)
    results = {}
    
    for group in groups:
        group_mask = (sensitive_attr == group)
        positive_mask = (y_true == 1) & group_mask
        
        if np.sum(y_true[group_mask] == 1) > 0:
            tpr = np.sum(y_pred[positive_mask]) / np.sum(y_true[group_mask] == 1)
        else:
            tpr = np.nan
            
        if sensitive_encoder is not None:
            group_name = sensitive_encoder.inverse_transform([group])[0]
        else:
            group_name = f"Groupe {group}"
        results[group_name] = tpr
    
    disparity = max(results.values()) - min(results.values()) if not np.isnan(list(results.values())).any() else np.nan
    return results, disparity

def predictive_parity(y_true, y_pred, sensitive_attr, sensitive_encoder=None):
    """
    Calcule l'équilibre des taux de précision entre les groupes.
    Formule: P(Y=1|A=a,Ŷ=1) = P(Y=1|A=b,Ŷ=1)
    """
    groups = np.unique(sensitive_attr)
    results = {}
    
    for group in groups:
        group_mask = (sensitive_attr == group)
        predicted_positive_mask = (y_pred == 1) & group_mask
        
        if np.sum(predicted_positive_mask) > 0:
            precision = np.sum((y_true == 1) & predicted_positive_mask) / np.sum(predicted_positive_mask)
        else:
            precision = np.nan
            
        if sensitive_encoder is not None:
            group_name = sensitive_encoder.inverse_transform([group])[0]
        else:
            group_name = f"Groupe {group}"
        results[group_name] = precision
    
    disparity = max(results.values()) - min(results.values()) if not np.isnan(list(results.values())).any() else np.nan
    return results, disparity

# Techniques de mitigation des biais
def reweighing(X, y, sensitive_attr):
    """
    Applique la technique de repondération pour atténuer les biais.
    """
    n_samples = len(y)
    unique_groups = np.unique(sensitive_attr)
    unique_labels = np.unique(y)
    
    # Calculer les fréquences
    weights = np.ones(n_samples)
    for group in unique_groups:
        for label in unique_labels:
            idx = (sensitive_attr == group) & (y == label)
            count = np.sum(idx)
            if count == 0:
                continue
                
            # P(A=a, Y=y)
            p_ay = count / n_samples
            
            # P(A=a) x P(Y=y)
            p_a = np.sum(sensitive_attr == group) / n_samples
            p_y = np.sum(y == label) / n_samples
            expected_p_ay = p_a * p_y
            
            # Poids: P(A=a) x P(Y=y) / P(A=a, Y=y)
            weights[idx] = expected_p_ay / p_ay
    
    return weights

def disparate_impact_remover(X, sensitive_attr, repair_level=1.0):
    """
    Transforme les caractéristiques pour réduire l'impact disparate.
    """
    X_transformed = X.copy()
    
    # Pour chaque caractéristique
    for col in range(X.shape[1]):
        # Vérifier si la caractéristique est numérique (pas un one-hot encoding)
        if np.array_equal(X[:, col], X[:, col].astype(int)) and len(np.unique(X[:, col])) <= 2:
            # Probablement une colonne binaire encodée, on la laisse telle quelle
            continue
            
        feature = X[:, col]
        
        # Pour chaque groupe
        groups = np.unique(sensitive_attr)
        transformed_feature = np.zeros_like(feature)
        
        for group in groups:
            group_mask = (sensitive_attr == group)
            group_feature = feature[group_mask]
            
            # Calculer les rangs dans le groupe
            if len(group_feature) > 0:
                ranks = np.argsort(np.argsort(group_feature))
                ranks = ranks / max(1, len(ranks) - 1)  # Normaliser entre 0 et 1
                
                # Interpoler entre la distribution originale et uniforme
                transformed_feature[group_mask] = (1 - repair_level) * group_feature + repair_level * ranks
    
        X_transformed[:, col] = transformed_feature
    
    return X_transformed

def adversarial_debiasing(X_train, y_train, sensitive_attr_train, X_test, base_model):
    """
    Simule un débiaisage adversarial (version simplifiée).
    En pratique, cela nécessiterait une implémentation de réseaux adversaires.
    """
    # Stratifier l'échantillonnage par attribut sensible
    models = {}
    predictions = np.zeros(len(X_test))
    
    for group in np.unique(sensitive_attr_train):
        mask = (sensitive_attr_train == group)
        if np.sum(mask) > 0:
            group_model = clone_model(base_model)
            group_model.fit(X_train[mask], y_train[mask])
            models[group] = group_model
    
    # Pondération des prédictions
    for group, model in models.items():
        group_preds = model.predict(X_test)
        predictions += group_preds / len(models)
    
    return (predictions > 0.5).astype(int)

def clone_model(model):
    """Crée une copie du modèle avec les mêmes hyperparamètres."""
    if isinstance(model, LogisticRegression):
        return LogisticRegression(C=model.C, penalty=model.penalty, solver=model.solver, random_state=42)
    elif isinstance(model, RandomForestClassifier):
        return RandomForestClassifier(n_estimators=model.n_estimators, max_depth=model.max_depth, random_state=42)
    elif isinstance(model, GradientBoostingClassifier):
        return GradientBoostingClassifier(n_estimators=model.n_estimators, learning_rate=model.learning_rate, random_state=42)
    elif isinstance(model, SVC):
        return SVC(C=model.C, kernel=model.kernel, probability=True, random_state=42)
    elif isinstance(model, MLPClassifier):
        return MLPClassifier(hidden_layer_sizes=model.hidden_layer_sizes, max_iter=model.max_iter, random_state=42)
    else:
        return model

def threshold_adjustment(y_scores, sensitive_attr, y_true=None, criterion='demographic_parity'):
    """
    Ajuste les seuils de classification par groupe pour réduire les disparités.
    """
    groups = np.unique(sensitive_attr)
    thresholds = {}
    predictions = np.zeros_like(y_scores, dtype=int)
    
    if criterion == 'demographic_parity':
        # Trouver des seuils qui égalisent les taux de prédictions positives
        target_rate = np.mean(y_scores > 0.5)
        
        for group in groups:
            group_mask = (sensitive_attr == group)
            group_scores = y_scores[group_mask]
            
            # Trouver un seuil qui donne le taux cible
            if len(group_scores) > 0:
                sorted_scores = np.sort(group_scores)
                idx = int((1 - target_rate) * len(sorted_scores))
                idx = min(max(0, idx), len(sorted_scores) - 1)
                thresholds[group] = sorted_scores[idx]
            else:
                thresholds[group] = 0.5
                
            # Appliquer le seuil
            predictions[group_mask] = (group_scores >= thresholds[group]).astype(int)
            
    elif criterion == 'equalized_odds' and y_true is not None:
        # Trouver des seuils qui égalisent TPR et FPR
        for group in groups:
            group_mask = (sensitive_attr == group)
            group_scores = y_scores[group_mask]
            group_true = y_true[group_mask]
            
            if len(group_scores) > 0:
                pos_scores = group_scores[group_true == 1]
                neg_scores = group_scores[group_true == 0]
                
                # Calculer TPR et FPR pour différents seuils
                thresholds_list = np.linspace(0, 1, 100)
                best_threshold = 0.5
                min_diff = float('inf')
                
                for t in thresholds_list:
                    if len(pos_scores) > 0:
                        tpr = np.mean(pos_scores >= t)
                    else:
                        tpr = 0
                        
                    if len(neg_scores) > 0:
                        fpr = np.mean(neg_scores >= t)
                    else:
                        fpr = 0
                        
                    # Différence entre TPR et FPR global
                    diff = abs(tpr - np.mean(y_scores[y_true == 1] >= 0.5)) + abs(fpr - np.mean(y_scores[y_true == 0] >= 0.5))
                    
                    if diff < min_diff:
                        min_diff = diff
                        best_threshold = t
                        
                thresholds[group] = best_threshold
            else:
                thresholds[group] = 0.5
                
            # Appliquer le seuil
            predictions[group_mask] = (group_scores >= thresholds[group]).astype(int)
    else:
        # Par défaut: seuil 0.5
        predictions = (y_scores >= 0.5).astype(int)
    
    return predictions, thresholds

# Fonction pour détecter les colonnes catégorielles
def detect_categorical_columns(df):
    categorical_columns = []
    for column in df.columns:
        # Détecter les colonnes catégorielles de manière plus robuste
        if (df[column].dtype == 'object' or 
            df[column].dtype.name == 'category' or 
            (df[column].dtype in ['int64', 'float64'] and df[column].nunique() < 10) or
            df[column].apply(lambda x: isinstance(x, str)).any()):
            categorical_columns.append(column)
    return categorical_columns

# Fonction pour analyser le dataframe et comprendre sa structure
def analyze_dataframe(df):
    """Analyse détaillée du dataframe pour débugger"""
    analysis = {}
    
    # Informations générales
    analysis["shape"] = df.shape
    analysis["columns"] = df.columns.tolist()
    analysis["dtypes"] = df.dtypes.apply(lambda x: str(x)).to_dict()
    
    # Nombre de valeurs uniques par colonne
    analysis["nunique"] = {col: df[col].nunique() for col in df.columns}
    
    # Échantillon de valeurs pour chaque colonne
    analysis["sample_values"] = {col: df[col].sample(min(5, len(df))).tolist() for col in df.columns}
    
    # Valeurs manquantes
    analysis["missing_values"] = df.isnull().sum().to_dict()
    
    # Détection automatique des colonnes catégorielles
    analysis["categorical_columns"] = detect_categorical_columns(df)
    
    # Colonnes qui contiennent des chaînes de caractères
    analysis["string_columns"] = [col for col in df.columns if df[col].apply(lambda x: isinstance(x, str)).any()]
    
    return analysis

# Fonction pour prétraiter les données de manière robuste
def preprocess_data_robust(df, feature_cols, target_col, sensitive_col):
    """
    Prétraite les données en gérant correctement tous les types de variables
    et en effectuant les encodages nécessaires.
    """
    # Analyse des types de colonnes
    categorical_cols = []
    numerical_cols = []
    
    # Pour chaque caractéristique (feature)
    for col in feature_cols:
        # Si c'est une colonne de type objet, catégorielle, ou avec peu de valeurs uniques, 
        # ou si elle contient des chaînes de caractères
        if (df[col].dtype == 'object' or 
            df[col].dtype.name == 'category' or 
            df[col].apply(lambda x: isinstance(x, str)).any() or
            (df[col].dtype in ['int64', 'float64'] and df[col].nunique() < 10)):
            categorical_cols.append(col)
        else:
            numerical_cols.append(col)
    
    # Création des encodeurs pour les colonnes catégorielles
    encoders = {}
    
    # Prétraitement des caractéristiques
    if categorical_cols and numerical_cols:
        # Si nous avons à la fois des colonnes catégorielles et numériques
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols),
                ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols)
            ]
        )
    elif categorical_cols:
        # Si nous n'avons que des colonnes catégorielles
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols)
            ]
        )
    elif numerical_cols:
        # Si nous n'avons que des colonnes numériques
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols)
            ]
        )
    else:
        # Si nous n'avons aucune colonne (ne devrait pas arriver normalement)
        raise ValueError("Aucune colonne valide trouvée parmi les caractéristiques")
    
    # Transformation des caractéristiques
    X = df[feature_cols].copy()
    X_preprocessed = preprocessor.fit_transform(X)
    
    # Traitement de la variable cible
    # Si la cible est catégorielle, l'encoder
    if (df[target_col].dtype == 'object' or 
        df[target_col].dtype.name == 'category' or 
        df[target_col].apply(lambda x: isinstance(x, str)).any()):
        y_encoder = LabelEncoder()
        y = y_encoder.fit_transform(df[target_col])
        encoders['target'] = y_encoder
    else:
        y = df[target_col].values
    
    # Traitement de l'attribut sensible
    # Si l'attribut sensible est catégoriel, l'encoder
    if (df[sensitive_col].dtype == 'object' or 
        df[sensitive_col].dtype.name == 'category' or 
        df[sensitive_col].apply(lambda x: isinstance(x, str)).any()):
        sensitive_encoder = LabelEncoder()
        sensitive_attr = sensitive_encoder.fit_transform(df[sensitive_col])
        encoders['sensitive'] = sensitive_encoder
    else:
        sensitive_attr = df[sensitive_col].values
    
    return X_preprocessed, y, sensitive_attr, encoders, preprocessor

# Interface utilisateur Streamlit
def main():
    st.title("📊 Analyse d'Équité Algorithmique")
    st.markdown("*Un outil pour détecter et atténuer les biais dans les modèles prédictifs*")
    
    # Section d'introduction
    st.header("Introduction à l'Équité Algorithmique")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **Définition**
        
        Un modèle est équitable s'il ne favorise ou ne désavantage aucun groupe particulier défini par des attributs sensibles.
        """)
    
    with col2:
        st.warning("""
        **Problématique**
        
        Les modèles peuvent amplifier les biais présents dans les données d'entraînement ou introduire de nouveaux biais.
        """)
    
    # Barre latérale pour les options
    with st.sidebar:
        st.header("Configuration")
        
        # Téléchargement de fichier
        uploaded_file = st.file_uploader("Charger un dataset (CSV)", type=["csv"])
        
        # Options de modèle
        model_type = st.selectbox(
            "Choisir un modèle",
            ["Régression Logistique", "Random Forest", "Gradient Boosting", "SVM", "Réseau de Neurones"]
        )
        
        # Options d'équité
        st.subheader("Techniques d'équité")
        fairness_techniques = st.multiselect(
            "Sélectionner les techniques à appliquer",
            ["Repondération", "Disparate Impact Remover", "Ajustement de seuil", "Débiaisage adversarial"],
            default=["Repondération"]
        )
        
        # Option de débogage
        debug_mode = st.checkbox("Mode débogage", value=True)
        
        # Paramètres avancés (facultatifs)
        with st.expander("Paramètres avancés"):
            test_size = st.slider("Proportion de test", 0.1, 0.5, 0.3, 0.05)
            random_state = st.slider("Graine aléatoire", 0, 100, 42)
            
            if "Disparate Impact Remover" in fairness_techniques:
                repair_level = st.slider("Niveau de réparation", 0.0, 1.0, 0.8, 0.1)
            
            if "Ajustement de seuil" in fairness_techniques:
                threshold_criterion = st.selectbox(
                    "Critère d'ajustement",
                    ["demographic_parity", "equalized_odds"]
                )
    
    # Partie principale de l'application
    if uploaded_file is not None:
        try:
            # Chargement des données
            df = pd.read_csv(uploaded_file)
            
            # Afficher l'aperçu du dataset
            st.subheader("Aperçu du dataset")
            st.dataframe(df.head())
            
            # Configuration de la préparation des données
            st.subheader("Préparation des données")
            
            # Sélection des colonnes
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Sélectionner les colonnes du modèle")
                feature_cols = st.multiselect(
                    "Caractéristiques",
                    df.columns.tolist(),
                    default=[col for col in df.columns if col not in [''] and df[col].nunique() > 1][:min(5, len(df.columns))]
                )
                
                target_options = [col for col in df.columns if col not in feature_cols]
                if target_options:
                    target_col = st.selectbox(
                        "Variable cible (binaire)",
                        target_options
                    )
                else:
                    target_col = st.selectbox(
                        "Variable cible (binaire)",
                        df.columns.tolist()
                    )
                    if target_col in feature_cols:
                        feature_cols.remove(target_col)
                        st.warning(f"'{target_col}' a été retiré des caractéristiques car c'est la variable cible.")
            
            with col2:
                st.write("Sélectionner l'attribut sensible")
                sensitive_options = [col for col in df.columns if col not in [target_col] + feature_cols]
                if sensitive_options:
                    sensitive_col = st.selectbox(
                        "Attribut sensible (groupe protégé)",
                        sensitive_options
                    )
                else:
                    sensitive_col = st.selectbox(
                        "Attribut sensible (groupe protégé)",
                        df.columns.tolist()
                    )
                    if sensitive_col in feature_cols:
                        feature_cols.remove(sensitive_col)
                        st.warning(f"'{sensitive_col}' a été retiré des caractéristiques car c'est l'attribut sensible.")
                
                # Vérification que les colonnes sélectionnées sont valides
                if not feature_cols:
                    st.error("Veuillez sélectionner au moins une caractéristique.")
                    return
                
                if target_col in feature_cols:
                    feature_cols.remove(target_col)
                    st.warning(f"'{target_col}' a été retiré des caractéristiques car c'est la variable cible.")
                    
                if sensitive_col in feature_cols:
                    feature_cols.remove(sensitive_col)
                    st.warning(f"'{sensitive_col}' a été retiré des caractéristiques car c'est l'attribut sensible.")
            
            # Préparation des données
            if st.button("Analyser et appliquer les techniques d'équité"):
                with st.spinner("Traitement en cours..."):
                    # Mode débogage - Analyse du dataframe
                    if debug_mode:
                        with st.expander("Informations de débogage"):
                            st.subheader("Analyse du dataframe")
                            analysis = analyze_dataframe(df)
                            
                            st.write(f"Dimensions du dataframe: {analysis['shape']}")
                            st.write(f"Colonnes: {analysis['columns']}")
                            
                            st.subheader("Types de données")
                            st.write(analysis['dtypes'])
                            
                            st.subheader("Colonnes catégorielles détectées")
                            st.write(analysis['categorical_columns'])
                            
                            st.subheader("Colonnes contenant des chaînes de caractères")
                            st.write(analysis['string_columns'])
                            
                            st.subheader("Valeurs uniques par colonne")
                            st.write(analysis['nunique'])
                            
                            st.subheader("Échantillon de valeurs par colonne")
                            st.json(analysis['sample_values'])
                    
                    # Prétraitement robuste des données
                    try:
                        X_preprocessed, y, sensitive_attr, encoders, preprocessor = preprocess_data_robust(
                            df, feature_cols, target_col, sensitive_col
                        )
                        
                        if debug_mode:
                            with st.expander("Informations sur les données prétraitées"):
                                st.write(f"Forme des données prétraitées: {X_preprocessed.shape}")
                                st.write(f"Variable cible: {np.unique(y)}, forme: {y.shape}")
                                st.write(f"Attribut sensible: {np.unique(sensitive_attr)}, forme: {sensitive_attr.shape}")
                                
                                # Afficher les encodeurs si disponibles
                                if 'target' in encoders:
                                    st.write(f"Classes de la cible: {encoders['target'].classes_}")
                                if 'sensitive' in encoders:
                                    st.write(f"Classes de l'attribut sensible: {encoders['sensitive'].classes_}")
                        
                        # Vérifier si la variable cible est binaire
                        unique_y = np.unique(y)
                        if len(unique_y) != 2:
                            st.error(f"La variable cible doit être binaire. La colonne '{target_col}' contient {len(unique_y)} valeurs uniques.")
                            return
                            
                    except Exception as e:
                        st.error(f"Erreur lors du prétraitement des données: {str(e)}")
                        if debug_mode:
                            st.write("Trace complète de l'erreur:")
                            st.code(traceback.format_exc())
                        return
                    
                    # Division des données
                    X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
                        X_preprocessed, y, sensitive_attr, test_size=test_size, random_state=random_state, stratify=y
                    )
                    
                    # Création du modèle de base
                    if model_type == "Régression Logistique":
                        base_model = LogisticRegression(random_state=random_state, max_iter=1000)
                    elif model_type == "Random Forest":
                        base_model = RandomForestClassifier(n_estimators=100, random_state=random_state)
                    elif model_type == "Gradient Boosting":
                        base_model = GradientBoostingClassifier(random_state=random_state)
                    elif model_type == "SVM":
                        base_model = SVC(probability=True, random_state=random_state)
                    elif model_type == "Réseau de Neurones":
                        base_model = MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=1000, random_state=random_state)
                    
                    # Entraînement et évaluation du modèle de base
                    try:
                        base_model.fit(X_train, y_train)
                        
                        if hasattr(base_model, 'predict_proba'):
                            base_probas = base_model.predict_proba(X_test)[:, 1]
                        else:
                            # Pour les modèles comme SVM qui n'ont pas toujours predict_proba
                            try:
                                base_probas = base_model.decision_function(X_test)
                                # Normalisation des scores entre 0 et 1 pour être utilisables comme des probabilités
                                base_probas = (base_probas - base_probas.min()) / (base_probas.max() - base_probas.min())
                            except:
                                base_probas = None
                                if debug_mode:
                                    st.warning("Ce modèle ne fournit pas de probabilités. Certaines métriques et techniques d'équité pourraient ne pas être disponibles.")
                        
                        base_preds = base_model.predict(X_test)
                        
                        # Métriques du modèle de base
                        base_metrics = {
                            "Précision": accuracy_score(y_test, base_preds),
                            "Rappel": recall_score(y_test, base_preds, zero_division=0),
                            "F1-score": f1_score(y_test, base_preds, zero_division=0),
                            "AUC": roc_auc_score(y_test, base_probas) if base_probas is not None else None
                        }
                        
                        # Métriques d'équité du modèle de base
                        base_dp, base_dp_disparity = demographic_parity(base_preds, sens_test, encoders.get('sensitive'))
                        base_eo, base_tpr_disparity, base_fpr_disparity = equalized_odds(y_test, base_preds, sens_test, encoders.get('sensitive'))
                        base_eop, base_eop_disparity = equal_opportunity(y_test, base_preds, sens_test, encoders.get('sensitive'))
                        base_pp, base_pp_disparity = predictive_parity(y_test, base_preds, sens_test, encoders.get('sensitive'))
                        
                        base_fairness_metrics = {
                            "Disparité de parité démographique": base_dp_disparity,
                            "Disparité TPR": base_tpr_disparity,
                            "Disparité FPR": base_fpr_disparity,
                            "Disparité d'égalité des chances": base_eop_disparity,
                            "Disparité de parité prédictive": base_pp_disparity
                        }
                        
                    except Exception as e:
                        st.error(f"Erreur lors de l'entraînement ou de l'évaluation du modèle: {str(e)}")
                        if debug_mode:
                            st.write("Trace complète de l'erreur:")
                            st.code(traceback.format_exc())
                        return
                    
                    # Appliquer les techniques d'équité sélectionnées
                    fairness_results = {}
                    
                    for technique in fairness_techniques:
                        try:
                            if technique == "Repondération":
                                weights = reweighing(X_train, y_train, sens_train)
                                fair_model = clone_model(base_model)
                                fair_model.fit(X_train, y_train, sample_weight=weights)
                                
                                if hasattr(fair_model, 'predict_proba'):
                                    fair_probas = fair_model.predict_proba(X_test)[:, 1]
                                else:
                                    try:
                                        fair_probas = fair_model.decision_function(X_test)
                                        fair_probas = (fair_probas - fair_probas.min()) / (fair_probas.max() - fair_probas.min())
                                    except:
                                        fair_probas = None
                                
                                fair_preds = fair_model.predict(X_test)
                                
                            elif technique == "Disparate Impact Remover":
                                repair_level_val = repair_level if "repair_level" in locals() else 0.8
                                X_train_repaired = disparate_impact_remover(X_train, sens_train, repair_level_val)
                                X_test_repaired = disparate_impact_remover(X_test, sens_test, repair_level_val)
                                
                                fair_model = clone_model(base_model)
                                fair_model.fit(X_train_repaired, y_train)
                                
                                if hasattr(fair_model, 'predict_proba'):
                                    fair_probas = fair_model.predict_proba(X_test_repaired)[:, 1]
                                else:
                                    try:
                                        fair_probas = fair_model.decision_function(X_test_repaired)
                                        fair_probas = (fair_probas - fair_probas.min()) / (fair_probas.max() - fair_probas.min())
                                    except:
                                        fair_probas = None
                                
                                fair_preds = fair_model.predict(X_test_repaired)
                                
                            elif technique == "Ajustement de seuil":
                                if base_probas is not None:
                                    criterion = threshold_criterion if "threshold_criterion" in locals() else "demographic_parity"
                                    fair_preds, thresholds = threshold_adjustment(base_probas, sens_test, y_test, criterion)
                                    fair_probas = base_probas
                                else:
                                    st.warning(f"L'ajustement de seuil nécessite des probabilités, ce qui n'est pas disponible pour {model_type}. Technique ignorée.")
                                    continue
                                    
                            elif technique == "Débiaisage adversarial":
                                fair_preds = adversarial_debiasing(X_train, y_train, sens_train, X_test, base_model)
                                fair_probas = None
                            
                            # Métriques du modèle équitable
                            fair_metrics = {
                                "Précision": accuracy_score(y_test, fair_preds),
                                "Rappel": recall_score(y_test, fair_preds, zero_division=0),
                                "F1-score": f1_score(y_test, fair_preds, zero_division=0),
                                "AUC": roc_auc_score(y_test, fair_probas) if fair_probas is not None else None
                            }
                            
                            # Métriques d'équité du modèle équitable
                            fair_dp, fair_dp_disparity = demographic_parity(fair_preds, sens_test, encoders.get('sensitive'))
                            fair_eo, fair_tpr_disparity, fair_fpr_disparity = equalized_odds(y_test, fair_preds, sens_test, encoders.get('sensitive'))
                            fair_eop, fair_eop_disparity = equal_opportunity(y_test, fair_preds, sens_test, encoders.get('sensitive'))
                            fair_pp, fair_pp_disparity = predictive_parity(y_test, fair_preds, sens_test, encoders.get('sensitive'))
                            
                            fair_fairness_metrics = {
                                "Disparité de parité démographique": fair_dp_disparity,
                                "Disparité TPR": fair_tpr_disparity,
                                "Disparité FPR": fair_fpr_disparity,
                                "Disparité d'égalité des chances": fair_eop_disparity,
                                "Disparité de parité prédictive": fair_pp_disparity
                            }
                            
                            # Stocker les résultats
                            fairness_results[technique] = {
                                "Métriques de performance": fair_metrics,
                                "Métriques d'équité": fair_fairness_metrics,
                                "Parité démographique": fair_dp,
                                "Égalité des chances": {"TPR": {k: v["TPR"] for k, v in fair_eo.items()}, 
                                                      "FPR": {k: v["FPR"] for k, v in fair_eo.items()}},
                                "Égalité des opportunités": fair_eop,
                                "Parité prédictive": fair_pp
                            }
                        except Exception as e:
                            st.warning(f"Erreur lors de l'application de la technique '{technique}': {str(e)}")
                            if debug_mode:
                                st.write("Trace complète de l'erreur:")
                                st.code(traceback.format_exc())
                    
                    # Afficher les résultats
                    st.header("Résultats")
                    
                    # Modèle de base
                    st.subheader("Modèle de base")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("Métriques de performance")
                        performance_df = pd.DataFrame([base_metrics])
                        st.dataframe(performance_df)
                        
                    with col2:
                        st.write("Métriques d'équité")
                        fairness_df = pd.DataFrame([base_fairness_metrics])
                        st.dataframe(fairness_df)
                    
                    # Visualisations pour le modèle de base
                    st.subheader("Visualisations du modèle de base")
                    
                    try:
                        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                        
                        # Parité Démographique
                        ax = axes[0, 0]
                        groups = list(base_dp.keys())
                        values = list(base_dp.values())
                        ax.bar(groups, values, color='salmon')
                        ax.set_title('Parité Démographique')
                        ax.set_ylabel('Taux de prédictions positives')
                        ax.set_ylim(0, 1)
                        
                        # Égalité des Taux d'Erreurs
                        ax = axes[0, 1]
                        groups = list(base_eo.keys())
                        tpr_values = [v["TPR"] for v in base_eo.values()]
                        fpr_values = [v["FPR"] for v in base_eo.values()]
                        
                        x = np.arange(len(groups))
                        width = 0.35
                        
                        ax.bar(x - width/2, tpr_values, width, label='TPR', color='mediumorchid')
                        ax.bar(x + width/2, fpr_values, width, label='FPR', color='plum')
                        ax.set_title('Égalité des Taux d\'Erreurs')
                        ax.set_xticks(x)
                        ax.set_xticklabels(groups)
                        ax.set_ylabel('Taux')
                        ax.set_ylim(0, 1)
                        ax.legend()
                        
                        # Égalité des Chances
                        ax = axes[1, 0]
                        groups = list(base_eop.keys())
                        values = list(base_eop.values())
                        ax.bar(groups, values, color='cornflowerblue')
                        ax.set_title('Égalité des Chances')
                        ax.set_ylabel('Taux de vrais positifs')
                        ax.set_ylim(0, 1)
                        
                        # Équilibre des Taux de Précision
                        ax = axes[1, 1]
                        groups = list(base_pp.keys())
                        values = list(base_pp.values())
                        ax.bar(groups, values, color='mediumseagreen')
                        ax.set_title('Équilibre des Taux de Précision')
                        ax.set_ylabel('Précision parmi les prédictions positives')
                        ax.set_ylim(0, 1)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Erreur lors de la génération des visualisations: {str(e)}")
                        if debug_mode:
                            st.write("Trace complète de l'erreur:")
                            st.code(traceback.format_exc())
                    
                    # Matrice de confusion par groupe
                    st.subheader("Matrices de confusion par groupe")
                    
                    try:
                        unique_groups = np.unique(sens_test)
                        n_groups = len(unique_groups)
                        
                        fig, axes = plt.subplots(1, n_groups, figsize=(5*n_groups, 4))
                        if n_groups == 1:
                            axes = [axes]
                            
                        for i, group in enumerate(unique_groups):
                            group_mask = (sens_test == group)
                            cm = confusion_matrix(y_test[group_mask], base_preds[group_mask])
                            
                            if 'sensitive' in encoders:
                                group_name = encoders['sensitive'].inverse_transform([group])[0]
                            else:
                                group_name = f"Groupe {group}"
                            
                            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i])
                            axes[i].set_title(f'{group_name}')
                            axes[i].set_xlabel('Prédiction')
                            axes[i].set_ylabel('Valeur réelle')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Erreur lors de la génération des matrices de confusion: {str(e)}")
                        if debug_mode:
                            st.write("Trace complète de l'erreur:")
                            st.code(traceback.format_exc())
                    
                    # Résultats des techniques d'équité
                    if fairness_results:
                        st.header("Comparaison des techniques d'équité")
                        
                        # Tableau comparatif des performances
                        st.subheader("Comparaison des métriques de performance")
                        
                        performance_comparison = {
                            "Modèle de base": base_metrics
                        }
                        
                        for technique, results in fairness_results.items():
                            performance_comparison[technique] = results["Métriques de performance"]
                            
                        performance_df = pd.DataFrame(performance_comparison).T
                        st.dataframe(performance_df)
                        
                        # Tableau comparatif des métriques d'équité
                        st.subheader("Comparaison des métriques d'équité")
                        
                        fairness_comparison = {
                            "Modèle de base": base_fairness_metrics
                        }
                        
                        for technique, results in fairness_results.items():
                            fairness_comparison[technique] = results["Métriques d'équité"]
                            
                        fairness_df = pd.DataFrame(fairness_comparison).T
                        st.dataframe(fairness_df)
                        
                        # Visualisations comparatives
                        st.subheader("Comparaison visuelle des métriques d'équité")
                        
                        try:
                            # Préparer les données pour les graphiques
                            techniques = ["Modèle de base"] + list(fairness_results.keys())
                            
                            dp_disparities = [base_dp_disparity] + [results["Métriques d'équité"]["Disparité de parité démographique"] 
                                                                 for results in fairness_results.values()]
                            
                            eop_disparities = [base_eop_disparity] + [results["Métriques d'équité"]["Disparité d'égalité des chances"] 
                                                                    for results in fairness_results.values()]
                            
                            pp_disparities = [base_pp_disparity] + [results["Métriques d'équité"]["Disparité de parité prédictive"] 
                                                                  for results in fairness_results.values()]
                            
                            # Graphiques comparatifs
                            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
                            
                            # Disparité de parité démographique
                            axes[0].bar(techniques, dp_disparities, color='salmon')
                            axes[0].set_title('Disparité de parité démographique')
                            axes[0].set_ylabel('Disparité')
                            plt.setp(axes[0].get_xticklabels(), rotation=45, ha='right')
                            
                            # Disparité d'égalité des chances
                            axes[1].bar(techniques, eop_disparities, color='cornflowerblue')
                            axes[1].set_title('Disparité d\'égalité des chances')
                            axes[1].set_ylabel('Disparité')
                            plt.setp(axes[1].get_xticklabels(), rotation=45, ha='right')
                            
                            # Disparité de parité prédictive
                            axes[2].bar(techniques, pp_disparities, color='mediumseagreen')
                            axes[2].set_title('Disparité de parité prédictive')
                            axes[2].set_ylabel('Disparité')
                            plt.setp(axes[2].get_xticklabels(), rotation=45, ha='right')
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                        except Exception as e:
                            st.error(f"Erreur lors de la génération des visualisations comparatives: {str(e)}")
                            if debug_mode:
                                st.write("Trace complète de l'erreur:")
                                st.code(traceback.format_exc())
                        
                        # Évaluation détaillée des techniques
                        st.header("Analyse détaillée des techniques d'équité")
                        
                        for technique, results in fairness_results.items():
                            with st.expander(f"Détails pour {technique}"):
                                st.subheader(f"Métriques de parité pour {technique}")
                                
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    # Parité Démographique
                                    st.write("**Parité Démographique**")
                                    dp_data = results["Parité démographique"]
                                    dp_df = pd.DataFrame(dp_data.items(), columns=["Groupe", "Taux de prédictions positives"])
                                    st.dataframe(dp_df)
                                    
                                    # Égalité des Opportunités
                                    st.write("**Égalité des Opportunités**")
                                    eop_data = results["Égalité des opportunités"]
                                    eop_df = pd.DataFrame(eop_data.items(), columns=["Groupe", "Taux de vrais positifs"])
                                    st.dataframe(eop_df)
                                
                                with col2:
                                    # Égalité des Chances (TPR)
                                    st.write("**Égalité des Chances (TPR)**")
                                    tpr_data = results["Égalité des chances"]["TPR"]
                                    tpr_df = pd.DataFrame(tpr_data.items(), columns=["Groupe", "TPR"])
                                    st.dataframe(tpr_df)
                                    
                                    # Parité Prédictive
                                    st.write("**Parité Prédictive**")
                                    pp_data = results["Parité prédictive"]
                                    pp_df = pd.DataFrame(pp_data.items(), columns=["Groupe", "Précision parmi prédictions positives"])
                                    st.dataframe(pp_df)
                                
                                # Matrice de confusion par groupe pour cette technique
                                st.write("**Matrices de confusion par groupe**")
                                
                                try:
                                    if technique == "Ajustement de seuil":
                                        fair_preds = fairness_results[technique].get("predictions", 
                                                                                  threshold_adjustment(base_probas, sens_test, y_test)[0])
                                    elif technique == "Débiaisage adversarial":
                                        fair_preds = adversarial_debiasing(X_train, y_train, sens_train, X_test, base_model)
                                    else:
                                        fair_model = clone_model(base_model)
                                        if technique == "Repondération":
                                            weights = reweighing(X_train, y_train, sens_train)
                                            fair_model.fit(X_train, y_train, sample_weight=weights)
                                            fair_preds = fair_model.predict(X_test)
                                        elif technique == "Disparate Impact Remover":
                                            repair_level_val = repair_level if "repair_level" in locals() else 0.8
                                            X_train_repaired = disparate_impact_remover(X_train, sens_train, repair_level_val)
                                            X_test_repaired = disparate_impact_remover(X_test, sens_test, repair_level_val)
                                            fair_model.fit(X_train_repaired, y_train)
                                            fair_preds = fair_model.predict(X_test_repaired)
                                    
                                    fig, axes = plt.subplots(1, n_groups, figsize=(5*n_groups, 4))
                                    if n_groups == 1:
                                        axes = [axes]
                                        
                                    for i, group in enumerate(unique_groups):
                                        group_mask = (sens_test == group)
                                        cm = confusion_matrix(y_test[group_mask], fair_preds[group_mask])
                                        
                                        if 'sensitive' in encoders:
                                            group_name = encoders['sensitive'].inverse_transform([group])[0]
                                        else:
                                            group_name = f"Groupe {group}"
                                        
                                        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i])
                                        axes[i].set_title(f'{group_name}')
                                        axes[i].set_xlabel('Prédiction')
                                        axes[i].set_ylabel('Valeur réelle')
                                    
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                except Exception as e:
                                    st.error(f"Erreur lors de la génération des matrices de confusion pour {technique}: {str(e)}")
                                    if debug_mode:
                                        st.write("Trace complète de l'erreur:")
                                        st.code(traceback.format_exc())
                        
                        # Recommandation finale
                        st.header("Recommandation")
                        
                        try:
                            # Calculer un score composite pour chaque technique (trade-off performance-équité)
                            # Plus petit = meilleur
                            composite_scores = {}
                            
                            # Base model as reference
                            base_perf = (base_metrics["F1-score"] if base_metrics["F1-score"] is not None else 0)
                            base_fairness = np.mean([v for v in base_fairness_metrics.values() if v is not None and not np.isnan(v)])
                            
                            for technique, results in fairness_results.items():
                                perf_metrics = results["Métriques de performance"]
                                fair_metrics = results["Métriques d'équité"]
                                
                                # Performance relative (F1-score)
                                perf = (perf_metrics["F1-score"] if perf_metrics["F1-score"] is not None else 0)
                                perf_diff = max(0, base_perf - perf)  # Pénalité si performance diminue
                                
                                # Équité relative (moyenne des disparités)
                                fairness = np.mean([v for v in fair_metrics.values() if v is not None and not np.isnan(v)])
                                fairness_improvement = max(0, base_fairness - fairness)  # Bonus si équité s'améliore
                                
                                # Score composite (plus petit = meilleur)
                                composite_scores[technique] = perf_diff - fairness_improvement
                            
                            # Trouver la meilleure technique
                            if composite_scores:
                                best_technique = min(composite_scores.items(), key=lambda x: x[1])[0]
                                
                                st.info(f"**Technique recommandée : {best_technique}**")
                                st.write("Cette technique offre le meilleur compromis entre la performance du modèle et l'équité algorithmique pour ce dataset.")
                                
                                # Détails sur la technique recommandée
                                st.write(f"**Détails sur {best_technique}:**")
                                
                                if best_technique == "Repondération":
                                    st.write("Cette technique attribue des poids aux échantillons d'entraînement pour garantir une représentation équilibrée des groupes protégés et non protégés.")
                                elif best_technique == "Disparate Impact Remover":
                                    st.write("Cette technique transforme les caractéristiques pour réduire les corrélations avec l'attribut sensible tout en préservant la capacité prédictive.")
                                elif best_technique == "Ajustement de seuil":
                                    st.write("Cette technique applique des seuils de classification différents pour chaque groupe afin d'équilibrer les taux de prédictions positives ou les taux d'erreur.")
                                elif best_technique == "Débiaisage adversarial":
                                    st.write("Cette technique utilise une approche d'apprentissage adversaire pour réduire les biais en empêchant le modèle d'apprendre des corrélations avec l'attribut sensible.")
                            else:
                                st.warning("Aucune technique d'équité n'a été appliquée ou évaluée.")
                        except Exception as e:
                            st.error(f"Erreur lors de la génération de la recommandation: {str(e)}")
                            if debug_mode:
                                st.write("Trace complète de l'erreur:")
                                st.code(traceback.format_exc())
                    else:
                        st.warning("Aucune technique d'équité n'a été sélectionnée ou appliquée.")
        except Exception as e:
            st.error(f"Une erreur s'est produite lors du traitement des données : {str(e)}")
            if debug_mode:
                st.write("Trace complète de l'erreur:")
                st.code(traceback.format_exc())

if __name__ == "__main__":
    main()