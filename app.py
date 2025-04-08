import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import plotly.express as px
import plotly.graph_objects as go
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from fairlearn.metrics import MetricFrame, selection_rate
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds
import plotly.figure_factory as ff

# Configuration de la page
st.set_page_config(
    page_title="Équité dans l'Apprentissage Automatique",
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

# Fonction pour détecter les colonnes catégorielles
def detect_categorical_columns(df):
    categorical_columns = []
    for column in df.columns:
        if df[column].dtype == 'object' or (df[column].dtype in ['int64', 'float64'] and df[column].nunique() < 10):
            categorical_columns.append(column)
    return categorical_columns

# Fonction pour le prétraitement des données
def preprocess_data(df, target, sensitive_features):
    # Copie du DataFrame original
    df_processed = df.copy()
    
    # Identification des colonnes catégorielles
    categorical_columns = detect_categorical_columns(df)
    st.write("Colonnes catégorielles détectées:", categorical_columns)
    
    # Création d'un dictionnaire pour stocker les encodeurs
    encoders = {}
    
    # Label Encoding pour toutes les colonnes catégorielles
    for column in categorical_columns:
        if column != target:  # Ne pas encoder la cible pour l'instant
            encoders[column] = LabelEncoder()
            df_processed[column] = encoders[column].fit_transform(df[column].astype(str))
    
    # Encoder la variable cible séparément
    if target in categorical_columns:
        encoders[target] = LabelEncoder()
        df_processed[target] = encoders[target].fit_transform(df[target].astype(str))
    
    # Afficher les mappings des encodages
    #st.subheader("Mappings des encodages")
    for column, encoder in encoders.items():
        mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
        #st.write(f"{column}:", mapping)
    
    return df_processed, encoders

# Titre principal
st.title("Équité dans l'Apprentissage Automatique")
st.markdown("*Modélisation et visualisation des biais algorithmiques*")

# Section Introduction
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

# Upload du fichier CSV
uploaded_file = st.file_uploader("Choisissez un fichier CSV", type=['csv'])

if uploaded_file is not None:
    # Lecture du fichier
    df = pd.read_csv(uploaded_file)
    
    # Affichage des données brutes
    st.subheader("Données brutes")
    st.write(df.head())
    st.write("Statistiques descriptives:")
    st.write(df.describe())
    
    # Configuration de l'analyse
    st.subheader("Configuration de l'Analyse")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        target = st.selectbox("Variable cible", df.columns)
    with col2:
        sensitive_features = st.multiselect(
            "Caractéristiques sensibles",
            [col for col in df.columns if col != target]
        )
    with col3:
        model_choice = st.selectbox(
            "Modèle",
            ["Régression Logistique", "Random Forest", "Gradient Boosting"]
        )

    if target and sensitive_features:
        # Prétraitement des données
        #st.subheader("Prétraitement des données")
        df_processed, encoders = preprocess_data(df, target, sensitive_features)
        
        # Affichage des données prétraitées
        #st.write("Données après prétraitement:")
        #st.write(df_processed.head())
        
        # Préparation des données
        X = df_processed.drop(columns=[target])
        y = df_processed[target]
        
        # Sauvegarde des caractéristiques sensibles
        sensitive_data = X[sensitive_features].copy()
        
        # Division des données
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Normalisation (en excluant les caractéristiques sensibles)
        scaler = StandardScaler()
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        # Colonnes à normaliser (toutes sauf les caractéristiques sensibles)
        cols_to_scale = [col for col in X_train.columns if col not in sensitive_features]
        
        X_train_scaled[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
        X_test_scaled[cols_to_scale] = scaler.transform(X_test[cols_to_scale])
        
        # Sélection et entraînement du modèle
        if model_choice == "Régression Logistique":
            model = LogisticRegression(random_state=42)
        elif model_choice == "Random Forest":
            model = RandomForestClassifier(random_state=42)
        else:
            model = GradientBoostingClassifier(random_state=42)
        
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        # Simulateur d'Équité Algorithmique
        st.header("Simulateur d'Équité Algorithmique")
        
        tab1, tab2, tab3 = st.tabs([
            "Distribution des Prédictions", 
            "Métriques d'Équité",
            "Matrices de Confusion par Groupe"
        ])
        
        with tab1:
            st.subheader("Distribution des Prédictions")
            for feature in sensitive_features:
                # Création du DataFrame pour la visualisation avec les valeurs décodées
                pred_df = pd.DataFrame({
                    'Groupe': encoders[feature].inverse_transform(X_test[feature].astype(int)) if feature in encoders else X_test[feature],
                    'Prédiction': encoders[target].inverse_transform(y_pred) if target in encoders else y_pred,
                    'Type': 'Prédiction'
                })
                true_df = pd.DataFrame({
                    'Groupe': encoders[feature].inverse_transform(X_test[feature].astype(int)) if feature in encoders else X_test[feature],
                    'Prédiction': encoders[target].inverse_transform(y_test) if target in encoders else y_test,
                    'Type': 'Réel'
                })
                viz_df = pd.concat([pred_df, true_df])
                
                fig = px.histogram(
                    viz_df,
                    x="Prédiction",
                    color="Type",
                    barmode="group",
                    facet_col="Groupe",
                    title=f"Distribution des prédictions par {feature}"
                )
                st.plotly_chart(fig)

        with tab2:
            st.subheader("Métriques d'Équité")
            for feature in sensitive_features:
                metrics = MetricFrame(
                    metrics={
                        'selection_rate': selection_rate,
                        'accuracy': accuracy_score
                    },
                    y_true=y_test,
                    y_pred=y_pred,
                    sensitive_features=X_test[feature]
                )
                
                # Création d'un graphique en radar pour les métriques
                categories = ['Égalité des chances', 'Parité démographique', 
                            'Égalité des taux', 'Équité des faux positifs']
                fig = go.Figure()
                
                fig.add_trace(go.Scatterpolar(
                    r=[
                        metrics.overall['accuracy'],
                        demographic_parity_difference(y_test, y_pred, 
                                                    sensitive_features=X_test[feature]),
                        metrics.difference()['selection_rate'],
                        equalized_odds_difference(y_test, y_pred, 
                                               sensitive_features=X_test[feature])
                    ],
                    theta=categories,
                    fill='toself',
                    name='Métriques d\'équité'
                ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )),
                    showlegend=True,
                    title=f"Métriques d'équité pour {feature}"
                )
                st.plotly_chart(fig)

        with tab3:
            st.subheader("Matrices de Confusion par Groupe")
            for feature in sensitive_features:
                unique_groups = X_test[feature].unique()
                cols = st.columns(len(unique_groups))
                
                for idx, group in enumerate(unique_groups):
                    mask = X_test[feature] == group
                    cm = confusion_matrix(y_test[mask], y_pred[mask])
                    
                    # Obtenir le nom décodé du groupe
                    group_name = encoders[feature].inverse_transform([group])[0] if feature in encoders else group
                    
                    with cols[idx]:
                        fig = go.Figure(data=go.Heatmap(
                            z=cm,
                            x=['Prédit 0', 'Prédit 1'],
                            y=['Réel 0', 'Réel 1'],
                            text=cm,
                            texttemplate="%{text}",
                            textfont={"size": 16},
                            colorscale='Blues'
                        ))
                        fig.update_layout(
                            title=f'Matrice de Confusion - {feature}={group_name}',
                            height=400
                        )
                        st.plotly_chart(fig)

        # Section Mitigation des Biais
        st.header("Mitigation des Biais")
        mitigation_technique = st.selectbox(
            "Choisissez une technique de mitigation",
            ["Aucune", "Parité Démographique", "Égalité des Chances"]
        )

        if mitigation_technique != "Aucune":
            # Créer le modèle de base pour la mitigation
            estimator = LogisticRegression(random_state=42)
            
            if mitigation_technique == "Parité Démographique":
                # Utiliser ExponentiatedGradient avec DemographicParity
                constraint = ExponentiatedGradient(
                    estimator=estimator,
                    constraints=DemographicParity(),
                    eps=0.01
                )
            else:
                # Pour l'égalité des chances, utiliser EqualizedOdds
                constraint = ExponentiatedGradient(
                    estimator=estimator,
                    constraints=EqualizedOdds(),
                    eps=0.01
                )
            
            try:
                # Entraînement du modèle avec contrainte
                constraint.fit(
                    X_train_scaled,
                    y_train,
                    sensitive_features=X_train[sensitive_features[0]]
                )
                y_pred_fair = constraint.predict(X_test_scaled)
                
                # Comparaison des résultats
                st.subheader("Comparaison des Résultats")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("Modèle Original:")
                    st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
                    dpd_original = demographic_parity_difference(
                        y_test, y_pred, 
                        sensitive_features=X_test[sensitive_features[0]]
                    )
                    st.write(f"Disparité Démographique: {dpd_original:.3f}")
                
                with col2:
                    st.write("Modèle Équitable:")
                    st.write(f"Accuracy: {accuracy_score(y_test, y_pred_fair):.3f}")
                    dpd_fair = demographic_parity_difference(
                        y_test, y_pred_fair, 
                        sensitive_features=X_test[sensitive_features[0]]
                    )
                    st.write(f"Disparité Démographique: {dpd_fair:.3f}")
                
                # Ajout de visualisations comparatives
                st.subheader("Comparaison des distributions")
                
                # Création du DataFrame pour la visualisation
                compare_df = pd.DataFrame({
                    'Groupe': np.concatenate([
                        encoders[sensitive_features[0]].inverse_transform(X_test[sensitive_features[0]].astype(int)) 
                        if sensitive_features[0] in encoders else X_test[sensitive_features[0]],
                        encoders[sensitive_features[0]].inverse_transform(X_test[sensitive_features[0]].astype(int))
                        if sensitive_features[0] in encoders else X_test[sensitive_features[0]]
                    ]),
                    'Prédiction': np.concatenate([
                        encoders[target].inverse_transform(y_pred) if target in encoders else y_pred,
                        encoders[target].inverse_transform(y_pred_fair) if target in encoders else y_pred_fair
                    ]),
                    'Modèle': np.concatenate([
                        np.repeat('Original', len(y_pred)),
                        np.repeat('Équitable', len(y_pred_fair))
                    ])
                })
                
                fig = px.histogram(
                    compare_df,
                    x="Prédiction",
                    color="Modèle",
                    barmode="group",
                    facet_col="Groupe",
                    title="Comparaison des distributions de prédictions entre les modèles"
                )
                st.plotly_chart(fig)
                
                # Ajout des métriques d'équité comparatives
                st.subheader("Métriques d'équité comparatives")
                
                metrics_original = MetricFrame(
                    metrics={'selection_rate': selection_rate},
                    y_true=y_test,
                    y_pred=y_pred,
                    sensitive_features=X_test[sensitive_features[0]]
                )
                
                metrics_fair = MetricFrame(
                    metrics={'selection_rate': selection_rate},
                    y_true=y_test,
                    y_pred=y_pred_fair,
                    sensitive_features=X_test[sensitive_features[0]]
                )
                
                # Création d'un tableau comparatif
                comparison_data = {
                    'Métrique': ['Taux de sélection par groupe (Original)', 'Taux de sélection par groupe (Équitable)'],
                    'Valeurs': [
                        str(dict(metrics_original.by_group['selection_rate'])),
                        str(dict(metrics_fair.by_group['selection_rate']))
                    ]
                }
                st.table(pd.DataFrame(comparison_data))
                
                if mitigation_technique == "Égalité des Chances":
                    # Ajout des métriques spécifiques à l'égalité des chances
                    st.subheader("Métriques d'Égalité des Chances")
                    
                    eod_before = equalized_odds_difference(
                        y_test,
                        y_pred,
                        sensitive_features=X_test[sensitive_features[0]]
                    )
                    
                    eod_after = equalized_odds_difference(
                        y_test,
                        y_pred_fair,
                        sensitive_features=X_test[sensitive_features[0]]
                    )
                    
                    st.write(f"Différence d'égalité des chances (avant): {eod_before:.3f}")
                    st.write(f"Différence d'égalité des chances (après): {eod_after:.3f}")
                    
                    # Visualisation des taux de faux positifs et faux négatifs
                    for group in np.unique(X_test[sensitive_features[0]]):
                        mask = X_test[sensitive_features[0]] == group
                        group_name = encoders[sensitive_features[0]].inverse_transform([group])[0] if sensitive_features[0] in encoders else group
                        
                        st.write(f"\nGroupe: {group_name}")
                        
                        # Matrice de confusion pour le modèle original
                        cm_original = confusion_matrix(y_test[mask], y_pred[mask])
                        fpr_original = cm_original[0, 1] / (cm_original[0, 0] + cm_original[0, 1])
                        fnr_original = cm_original[1, 0] / (cm_original[1, 0] + cm_original[1, 1])
                        
                        # Matrice de confusion pour le modèle équitable
                        cm_fair = confusion_matrix(y_test[mask], y_pred_fair[mask])
                        fpr_fair = cm_fair[0, 1] / (cm_fair[0, 0] + cm_fair[0, 1])
                        fnr_fair = cm_fair[1, 0] / (cm_fair[1, 0] + cm_fair[1, 1])
                        
                        rates_df = pd.DataFrame({
                            'Taux': ['Taux de Faux Positifs', 'Taux de Faux Négatifs'],
                            'Original': [fpr_original, fnr_original],
                            'Équitable': [fpr_fair, fnr_fair]
                        })
                        
                        fig = px.bar(
                            rates_df,
                            x='Taux',
                            y=['Original', 'Équitable'],
                            barmode='group',
                            title=f"Taux d'erreur pour le groupe {group_name}"
                        )
                        st.plotly_chart(fig)
                
            except Exception as e:
                st.error(f"Une erreur s'est produite lors de l'application de la technique de mitigation: {str(e)}")
                st.write("Détails de l'erreur pour le débogage:", e.__class__.__name__) 