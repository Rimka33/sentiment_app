import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Configuration de la page
st.set_page_config(
    page_title="Analyse de Sentiment des Tweets",
    page_icon="📊",
    layout="wide"
)

# Titre de l'application
st.title("📊 Analyse de Sentiment des Tweets")

# Chargement des données
@st.cache_data
def load_data():
    train_data = pd.read_csv('train.csv')
    # Nettoyage des données
    train_data = train_data.dropna(subset=['text', 'sentiment'])  # Supprime les lignes avec NaN
    train_data = train_data[train_data['text'].str.strip().astype(bool)]  # Supprime les textes vides
    return train_data

# Entraînement et sauvegarde du modèle
def train_and_save_model():
    train_data = load_data()
    
    # Vectorisation du texte
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(train_data['text'])
    y = train_data['sentiment']
    
    # Entraînement du modèle
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Sauvegarde du modèle et du vectorizer
    joblib.dump(model, 'sentiment_model.joblib')
    joblib.dump(vectorizer, 'vectorizer.joblib')
    
    return model, vectorizer

# Chargement du modèle
def load_model():
    if not os.path.exists('sentiment_model.joblib') or not os.path.exists('vectorizer.joblib'):
        return train_and_save_model()
    
    model = joblib.load('sentiment_model.joblib')
    vectorizer = joblib.load('vectorizer.joblib')
    return model, vectorizer

# Interface utilisateur
st.write("Entrez un tweet pour analyser son sentiment :")

# Zone de texte pour l'entrée
tweet = st.text_area("Tweet", height=100)

# Bouton d'analyse
if st.button("Analyser le sentiment"):
    if tweet:
        # Chargement du modèle et du vectorizer
        model, vectorizer = load_model()
        
        # Prédiction
        X_new = vectorizer.transform([tweet])
        prediction = model.predict(X_new)[0]
        probabilities = model.predict_proba(X_new)[0]
        
        # Affichage des résultats
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Résultats")
            st.write(f"Sentiment prédit : {prediction}")
            st.write("Probabilités :")
            for sentiment, prob in zip(model.classes_, probabilities):
                st.write(f"- {sentiment}: {prob:.2%}")
        
        with col2:
            st.subheader("Visualisation")
            # Création du graphique avec Plotly
            df = pd.DataFrame({
                'Sentiment': model.classes_,
                'Probabilité': probabilities
            })
            fig = px.bar(df, x='Sentiment', y='Probabilité', 
                        color='Sentiment',
                        color_discrete_map={
                            'positive': 'green',
                            'negative': 'red',
                            'neutral': 'blue'
                        })
            fig.update_layout(yaxis_range=[0, 1])
            st.plotly_chart(fig)
            
        # Statistiques des données d'entraînement
        st.subheader("Statistiques des données d'entraînement")
        train_data = load_data()
        sentiment_counts = train_data['sentiment'].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Distribution des sentiments :")
            fig = px.pie(values=sentiment_counts.values,
                        names=sentiment_counts.index,
                        title='Distribution des sentiments dans les données d\'entraînement')
            st.plotly_chart(fig)
        
        with col2:
            st.write("Nombre de tweets par sentiment :")
            st.dataframe(sentiment_counts)
    else:
        st.warning("Veuillez entrer un tweet à analyser.")

# Pied de page
st.markdown("---")
st.markdown("Développé avec ❤️ par [Votre Nom]") 