import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from googletrans import Translator
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self):
        # Chemin vers le modèle entraîné
        self.model_path = "./models/twitter-sentiment-model"
        self.translator = Translator()
        
        try:
            # Essayer de charger le modèle entraîné
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        except:
            # Si le modèle n'existe pas, utiliser le modèle de base
            logger.info("Modèle entraîné non trouvé, utilisation du modèle de base")
            self.model_name = "bert-base-uncased"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=3)
    
    def translate_to_english(self, text):
        try:
            logger.info(f"Texte original : {text}")
            # Vérifier si le texte est vide ou None
            if not text or not text.strip():
                logger.warning("Texte vide reçu")
                return text
            
            # Détecter la langue
            detected = self.translator.detect(text)
            logger.info(f"Langue détectée : {detected.lang}")
            
            # Traduire seulement si ce n'est pas déjà en anglais
            if detected.lang != 'en':
                translation = self.translator.translate(text, dest='en')
                logger.info(f"Texte traduit : {translation.text}")
                return translation.text
            else:
                logger.info("Le texte est déjà en anglais")
                return text
                
        except Exception as e:
            logger.error(f"Erreur de traduction : {str(e)}")
            return text
    
    def preprocess_text(self, text):
        # Nettoyage du texte
        text = str(text).lower().strip()
        logger.info(f"Texte après nettoyage : {text}")
        
        # Traduction en anglais si nécessaire
        text = self.translate_to_english(text)
        logger.info(f"Texte final après prétraitement : {text}")
        return text
    
    def predict(self, text):
        # Prétraitement
        text = self.preprocess_text(text)
        
        # Tokenization
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        
        # Prédiction
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
        # Conversion en probabilités
        probs = predictions.detach().numpy()[0]
        
        # Mapping des indices aux sentiments
        sentiment_map = {0: "Négatif", 1: "Neutre", 2: "Positif"}
        sentiment = sentiment_map[np.argmax(probs)]
        
        logger.info(f"Sentiment prédit : {sentiment}")
        logger.info(f"Probabilités : {probs}")
        
        return {
            "sentiment": sentiment,
            "probabilities": {
                "Négatif": float(probs[0]),
                "Neutre": float(probs[1]),
                "Positif": float(probs[2])
            }
        } 