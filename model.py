import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class SentimentAnalyzer:
    def __init__(self):
        # Chemin vers le modèle entraîné
        self.model_path = "./models/twitter-sentiment-model"
        
        try:
            # Essayer de charger le modèle entraîné
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        except:
            # Si le modèle n'existe pas, utiliser le modèle de base
            print("Modèle entraîné non trouvé, utilisation du modèle de base")
            self.model_name = "bert-base-uncased"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=3)
        
    def preprocess_text(self, text):
        # Nettoyage du texte
        text = str(text).lower()
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
        
        return {
            "sentiment": sentiment,
            "probabilities": {
                "Négatif": float(probs[0]),
                "Neutre": float(probs[1]),
                "Positif": float(probs[2])
            }
        } 