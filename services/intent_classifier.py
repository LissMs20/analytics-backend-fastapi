import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from typing import List, Optional
import unicodedata
import re
import numpy as np

# --- ⚙️ Função de Pré-processamento e Normalização (Exportada para uso externo, se necessário) ---
def normalize_text(text: str) -> str:
    """Aplica tokenização leve, minúsculas, remove acentos e caracteres não-alfanuméricos."""
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    # 1. Remove acentos
    text = unicodedata.normalize("NFD", text).encode("ascii", "ignore").decode("utf-8")
    # 2. Remove caracteres especiais, mantendo apenas letras, números e espaços
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # 3. Normaliza múltiplos espaços
    text = re.sub(r"\s+", " ", text)
    return text

class IntentClassifier:
    """
    Classificador de intenção local (fallback) usando TF-IDF e Regressão Logística.
    Robusto a variações de texto e desbalanceamento de classes.
    """
    def __init__(self):
        # n-grams (1,2) para capturar termos como "taxa rejeição"
        self.vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=1, max_df=0.9)
        # class_weight='balanced' e max_iter=500 para estabilidade
        self.model = LogisticRegression(max_iter=500, class_weight='balanced', random_state=42)
        self.is_trained = False

    def train(self, texts: List[str], labels: List[str]):
        """Treina o modelo com o conjunto de dados fornecido."""
        
        # Pré-processamento de Normalização ANTES de vetorizar (Melhoria 1)
        normalized_texts = [normalize_text(t) for t in texts]
        
        X = self.vectorizer.fit_transform(normalized_texts)
        self.model.fit(X, labels)
        self.is_trained = True
        
        # 📊 Avaliação de acurácia usando cross-validation (Melhoria 2)
        try:
            # Garante cv não maior que o número de classes ou 5
            cv_val = min(5, len(set(labels))) 
            scores = cross_val_score(self.model, X, labels, cv=cv_val)
            print(f"Acurácia média (cross-validation, cv={cv_val}): {scores.mean():.2f}")
        except Exception as e:
            print(f"Aviso: Não foi possível rodar cross-validation. Detalhe: {e}")


    def predict(self, text: str) -> str:
        """Prevê a intenção de uma única string de texto."""
        if not self.is_trained:
            return "general" # Fallback seguro
            
        normalized_text = normalize_text(text)
        X = self.vectorizer.transform([normalized_text])
        return self.model.predict(X)[0]

    def predict_proba(self, text: str) -> Optional[np.ndarray]:
        """Retorna as probabilidades de intenção."""
        if not self.is_trained:
            return None
            
        normalized_text = normalize_text(text)
        X = self.vectorizer.transform([normalized_text])
        return self.model.predict_proba(X)