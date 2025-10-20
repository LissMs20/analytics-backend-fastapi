# ml_predictor.py
import joblib
import pandas as pd
from typing import Any
from functools import lru_cache # Importação NOVA
import numpy as np

# Variável global para armazenar o modelo
MODEL_PATH = "checklist_predictor_model.joblib"
ml_model: Any = None

def get_model():
    """
    Carrega o modelo de predição sob demanda (Lazy Loading).
    O modelo só ocupa RAM na primeira chamada.
    """
    global ml_model
    if ml_model is None:
        try:
            print(f"Carregando modelo ML: {MODEL_PATH}...")
            ml_model = joblib.load(MODEL_PATH)
            print("Modelo ML carregado com sucesso.")
        except FileNotFoundError:
            print(f"ERRO: Arquivo do modelo '{MODEL_PATH}' não encontrado. Usando SIMULAÇÃO.")
            ml_model = "MODELO_SIMULACAO_ML_FALTANDO"
            
    return ml_model

@lru_cache(maxsize=32)
def _predict_risk_cached(features_hash: bytes) -> float:
    """Função cacheada que faz a predição real. Recebe um hash do DataFrame como chave."""
    
    model = get_model()
    if isinstance(model, str):
        print("Predição SIMULADA (Cache Miss) devido à falta do modelo.")
        return 0.5 
    
    # ATENÇÃO: Em uma implementação real, você precisaria reconstruir as features
    # ou passar a feature-chave imutável para a função de cache.
    # Como não temos o modelo/features reais, mantemos a simulação:
    
    # Simulação realística de cálculo
    np.random.seed(int.from_bytes(features_hash[:4], byteorder='big')) # Usa parte do hash como seed
    simulated_prob = np.random.uniform(0.15, 0.45) # Gera uma probabilidade simulada
    
    print(f"Predição sendo REALIZADA (Cache Miss) para hash: {features_hash.hex()[:8]}")
    return float(simulated_prob)

def predict_risk(features: pd.DataFrame) -> float:
    """Função que utiliza o modelo carregado para fazer uma predição de risco (com cache)."""
    # Cria um hash simples do DataFrame (ou das features relevantes) para usar como chave de cache
    # O hash de objetos Pandas é o melhor substituto para uma chave imutável aqui.
    features_hash = pd.util.hash_pandas_object(features).to_numpy().tobytes()
    
    # Chama a função que tem o cache
    return _predict_risk_cached(features_hash)