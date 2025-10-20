# services/intelligent_fallback.py
import pandas as pd
from typing import Dict, Any, List # <-- LINHA CORRIGIDA
from .explainers import generate_explanation
from .intent_classifier import IntentClassifier 

# As intenções aqui devem mapear para os tipos definidos em explainers.py
INTENT_MAPPING = {
    "quality": "falhas",
    "sector": "setores",
    "individual": "setores", 
    "causas": "causas",
    "general": "general",
    "smt": "causas" 
}

def fallback_analysis(query: str, df: pd.DataFrame, classifier: IntentClassifier) -> Dict[str, Any]:
    """
    Motor de análise que garante uma resposta estruturada baseada em dados
    quando o LLM externo (Gemini) falha.
    """
    
    # 1. Classificação Local 
    intent = classifier.predict(query)
    
    explanation_type = INTENT_MAPPING.get(intent, "general")

    # 2. Geração de Explicação com Dados (Pandas)
    summary, charts = generate_explanation(df, explanation_type)

    if not charts and explanation_type != "general":
        # Se a análise específica falhou (ex: coluna setor não existe), tenta o geral
        explanation_type = "general"
        summary, charts = generate_explanation(df, explanation_type)


    status_summary = f"""
    **Análise de Resiliência (Fallback Local Ativado)**
    
    A IA Principal encontrou um erro de comunicação ou parsing. 
    O sistema ativou o modo Resiliência, analisando a intenção **'{explanation_type.upper()}'** localmente.
    
    **Resumo Técnico:** {summary}
    """
    
    tips = [
        {"title": "Modo Autônomo", "detail": f"O sistema respondeu sem depender de LLM externo. Intenção classificada como '{explanation_type}'."},
    ]
    
    return {
        'status': 'INFO', 
        'summary': status_summary, 
        'visualization_data': charts, 
        'tips': tips,
        'llm_raw_analysis': {'summary': status_summary, 'topics_data': []}
    }