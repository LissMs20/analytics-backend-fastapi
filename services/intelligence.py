# services/intelligence.py (CÓDIGO CORRIGIDO)

import pandas as pd
from typing import Dict, Any, List
import asyncio

# Importa o motor de Análise de Domínio (onde está a lógica complexa de N intenções)
from .analyst import run_domain_analysis_composite
from services import ml_predictor # Importação NOVA

# (Funções predict_with_ml_model e dispatch_heavy_report_task são mantidas, mas precisam 
# da assinatura correta. O predict_with_ml_model precisa do df, mas o dispatch_heavy_report_task 
# é apenas uma simulação, vamos manter.)

async def predict_with_ml_model(df: pd.DataFrame) -> Dict[str, Any]:
    """Faz a predição rápida usando o modelo ML local (joblib) e o cache."""
    await asyncio.sleep(0.01) # Simula um pequeno I/O assíncrono

    total_records = len(df)
    if total_records == 0:
        return {"status": "INFO", "type": "Preditivo", "summary": "Nenhum dado para predição."}
    
    # CHAMA A CAMADA REAL DE PREDIÇÃO COM CACHE
    prob_falha_calculada = ml_predictor.predict_risk(df) 
    
    # Usa a probabilidade calculada/simulada
    prob_falha = min(0.95, prob_falha_calculada + 0.2) 
    
    return {
        "status": "OK",
        "type": "Preditivo",
        "summary": "Análise de Risco ML Concluída.",
        "tips": [
            {"title": "Risco Calculado", "detail": f"Probabilidade de falha no lote: **{prob_falha * 100:.2f}%**."},
            {"title": "Ação Sugerida", "detail": "Ajustar o perfil de temperatura do forno de refusão."}
        ],
        "visualization_data": []
    }

async def dispatch_heavy_report_task(df: pd.DataFrame, report_type: str) -> Dict[str, Any]:
    """
    Simula o disparo de uma tarefa pesada (Celery/Worker) para processamento em background.
    """

    task_id = "TASK-" + pd.util.hash_pandas_object(df).to_numpy().tobytes().hex()[:8]
    
    return {
        "status": "INFO",
        "type": "Worker",
        "summary": f"Relatório de **{report_type}** em processamento no background.",
        "tips": [{"title": "Aguardando", "detail": f"Acompanhe o status da tarefa com ID: {task_id}"}],
        "visualization_data": []
    }

# CORREÇÃO CRUCIAL NA ASSINATURA: O router agora passa o df e a lista de dicts
async def get_strategic_analysis(df: pd.DataFrame, query: str, data_to_analyze: List[Dict]) -> Dict[str, Any]:
    """
    O Orquestrador Central de Inteligência. Decide qual camada de IA usar.
    """
    query_lower = query.strip().lower()

    # 1. INTENÇÃO DE WORKER (Relatório Pesado)
    if "gerar relatório" in query_lower or "cálculo shap" in query_lower or "analisar lote inteiro" in query_lower:
        print(f"Intenção: Relatório Pesado. Disparando Worker...")
        return await dispatch_heavy_report_task(df, "SHAP Completo/Relatório PDF")

    # 2. INTENÇÃO DE ML (Predição)
    if "prever falha" in query_lower or "risco" in query_lower or "probabilidade" in query_lower:
        print(f"Intenção: Predição. Usando Modelo ML local (com cache)...")
        # ML Predictor precisa apenas do DataFrame
        return await predict_with_ml_model(df)
        
    # 3. INTENÇÃO DE ANÁLISE DE DOMÍNIO/COGNITIVA (Estatística, NLP, Tendência, Setor, Causa Raiz)
    # Se não for ML ou Worker, envia para o motor de Análise de Domínio (analyst.py)
    print(f"Intenção: Análise de Domínio/Cognitiva. Chamando run_domain_analysis_composite...")
    
    # CHAMA O MOTOR DE ANÁLISE ESTATÍSTICA/COGNITIVA COMBINADO
    # Ele lida com todas as outras sub-intenções (qualidade, setor, nlp, fallback)
    analysis_result = await run_domain_analysis_composite(query, data_to_analyze)

    # O run_domain_analysis_composite já retorna o Dicionário completo e formatado
    # para ser mapeado no AnalysisResponse, incluindo status, summary, tips e visualization_data.
    return analysis_result