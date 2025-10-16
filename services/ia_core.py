# services/ia_core.py

import json
import joblib 
import numpy as np
import os
import logging
import pandas as pd 
from typing import Dict, Any, List
from datetime import datetime
from functools import lru_cache

# A função 'preprocessing.classify_product_line' e 'CAUSA_RAIZ_MAP' 
# devem ser acessíveis ou importadas do seu módulo 'preprocessing'.
from .preprocessing import classify_product_line, CAUSA_RAIZ_MAP 

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ----------------------------------------------------
# REMOVIDO: Importações de torch, transformers e HAS_NLP
# ----------------------------------------------------

# --- LOGICA DE REGRA DE NEGÓCIO SIMULADA (MANTIDA) ---
LOGICA_MODELO_SIMULADO = {
    ("QUEBRA DO PINO", "MONTAGEM MECÂNICA"): "Ajustar o torque da ferramenta pneumática (limite em 5Nm).",
    ("FALHA DE SOLDA", "SMT"): "Revisar o perfil de temperatura do forno e a pasta de solda utilizada.",
    ("CURTO CIRCUITO", "TESTE FUNCIONAL"): "Aumentar a inspeção visual na etapa SMT e verificar o alinhamento de componentes críticos.",
    ("FALHA DE COMPONENTE", "COMPRA/RECEBIMENTO"): "Notificar o fornecedor e solicitar análise de lote do componente X.",
}

MODEL_FILE = 'checklist_predictor_model.joblib'
CLASSES_FILE = 'checklist_classes.json'

ML_MODEL_PIPELINE = None
TIPOS_DE_FALHA = [] 

# ----------------------------------------------------
# REMOVIDO: topic_pipeline = None (Não é mais necessário)
# ----------------------------------------------------

@lru_cache(maxsize=1)
def get_ml_model():
    """Carrega o modelo Scikit-learn, uma única vez na primeira chamada, e armazena em cache."""
    global TIPOS_DE_FALHA, ML_MODEL_PIPELINE 
    
    if ML_MODEL_PIPELINE is not None:
        return ML_MODEL_PIPELINE 

    if os.path.exists(MODEL_FILE):
        try:
            model = joblib.load(MODEL_FILE)
            with open(CLASSES_FILE, 'r') as f:
                TIPOS_DE_FALHA = json.load(f) 
            
            ML_MODEL_PIPELINE = model 
            
            logger.info(f"[IA] Modelo Scikit-learn carregado LAZY. Classes: {len(TIPOS_DE_FALHA)}")
            return ML_MODEL_PIPELINE
        except Exception as e:
            logger.error(f"[IA] ERRO ao carregar Scikit-learn: {e}.")
            ML_MODEL_PIPELINE = None 
            return None
    logger.warning("[IA] AVISO: Modelo Scikit-learn não encontrado. Retornando None.")
    return None

# ----------------------------------------------------
# REMOVIDO: carregar_modelos_ia_nlp_only()
# REMOVIDO: classificar_observacao_topico(text: str) -> str
# REMOVIDO: A chamada global 'carregar_modelos_ia_nlp_only()'
# ----------------------------------------------------


# --- LÓGICA DO MODELO (ANÁLISE EM TEMPO REAL) ---

def analisar_checklist(dados_checklist: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executa todas as análises de domínio, regras e ML para um único registro de falha.
    Retorna um dicionário com os resultados.
    """
    model_pipeline = get_ml_model() 
    global TIPOS_DE_FALHA
    
    # 1. Extração Segura de Features
    falha = dados_checklist.get('falha', '').strip()
    setor = dados_checklist.get('setor', '').strip()
    produto = dados_checklist.get('produto', '').strip()
    # Observação é combinada, mas a classificação do TÓPICO (NLP) NÃO É MAIS FEITA AQUI.
    obs_prod = dados_checklist.get('observacao_producao', '') if isinstance(dados_checklist.get('observacao_producao'), str) else ''
    obs_ass = dados_checklist.get('observacao_assistencia', '') if isinstance(dados_checklist.get('observacao_assistencia'), str) else ''
    observacao = f"{obs_prod} {obs_ass}".strip() # Mantida, mas não classificada por NLP local
    
    # 2. Análise de Domínio
    causa_raiz_sugerida = CAUSA_RAIZ_MAP.get(falha, 'Causa Indeterminada')
    linha_produto = classify_product_line(produto)
    
    # 🚨 topico_ia é definido como N/A, pois a classificação local foi removida
    topico_ia = "N/A - Gemini Analisa via Query" 
    
    mensagem_base = (
        f"**Causa Raiz Sugerida (Domínio):** {causa_raiz_sugerida}. "
        f"**Linha de Produto:** {linha_produto}. "
        f"**Tópico Inferido (IA):** {topico_ia}."
    )

    resultado = {
        "timestamp_analise": datetime.now().isoformat(),
        "causa_raiz_dominio": causa_raiz_sugerida,
        "linha_produto": linha_produto,
        "topico_ia": topico_ia, 
        "status": "Análise de Domínio", # Default
        "mensagem": mensagem_base
    }
    
    # 3. Análise Baseada em Regras (Base de Conhecimento)
    chave = (falha, setor)
    recomendacao_simulada = LOGICA_MODELO_SIMULADO.get(chave)
    if recomendacao_simulada:
        resultado.update({
            "status": "Recomendação Encontrada (Base de Conhecimento)",
            "recomendacao": recomendacao_simulada,
            "mensagem": f"{mensagem_base} **RECOMENDAÇÃO DE AÇÃO:** {recomendacao_simulada}"
        })
        return resultado
            
    # 4. Análise Preditiva (Scikit-learn)
    if model_pipeline is not None:
        try:
            # Prepara o DataFrame para o Pipeline (uso local de pandas)
            data_for_df = {
                'produto': [produto], 
                'quantidade': [dados_checklist.get('quantidade', 1)], 
                'setor': [setor], 
                # Adiciona colunas que podem ser esperadas pelo pipeline, com fallback seguro
                'localizacao_componente': [dados_checklist.get('localizacao_componente', '')], 
                'lado_placa': [dados_checklist.get('lado_placa', '')]
            }
            df_predict = pd.DataFrame.from_dict(data_for_df)
            
            probabilities = model_pipeline.predict_proba(df_predict)[0]
            predicted_index = np.argmax(probabilities)
            predicted_falha = TIPOS_DE_FALHA[predicted_index]
            confidence = probabilities[predicted_index] * 100
            
            status_ia = "ALERTA: Previsão de Alto Risco" if confidence > 70 else "Análise ML Sugestiva"
            mensagem_ml = f"Probabilidade ({confidence:.2f}%) de a falha real ser **{predicted_falha}** (Input: {falha if falha else 'N/A'})."
            
            resultado.update({
                "status": status_ia,
                "previsao_falha_ml": predicted_falha,
                "confianca": f"{confidence:.2f}%",
                "mensagem": f"{mensagem_base} {mensagem_ml}"
            })
            return resultado
            
        except Exception as e:
            logger.exception(f"Erro na previsão ML. Retornando análise de domínio. Erro: {e}") 
            
    # 5. Retorno Padrão (Fallback)
    resultado.update({
        "status": "Análise de Domínio (ML Indisponível)",
        "mensagem": mensagem_base + " Modelo de previsão ML indisponível ou falhou."
    })
    return resultado


def analisar_checklist_multifalha(lista_de_falhas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Executa a análise de IA para uma lista de falhas. (Lógica de loop inalterada)
    """
    
    if not isinstance(lista_de_falhas, list) or not lista_de_falhas:
        return []
    
    resultados_consolidados = []
    
    for i, falha_data in enumerate(lista_de_falhas):
        try:
            # Reutiliza a função de análise de falha única
            resultado_analise = analisar_checklist(falha_data)
            
            # Adiciona o índice original para rastreamento
            resultado_analise['falha_index'] = i
            
            resultados_consolidados.append(resultado_analise)
            
        except Exception as e:
            logger.error(f"Erro ao analisar falha {i} em multifalha: {e}")
            resultados_consolidados.append({
                "falha_index": i,
                "status": "ERRO",
                "mensagem": f"Falha na análise da IA: {e}"
            })
            
    return resultados_consolidados