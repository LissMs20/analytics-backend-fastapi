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

# 🚨 CORREÇÃO: Remove a importação de LOGICA_MODELO_SIMULADO 
# Importações de domínio do módulo de pré-processamento
from .preprocessing import classify_product_line, CAUSA_RAIZ_MAP # LOGICA_MODELO_SIMULADO removida daqui

# Configuração de logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Importação condicional do torch (para NLP)
try:
    import torch 
    from transformers import pipeline
    HAS_NLP = True
except ImportError:
    logger.warning("[IA] 'torch' e 'transformers' não instalados. Funções de NLP desativadas.")
    torch = None
    pipeline = None
    HAS_NLP = False

# --- CONSTANTE DE LÓGICA DE REGRAS (Movida para cá) ---
# 🚨 Definição da lógica que estava causando o erro de importação 🚨
LOGICA_MODELO_SIMULADO = {
    # (Falha, Setor): Recomendação
    ("QUEBRA DO PINO", "MONTAGEM MECÂNICA"): "Ajustar o torque da ferramenta pneumática (limite em 5Nm).",
    ("FALHA DE SOLDA", "SMT"): "Revisar o perfil de temperatura do forno e a pasta de solda utilizada.",
    ("CURTO CIRCUITO", "TESTE FUNCIONAL"): "Aumentar a inspeção visual na etapa SMT e verificar o alinhamento de componentes críticos.",
    ("FALHA DE COMPONENTE", "COMPRA/RECEBIMENTO"): "Notificar o fornecedor e solicitar análise de lote do componente X.",
}
# ---------------------------------------------------------------------

# --- Configurações de Arquivo Scikit-learn ---
MODEL_FILE = 'checklist_predictor_model.joblib'
CLASSES_FILE = 'checklist_classes.json'
# ---------------------------------------------

# Variáveis globais para os modelos (Escopo do módulo)
model_pipeline = None 
TIPOS_DE_FALHA = [] 
topic_pipeline = None 

# --- FUNÇÕES DE CARREGAMENTO E CLASSIFICAÇÃO ---

def carregar_modelos_ia():
# ... [O restante da função carregar_modelos_ia() permanece inalterado] ...
    """Carrega o modelo Scikit-learn e o Transformer do disco/HuggingFace."""
    global model_pipeline, TIPOS_DE_FALHA, topic_pipeline
    
    # 1. Carregar Modelo Scikit-learn (Previsão de Falhas)
    if os.path.exists(MODEL_FILE):
        try:
            model_pipeline = joblib.load(MODEL_FILE)
            with open(CLASSES_FILE, 'r') as f:
                TIPOS_DE_FALHA = json.load(f)
            logger.info(f"[IA] Modelo de Checklist Scikit-learn carregado. Classes: {len(TIPOS_DE_FALHA)}")
        except Exception as e:
            logger.error(f"[IA] ERRO ao carregar Scikit-learn: {e}. Execute 'python train_models.py'.")
            model_pipeline = None
    else:
        logger.warning("[IA] AVISO: Modelo Scikit-learn não encontrado. Execute o script de treino.")

    # 2. Carregar Modelo Transformer (Classificação de Tópico)
    if HAS_NLP:
        try:
            device_id = 0 if torch.cuda.is_available() else -1
            topic_pipeline = pipeline(
                "zero-shot-classification", 
                model="facebook/bart-large-mnli", 
                device=device_id
            )
            logger.info(f"[IA] Modelo Transformer (Classificação Zero-Shot) carregado com sucesso (Device: {device_id}).")
        except Exception as e:
            logger.warning(f"[IA] AVISO: Falha ao carregar o Transformer: {e}")
            topic_pipeline = None

@lru_cache(maxsize=128)
def classificar_observacao_topico(text: str) -> str:
# ... [A função classificar_observacao_topico() permanece inalterada] ...
    """Classifica o tópico de uma observação usando o Transformer."""
    global topic_pipeline
    
    TOPIC_LABELS = [
        "Problema de Máquina", "Problema de Material/Componente", 
        "Problema de Processo/Ajuste", "Erro Operacional/Humano", 
        "Revisão (Sem Defeito / Inconclusivo)"
    ]

    # NLP só é útil para textos razoavelmente longos
    if not topic_pipeline or not text or len(text.strip()) < 15:
        return "N/A - Texto Curto/Modelo Indisponível"

    try:
        # Garante que o texto não seja muito longo para o modelo (truncamento implícito ou explícito)
        result = topic_pipeline(text, TOPIC_LABELS, multi_label=False)
        return result.get('labels', ['N/A - Inferência Falhou'])[0]
    except Exception as e:
        logger.warning(f"[IA] Falha na inferência NLP: {e}")
        return "ERRO - Inferência"

# Carrega os modelos na inicialização do módulo
carregar_modelos_ia()


# --- LÓGICA DO MODELO (ANÁLISE EM TEMPO REAL) ---

def analisar_checklist(dados_checklist: Dict[str, Any]) -> Dict[str, Any]:
# ... [O restante da função analisar_checklist() permanece inalterado] ...
    """
    Executa todas as análises de domínio, regras e ML/NLP para um único registro de falha.
    Retorna um dicionário com os resultados.
    """
    global model_pipeline, TIPOS_DE_FALHA
    
    # 1. Extração Segura de Features
    falha = dados_checklist.get('falha', '').strip()
    setor = dados_checklist.get('setor', '').strip()
    produto = dados_checklist.get('produto', '').strip()
    # Combina observações de forma robusta, caindo para string vazia
    obs_prod = dados_checklist.get('observacao_producao', '') if isinstance(dados_checklist.get('observacao_producao'), str) else ''
    obs_ass = dados_checklist.get('observacao_assistencia', '') if isinstance(dados_checklist.get('observacao_assistencia'), str) else ''
    observacao = f"{obs_prod} {obs_ass}".strip()
    
    # 2. Análise de Domínio e NLP
    causa_raiz_sugerida = CAUSA_RAIZ_MAP.get(falha, 'Causa Indeterminada')
    linha_produto = classify_product_line(produto)
    topico_ia = classificar_observacao_topico(observacao) 
    
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
    recomendacao_simulada = LOGICA_MODELO_SIMULADO.get(chave) # <--- AGORA USA A CONSTANTE LOCAL
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
# ... [O restante da função analisar_checklist_multifalha() permanece inalterado] ...
    """
    Executa a análise de IA para uma lista de falhas (como em 'falhas_json').
    
    A rota /checklists usa essa função para analisar o campo 'falhas' em background.
    """
    
    if not isinstance(lista_de_falhas, list) or not lista_de_falhas:
        return []
    
    resultados_consolidados = []
    
    for i, falha_data in enumerate(lista_de_falhas):
        try:
            # Reutiliza a função de análise de falha única
            # Presume que 'falha_data' contém as chaves 'falha', 'setor', etc.
            # Nota: O produto e observação de nível superior devem estar presentes em 'falha_data'
            # se estiver sendo chamado diretamente após um 'flatten'. Se for apenas a lista 
            # aninhada, as chaves de produto/observacao podem estar ausentes e resultar em N/A.
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