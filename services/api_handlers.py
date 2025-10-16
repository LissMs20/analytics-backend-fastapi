# services/api_handlers.py
import json
import pandas as pd
from typing import Dict, List, Any
from datetime import datetime
import schemas 

# Importações dos módulos modulares
from .preprocessing import prepare_dataframe # extract_period_and_date não é mais necessário aqui
# Importações do ia_core mantidas para a função 'processar_analise_checklist'
from .ia_core import analisar_checklist, analisar_checklist_multifalha

# 🚨 Importação da nova função Gemini 🚨
from .gemini_analyst import handle_query_analysis_gemini

# Importações de schemas para tipagem (Pydantic)
Falha = schemas.Falha
AnalysisResponse = schemas.AnalysisResponse
Tip = schemas.Tip


# --- HANDLERS DE API ---

def processar_analise_checklist(dados_completos: Dict[str, Any], falhas_lista: List[Falha]) -> str:
    """
    Processa uma lista de falhas de um checklist, chamando o núcleo de IA para cada uma.
    Consolida os resultados para uma resposta formatada.
    
    Esta função atua como um 'manipulador' para a API, formatando a entrada e saída.
    A lógica pesada de IA está encapsulada em ia_core.
    """
    # Cria uma cópia dos dados para cada falha, garantindo que as informações de topo
    # como 'produto' e 'quantidade' estejam em cada dicionário de falha para a IA
    lista_para_analise = []
    
    produto_global = dados_completos.get("produto")
    quantidade_global = dados_completos.get("quantidade")
    obs_prod_global = dados_completos.get("observacao_producao", "")
    obs_assist_global = dados_completos.get("observacao_assistencia", "")

    for falha_data in falhas_lista:
        # Assume que 'falha_data' é um objeto Pydantic (Falha) ou um dicionário.
        falha_dict = falha_data.model_dump() if hasattr(falha_data, 'model_dump') else falha_data
        
        dados_para_ia = {
            "produto": produto_global,
            "quantidade": quantidade_global,
            "observacao_producao": obs_prod_global,
            "observacao_assistencia": obs_assist_global,
            **falha_dict 
        }
        lista_para_analise.append(dados_para_ia)

    # Chama a função principal do ia_core para lidar com a lógica de análise em massa
    resultados_analises = analisar_checklist_multifalha(lista_para_analise)
    
    # Consolidação e sumarização para a resposta da API
    riscos = [res.get("status", "") for res in resultados_analises]
    num_alertas = sum(1 for r in riscos if "ALERTA" in r)
    resumo_geral = f"Checklist com **{len(falhas_lista)} falhas**. A IA detectou **{num_alertas} alertas de alto risco**."
    
    resultado_agregado = {
        "tipo": "multi-falha-agregada",
        "timestamp_analise": datetime.now().isoformat(),
        "produto": produto_global,
        "quantidade": quantidade_global,
        "resumo_geral": resumo_geral,
        "analises_individuais": resultados_analises
    }

    return json.dumps(resultado_agregado, indent=2, ensure_ascii=False)


def handle_query_analysis(query: str, data_to_analyze: List[Dict]) -> AnalysisResponse:
    """
    Substitui toda a lógica de if/elif por uma chamada ao Gemini, delegando a
    análise de dados, a lógica estatística e o NLP ao modelo.
    """
    
    # 1. Pré-processamento (Necessário para criar colunas como 'linha_produto', 'dppm_registro', etc.)
    # Este passo é crucial para garantir que o Gemini receba os dados de domínio enriquecidos.
    df_processed = prepare_dataframe(data_to_analyze)
    
    # Converte o DataFrame processado para o formato esperado pelo Gemini
    data_for_gemini = df_processed.to_dict('records')

    if df_processed.empty:
        return AnalysisResponse(
            query=query,
            summary="Nenhum dado encontrado para análise ou dados inválidos após o pré-processamento.",
            tips=[Tip(title="Base de Dados Vazia", detail="Verifique a fonte de dados e o filtro inicial.")]
        )

    # 2. Delega a análise completa ao Gemini (Substitui TODAS as lógicas de if/elif)
    return handle_query_analysis_gemini(query, data_for_gemini)