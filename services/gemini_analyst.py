# services/gemini_analyst.py

from google import genai
from google.genai import types
from google.genai.errors import APIError
from typing import Dict, List, Any
import pandas as pd
import json
import os
import logging
import schemas

logger = logging.getLogger(__name__)

# --- CONFIGURAÇÃO INICIAL DO GEMINI ---
try:
    # O cliente procura automaticamente pela variável de ambiente GEMINI_API_KEY
    client = genai.Client()
except Exception:
    logger.error("ERRO: Cliente Gemini não inicializado. Verifique a variável de ambiente GEMINI_API_KEY.")
    client = None

MODEL_NAME = "gemini-2.5-flash"
# -------------------------------------

def format_data_for_prompt(df_analysis: pd.DataFrame) -> str:
    """
    Formata o DataFrame limpo em uma string JSON para o Prompt do Gemini.
    """
    # Usamos as colunas que seu pré-processamento criou/juntou
    relevant_cols = [
        'documento_id', 'produto', 'linha_produto', 'falha_individual', 
        'causa_raiz_processo', 'setor_falha_individual', 
        'quantidade', 'quantidade_produzida', 'dppm_registro', 
        'observacao_combinada'
    ]
    
    cols_to_use = [col for col in relevant_cols if col in df_analysis.columns]

    # Garante que só enviamos os primeiros 50 registros para economizar tokens
    df_sample = df_analysis[cols_to_use].head(50) 
    
    data_list = df_sample.to_dict(orient='records')
    return json.dumps(data_list, indent=2, ensure_ascii=False)


def handle_query_analysis_gemini(analysis_query: str, data_to_analyze: List[Dict]) -> schemas.AnalysisResponse:
    """
    Usa a IA do Gemini para analisar os dados de falhas e produção baseada em uma query
    de linguagem natural, substituindo a lógica complexa de `elif`s do handler original.
    """
    if client is None:
         return schemas.AnalysisResponse(
            query=analysis_query,
            summary="ERRO DE CONFIGURAÇÃO: Cliente Gemini não inicializado.",
            tips=[]
         )

    df = pd.DataFrame(data_to_analyze)
    
    # 1. Pré-processamento e Formatação
    if df.empty or 'quantidade' not in df.columns or df['quantidade'].sum() == 0:
        return schemas.AnalysisResponse(
            query=analysis_query,
            summary="Nenhum dado de falha relevante encontrado para análise.",
            tips=[schemas.Tip(title="Base de Dados Vazia", detail="Ajuste os filtros de busca no banco de dados.")]
        )
        
    formatted_data_string = format_data_for_prompt(df)

    # 2. ENGENHARIA DE PROMPT E INSTRUÇÃO DE SISTEMA (Foco na Análise de Engenharia)
    
    system_instruction = (
        "Você é um Engenheiro de Confiabilidade Sênior. Analise a solicitação do usuário e os dados JSON de falhas. "
        "A análise deve focar em: 1) **DPPM (ou Taxa de Falha)**, 2) **Causa Raiz de Processo**, e 3) **Linha de Produto**. "
        "Não gere texto fora do JSON obrigatório. Seja objetivo e profissional no resumo."
    )
    
    user_prompt = f"""
    # SOLICITAÇÃO DO ENGENHEIRO DE QUALIDADE
    Foco Principal: "{analysis_query}"

    # DADOS BRUTOS (AMOSTRA DE ATÉ 50 REGISTROS FORMATADOS)
    {formatted_data_string}
    
    # CONTEXTO
    A coluna 'falha_individual' representa a falha observada. 
    A coluna 'dppm_registro' é a métrica chave de qualidade (Defeitos por Milhão).
    
    Gere a resposta EXCLUSIVAMENTE no formato JSON estrito, aderindo ao esquema de resposta obrigatório.
    """

    # 3. ESQUEMA DE RESPOSTA (Garante que o Gemini retorne o JSON no formato Pydantic schemas.AnalysisResponse)
    response_schema = types.Schema(
        type=types.Type.OBJECT,
        properties={
            "query": types.Schema(type=types.Type.STRING),
            "summary": types.Schema(type=types.Type.STRING, description="Resumo executivo de 2-3 frases sobre a análise e o risco."),
            "tips": types.Schema(
                type=types.Type.ARRAY,
                description="Duas ou três dicas de ação imediata ou recomendações para o time de Qualidade.",
                items=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "title": types.Schema(type=types.Type.STRING),
                        "detail": types.Schema(type=types.Type.STRING)
                    },
                    required=["title", "detail"]
                )
            ),
            "visualization_data": types.Schema(
                type=types.Type.OBJECT,
                description="Dados para a visualização, focando no top 3 por falha ou dppm.",
                properties={
                    "title": types.Schema(type=types.Type.STRING),
                    "labels": types.Schema(type=types.Type.ARRAY, items=types.Type.STRING),
                    "datasets": types.Schema(
                        type=types.Type.ARRAY, 
                        items=types.Type.OBJECT,
                        properties={
                            "label": types.Schema(type=types.Type.STRING),
                            "data": types.Schema(type=types.Type.ARRAY, items=types.Type.NUMBER),
                            "type": types.Schema(type=types.Type.STRING, description="Ex: 'bar', 'line'"),
                            "backgroundColor": types.Schema(type=types.Type.ARRAY, items=types.Type.STRING, description="Cores em hexadecimal ou rgba.")
                        },
                        required=["label", "data"]
                    )
                },
                required=["title", "labels", "datasets"]
            )
        },
        required=["query", "summary", "tips", "visualization_data"]
    )


    # 4. CHAMADA DA API
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=response_schema,
                temperature=0.0 # Temperatura baixa para resultados analíticos
            )
        )
        
        # Converte a resposta JSON em um dicionário e depois no objeto Pydantic
        analysis_dict = json.loads(response.text)
        return schemas.AnalysisResponse(**analysis_dict)

    except APIError as e:
        logger.error(f"Erro na API do Gemini: {e}")
        return schemas.AnalysisResponse(
            query=analysis_query,
            summary=f"Falha na análise da IA (API Error): {e}",
            tips=[schemas.Tip(title="Verificar API", detail="O limite de requisições ou tokens pode ter sido excedido.")]
        )
    except Exception as e:
        logger.error(f"Erro inesperado no Gemini Handler: {e}")
        return schemas.AnalysisResponse(
            query=analysis_query,
            summary=f"Erro interno ao processar a resposta da IA: {e}",
            tips=[]
        )