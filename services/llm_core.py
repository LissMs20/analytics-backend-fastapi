# services/llm_core.py

import os
import pandas as pd
import json
from typing import Dict, Any, List
from google import genai
from google.genai import types
from google.genai.errors import APIError
from starlette.concurrency import run_in_threadpool 

# --- CONFIGURAÇÃO DO CLIENTE GEMINI ---
client = None
GEMINI_MODEL = "gemini-2.5-flash" # Modelo rápido para classificação e sumarização

try:
    # A variável de ambiente GEMINI_API_KEY deve ser configurada no ambiente (Render)
    client = genai.Client()
    print("INFO: Cliente Gemini inicializado com sucesso.")
except Exception as e:
    # Este log é crucial para identificar falha na chave de API
    print(f"ERRO CRÍTICO: Cliente Gemini NÃO inicializado. Verifique a API KEY. Detalhe: {e}")

# --- FUNÇÕES AUXILIARES DE FORMATAÇÃO ---
def format_data_for_llm(df: pd.DataFrame) -> str:
    """
    Formata as observações combinadas em uma string simples para o LLM.
    Reduzir este limite é a MELHOR forma de evitar timeout.
    """
    context_df = df[['documento_id', 'observacao_combinada']].copy()
    # 🚨 OTIMIZAÇÃO: Limite de 100 registros para evitar timeout.
    context_df = context_df.dropna(subset=['observacao_combinada']).head(100)

    if context_df.empty:
        return "Nenhuma observação de texto livre válida foi encontrada no conjunto de dados para análise."
    
    formatted_str = "Lista de Observações de Falhas (ID: Texto):\n"
    for index, row in context_df.iterrows():
        # Limita o texto de cada observação (200 chars) para garantir que não haja estouro de token
        text_content = row['observacao_combinada'].replace('\n', ' ').strip()
        formatted_str += f"{row['documento_id']}: \"{text_content[:200]}\"\n" 
        
    return formatted_str

# --- Funções Auxiliares Síncronas para o Threadpool ---
def _generate_content_sync(model: str, contents: str, config: types.GenerateContentConfig = None):
    """Função síncrona que chama a API Gemini. Rodará em um threadpool."""
    if client is None:
        raise Exception('Cliente Gemini não está inicializado para chamada síncrona.')
        
    return client.models.generate_content(
        model=model,
        contents=contents,
        config=config,
    )
# --------------------------------------------------------

# --- 1. CLASSIFICAÇÃO DE INTENÇÃO (Intenção do Usuário) ---

INTENT_SCHEMA = {
    "type": "array",
    "description": "Lista de intenções detectadas na consulta do usuário.",
    "items": {
        "type": "string",
        "enum": ["qualidade", "setor", "causa_raiz", "nlp", "default"]
    }
}

async def classify_query_intent(query: str) -> List[str]:
    """
    Classifica a intenção da consulta do usuário usando Gemini, retornando uma lista JSON estruturada.
    """
    if client is None:
        print("AVISO: classify_query_intent - Cliente Gemini não inicializado. Usando default.")
        # Retorna default para evitar que a falha do cliente quebre o fluxo
        return ["default"] 

    prompt = f"""
    Sua tarefa é classificar a intenção da seguinte consulta do usuário, usando APENAS as categorias pré-definidas.
    
    Categorias:
    - qualidade: Se a consulta focar em métricas (dppm, rejeição, falha, taxa, tendência, percentual).
    - setor: Se a consulta focar em áreas, departamentos, localização de origem ou detecção.
    - causa_raiz: Se a consulta focar em processos, linhas de produto, produtos específicos (placas, componentes), ou a raiz do problema.
    - nlp: Se a consulta focar em análise de texto, observações, comentários ou tópicos de texto livre.
    - default: Se a consulta não se encaixar claramente em nenhuma das outras categorias (ex: saudações, ou pedidos genéricos).

    Retorne APENAS um array JSON contendo UMA OU MAIS categorias que se aplicam.

    Consulta do Usuário: "{query}"
    """
    
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=INTENT_SCHEMA,
    )

    try:
        response = await run_in_threadpool(
            _generate_content_sync,
            GEMINI_MODEL,
            prompt,
            config
        )
        
        # O modelo deve retornar um JSON válido (array de strings)
        return json.loads(response.text)
        
    except (APIError, json.JSONDecodeError) as e:
        # Erro de API ou de parsing do JSON
        print(f"ERRO DE CLASSIFICAÇÃO: Falha na intenção. Detalhe: {e}") 
        return ["default"]
    except Exception as e:
        # Erro genérico (ex: falha do run_in_threadpool ao chamar _generate_content_sync)
        print(f"ERRO GENÉRICO CLASSIFICAÇÃO: Detalhe: {e}")
        return ["default"]


# --- 2. ANÁLISE DE OBSERVAÇÕES (NLP) ---

TOPIC_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "resumo_ia": {"type": "string", "description": "Um resumo de 2 a 3 frases dos principais achados e tendências no texto."},
        "topicos_ia": {
            "type": "array",
            "description": "Lista dos 5 tópicos mais comuns extraídos do texto, com contagem associada.",
            "items": {
                "type": "object",
                "properties": {
                    "nome": {"type": "string", "description": "Nome do tópico ou causa raiz principal."},
                    "contagem": {"type": "integer", "description": "Contagem de ocorrências deste tópico na lista de observações."}
                },
                "required": ["nome", "contagem"]
            }
        }
    },
    "required": ["resumo_ia", "topicos_ia"]
}


async def analyze_observations_with_gemini(df: pd.DataFrame, query: str) -> Dict[str, Any]:
    """
    Chama a API Gemini para classificar e resumir os tópicos das observações.
    """
    if client is None:
        print("ERRO CRÍTICO: analyze_observations_with_gemini - Cliente Gemini não inicializado.")
        return {'status': 'ERROR', 'error': 'Cliente Gemini não inicializado. Verifique a configuração da API KEY.'}

    observations_context = format_data_for_llm(df)

    if observations_context.startswith("Nenhuma"):
        return {'status': 'FAIL', 'error': observations_context}

    prompt_instruction = f"""
    Você é um especialista em Análise de Causa Raiz de Manufatura.
    Com base na lista de observações abaixo, realize uma Análise de Tópicos (NLP) para identificar as 5 principais causas raiz ou temas que estão sendo relatados nos comentários.

    1. GERE uma análise concisa (resumo_ia) de 2 a 3 frases.
    2. CLASSIFIQUE os 5 tópicos mais relevantes (topicos_ia) e conte quantas vezes cada tópico ou sinônimo é mencionado (contagem).

    Dados de Observação:
    ---
    {observations_context}
    ---

    Consulta do Usuário (Para Contexto): {query}

    Retorne o resultado estritamente no formato JSON definido.
    """
    
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=TOPIC_ANALYSIS_SCHEMA
    )

    try:
        # PONTO CHAVE: Usando run_in_threadpool
        response = await run_in_threadpool(
            _generate_content_sync,
            GEMINI_MODEL,
            prompt_instruction,
            config
        )

        if not response.text:
            print("ERRO NLP: Resposta vazia do modelo Gemini.")
            return {'status': 'FAIL', 'error': 'Resposta vazia do modelo Gemini.'}
            
        llm_analysis_data = json.loads(response.text)
        
        return {
            'status': 'OK',
            'summary': llm_analysis_data.get('resumo_ia', 'Resumo não gerado pelo modelo.'),
            'topics_data': llm_analysis_data.get('topicos_ia', [])
        }
    
    except APIError as e:
        print(f"ERRO API (NLP): Falha na análise de observações. Detalhe: {e}")
        return {'status': 'ERROR', 'error': f'Erro na API Gemini: {e}'}
    except json.JSONDecodeError:
        # Tenta pegar a resposta bruta para debug, se possível
        raw_text = response.text if 'response' in locals() and hasattr(response, 'text') else "N/A"
        print(f"ERRO JSON (NLP): Modelo não retornou JSON válido. Detalhe: {raw_text[:100]}...")
        return {'status': 'ERROR', 'error': f'O modelo Gemini não retornou JSON válido. Resposta bruta: {raw_text[:100]}...'}
    except Exception as e:
        print(f"ERRO GENÉRICO (NLP): Erro desconhecido: {e}")
        return {'status': 'ERROR', 'error': f'Erro desconhecido: {e}'}

# --- 3. SUMARIZAÇÃO ESTRATÉGICA (Insight Executivo) ---

async def summarize_analysis_with_gemini(analysis_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Gera um insight estratégico em linguagem natural a partir dos resultados combinados.
    """
    if client is None:
        # Altera o status para 'FAIL' para que o analyst.py não use o resultado
        print("ERRO CRÍTICO: summarize_analysis_with_gemini - Cliente Gemini não inicializado.")
        return {'status': 'FAIL', 'error': 'Cliente Gemini não inicializado.'} 
    
    topic_insights = analysis_data.get('topics_data', [])
    
    formatted_topics = ""
    for item in topic_insights:
        formatted_topics += f"- Tópico: {item.get('nome', 'N/A')} (Contagem: {item.get('contagem', 0)})\n"
        
    prompt = f"""
    Você é um Consultor Estratégico de Qualidade.
    Com base nos dados fornecidos da análise de tópicos, forneça um único parágrafo de 3 a 4 frases de Insight Estratégico.

    Foco do Insight:
    1. CONECTE a causa raiz mais frequente com a necessidade de ação imediata.
    2. SUGIRA o departamento ou processo que deve ser auditado.
    3. CRIE um tom de urgência e clareza.

    Dados da Análise de Tópicos (NLP):
    ---
    {formatted_topics}
    ---
    
    Insight Estratégico:
    """
    
    try:
        # PONTO CHAVE: Usando run_in_threadpool
        response = await run_in_threadpool(
            _generate_content_sync,
            GEMINI_MODEL,
            prompt
        )

        return {
            'status': 'OK',
            'strategic_insight': response.text.strip()
        }
    
    except APIError as e:
        # Log detalhado para o debug da falha 'Análise Avançada da IA'
        print(f"ERRO API (Summarize): Falha ao gerar insight estratégico. Detalhe: {e}")
        return {'status': 'FAIL', 'error': f'Erro na API Gemini durante a sumarização. Verifique a chave e cota de uso: {e}'}
    except Exception as e:
        # Log detalhado para o debug do erro genérico
        print(f"ERRO GENÉRICO (Summarize): Erro desconhecido na sumarização. Detalhe: {e}")
        return {'status': 'FAIL', 'error': f'Erro desconhecido na sumarização: {e}'}