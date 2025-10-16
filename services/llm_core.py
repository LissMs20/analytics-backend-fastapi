import os
import pandas as pd
import json
from typing import Dict, Any, List
from google import genai
from google.genai import types
from google.genai.errors import APIError

# Configuração do cliente Gemini (certifique-se de que a variável de ambiente está definida)
# A biblioteca cliente do Google geralmente busca a chave de forma automática
try:
    client = genai.Client()
except Exception as e:
    # Fallback caso a inicialização falhe (ex: chave não configurada)
    print(f"Erro ao inicializar o cliente Gemini: {e}")
    client = None

def format_data_for_llm(df: pd.DataFrame) -> str:
    """
    Formata as observações combinadas em uma string simples para o LLM.
    Limita o número de registros para evitar estourar o limite de tokens.
    """
    # Usaremos apenas as colunas relevantes para a análise de tópicos
    context_df = df[['documento_id', 'observacao_combinada']].copy()
    context_df = context_df.dropna(subset=['observacao_combinada']).head(500) # Limitar a 500 registros para o teste

    if context_df.empty:
        return "Nenhuma observação de texto livre válida foi encontrada no conjunto de dados para análise."
    
    # Formatação simples para o modelo ler
    formatted_str = "Lista de Observações de Falhas (ID: Texto):\n"
    for index, row in context_df.iterrows():
        # Usamos o ID do documento para rastreabilidade
        formatted_str += f"{row['documento_id']}: \"{row['observacao_combinada'].replace('\n', ' ').strip()}\"\n"
        
    return formatted_str

async def analyze_observations_with_gemini(df: pd.DataFrame, query: str) -> Dict[str, Any]:
    """
    Chama a API Gemini para classificar e resumir os tópicos das observações.
    """
    if client is None:
        return {'status': 'ERROR', 'error': 'Cliente Gemini não inicializado. Verifique a configuração da API KEY.'}

    # 1. Preparar os dados de observação
    observations_context = format_data_for_llm(df)

    if observations_context.startswith("Nenhuma"):
          return {'status': 'FAIL', 'error': observations_context}

    # 2. Definir o Prompt e o Formato da Resposta
    # Usamos o query original do usuário como contexto adicional
    prompt_instruction = f"""
    Você é um Analista de Qualidade industrial focado em placas eletrônicas e processos de manufatura (SMT).
    Sua tarefa é analisar a lista de Observações de Falhas fornecida abaixo para identificar os 5 tópicos ou causas-raiz mais recorrentes no texto.
    
    A pergunta do usuário é: "{query}"

    Passos da análise:
    1. Para cada registro, **classifique-o em um único TÓPICO (ex: 'Solda Incorreta', 'Componente Danificado', 'Erro Operacional', 'Limpeza/Contaminação')**. Mantenha os tópicos concisos (máximo 3 palavras).
    2. Conte a frequência de cada tópico.
    3. Identifique os 5 principais tópicos.
    4. Crie um resumo (máximo 3 frases) em Português sobre o que a análise de tópicos revela.
    
    INFORMAÇÕES DE CONTEXTO:
    ---
    {observations_context}
    ---
    
    Gere a resposta EXCLUSIVAMENTE no formato JSON, conforme o schema abaixo.
    A chave 'topicos' deve ser uma lista de objetos com 'nome' e 'contagem'.
    """

    # 3. Configurar a Geração (com uso do pydantic para forçar o formato)
    json_schema = {
        "type": "object",
        "properties": {
            "resumo_ia": {"type": "string", "description": "Resumo analítico dos tópicos de falha, em Português."},
            "topicos_ia": {
                "type": "array",
                "description": "Lista dos 5 tópicos mais frequentes e suas contagens.",
                "items": {
                    "type": "object",
                    "properties": {
                        "nome": {"type": "string", "description": "Nome conciso do tópico de falha."},
                        "contagem": {"type": "integer", "description": "Número de observações classificadas neste tópico."}
                    },
                    "required": ["nome", "contagem"]
                }
            }
        },
        "required": ["resumo_ia", "topicos_ia"]
    }
    
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=json_schema
    )

    try:
        # ATENÇÃO: A chamada do cliente Python do Google AI é síncrona.
        # Embora a função seja 'async def', esta chamada bloqueará o loop de eventos.
        # Em produção com FastAPI, é recomendado usar `await run_in_threadpool`
        # para mover esta operação para um thread separado e evitar bloqueio.
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt_instruction,
            config=config,
        )

        # O retorno é uma string JSON, que precisa ser parseada
        if not response.text:
            return {'status': 'FAIL', 'error': 'Resposta vazia do modelo Gemini.'}
            
        llm_analysis_data = json.loads(response.text)
        
        return {
            'status': 'OK',
            'summary': llm_analysis_data.get('resumo_ia', 'Resumo não gerado pelo modelo.'),
            'topics_data': llm_analysis_data.get('topicos_ia', [])
        }
    
    except APIError as e:
        return {'status': 'ERROR', 'error': f'Erro na API Gemini: {e}'}
    except json.JSONDecodeError:
        return {'status': 'ERROR', 'error': 'O modelo Gemini não retornou JSON válido. Tente refinar o prompt.'}
    except Exception as e:
        return {'status': 'ERROR', 'error': f'Erro desconhecido: {e}'}

# --- FUNÇÃO QUE ESTAVA FALTANDO PARA RESOLVER O IMPORTERROR ---

async def summarize_analysis_with_gemini(analysis_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Gera um insight estratégico em linguagem natural a partir dos resultados combinados
    das análises estatísticas (Causa Raiz) e da análise de tópicos do LLM.
    
    analysis_data deve conter as chaves 'summary' (str) e 'topics_data' (list).
    """
    if client is None:
        return {'status': 'ERROR', 'error': 'Cliente Gemini não inicializado.'}
    
    # 1. Extrai dados para o prompt
    statistical_summary = analysis_data.get('summary', 'Análise estatística não disponível.')
    topic_insights = analysis_data.get('topics_data', [])
    
    # Formata os tópicos da IA em um formato legível
    formatted_topics = ""
    for item in topic_insights:
        formatted_topics += f"- Tópico: {item.get('nome', 'N/A')} (Contagem: {item.get('contagem', 0)})\n"
        
    prompt = f"""
        Você é um **Estrategista Sênior de Manufatura** focado em otimização de processos (Lean Six Sigma).
        Seu papel é consolidar as seguintes descobertas em um único Insight Estratégico para a Diretoria, 
        com foco em ações de alto impacto.

        1. **Análise Estatística Chave (Origem/Causa Raiz):**
        {statistical_summary}

        2. **Principais Tópicos Identificados por IA nas Observações Livres:**
        {formatted_topics or 'Nenhuma informação de tópicos da IA disponível.'}

        Sua tarefa: Crie um **único parágrafo (máximo 4 frases)** conciso e profissional em Português que:
        a) Sintetize a descoberta mais crítica (o que está causando o maior problema, combinando estatística e texto livre).
        b) Apresente uma recomendação de ação estratégica de alto nível (o que deve ser feito).
        
        Sua resposta DEVE ser apenas o texto do insight.
    """
    
    try:
        # ATENÇÃO: Chamada síncrona dentro de função assíncrona.
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )

        return {
            'status': 'OK',
            'strategic_insight': response.text
        }
    
    except APIError as e:
        return {'status': 'ERROR', 'error': f'Erro na API Gemini durante a sumarização: {e}'}
    except Exception as e:
        return {'status': 'ERROR', 'error': f'Erro desconhecido na sumarização: {e}'}
