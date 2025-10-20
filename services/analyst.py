# services/analyst.py 
import json
import pandas as pd
import numpy as np
import re
from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncio 
from collections import deque
from fastapi.concurrency import run_in_threadpool
import gc

# Imports necessários para o Fallback Inteligente
import joblib 
import importlib 

from .llm_core import analyze_observations_with_gemini, summarize_analysis_with_gemini, classify_query_intent
from .preprocessing import prepare_dataframe, extract_period_and_date
# IMPORT CORRIGIDO PARA USAR OS NOVOS MÓDULOS DE RESILIÊNCIA E NORMALIZAÇÃO
from .intelligent_fallback import fallback_analysis 
from .intent_classifier import IntentClassifier, normalize_text # <-- ADICIONADO normalize_text
from colorama import Fore, Style, init
init(autoreset=True)

import schemas 

AnalysisResponse = schemas.AnalysisResponse
Tip = schemas.Tip
ChartData = schemas.ChartData 

class AnalysisMemory:
    def __init__(self, max_history=5):
        self.history = deque(maxlen=max_history)

    def add(self, query, summary):
        self.history.append({"query": query, "summary": summary})

    def last(self):
        return list(self.history)

memory = AnalysisMemory()

import os, importlib, sys

# Caminho absoluto até a raiz do backend
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "intent_model.joblib")

try:
    sys.modules['intent_classifier'] = importlib.import_module('services.intent_classifier')
    local_intent_classifier = joblib.load(MODEL_PATH)
    print(f"Modelo de Intenção Local carregado de: {MODEL_PATH}")
except FileNotFoundError:
    print(f"⚠️ Modelo não encontrado em {MODEL_PATH}. Usando classificador vazio.")
    local_intent_classifier = IntentClassifier()
except Exception as e:
    print(f"Erro ao carregar modelo local: {e}. Usando classificador vazio.")
    local_intent_classifier = IntentClassifier()

async def detect_intents(query: str) -> List[str]:
    """
    Identifica todas as intenções ativas na consulta usando o Gemini LLM.
    Retorna uma lista estruturada: ["qualidade", "causa_raiz", "individual", ...]
    """
    try:
        intents = await classify_query_intent(query)
        print(Fore.CYAN + f"[Gemini] Intenção detectada via LLM: {intents}" + Style.RESET_ALL)
        return intents
    except Exception as e:
        print(Fore.YELLOW + f"[Fallback Local] Erro no LLM ({e}). Usando modelo local..." + Style.RESET_ALL)
        
        # Fallback de Intenção: Classifica localmente se o LLM falhar
        local_intent = local_intent_classifier.predict(query)
        print(Fore.GREEN + f"[Local Model] Intenção detectada: {local_intent}" + Style.RESET_ALL)
        
        if local_intent != "general":
             return [local_intent] 
        return ["default"]

# --- FUNÇÕES DE SUPORTE PARA ANÁLISE SETORIAL (NOVAS) ---
SECTOR_ALIASES = {
    'protecao 1': 'Proteção 1',
    'protecao 2': 'Proteção 2',
    'sylmara': 'Revisão - Sylmara',
    'cryslainy': 'Revisão - Cryslainy',
    'venancio': 'Revisão - Venâncio',
    'evilla': 'Revisão - Evilla',
    'evelin': 'Revisão - Evelin',
    'revisao - outros': 'Revisão - Outros',
    'smt': 'SMT', 
    'pth': 'PTH', 
    'tempo': 'Tempo', 
    'nivel': 'Nível', 
    'assistencia': 'Assistência',
    'revisao': 'Revisão', # Deve ser o último item de Revisão para não sobrepor os nomes específicos
}

def _find_sector_in_query(query: str) -> Optional[str]:
    """Tenta extrair um nome de setor real da query usando aliases predefinidos."""
    # Usando a função de normalização importada
    normalized_query = normalize_text(query) 
    
    # Busca por correspondência exata ou parcial, priorizando os nomes longos/compostos
    for key, full_name in SECTOR_ALIASES.items():
        if key in normalized_query:
            return full_name
            
    return None

async def run_sector_specific_analysis(df: pd.DataFrame, query: str) -> Dict[str, Any]:
    """
    Executa análises múltiplas e focadas em um setor específico identificado na query.
    """
    
    # 1. Identificar o Setor (Executado fora do threadpool principal para eficiência)
    # ATENÇÃO: run_in_threadpool aqui está correto porque _find_sector_in_query é síncrona
    setor_nome = await run_in_threadpool(_find_sector_in_query, query) 

    if not setor_nome:
        return {'status': 'FAIL', 'summary': "Não foi possível identificar o setor específico (SMT, PTH, Revisão, Proteção 1, etc.) na sua pergunta. Por favor, especifique.", 'visualization_data': [], 'tips': []}

    # 2. Filtrar o DataFrame
    # Assumindo que a coluna do DF processado que contém o setor é 'setor_falha_individual'
    mask = df['setor_falha_individual'].str.contains(setor_nome, case=False, na=False) # Usando setor_falha_individual como coluna de detecção
    df_setor = df[mask].copy()

    if df_setor.empty:
        return {'status': 'OK', 'summary': f"✅ **Análise Setorial - {setor_nome}:** Sem registros de falha encontrados para este setor no período selecionado.", 'visualization_data': [], 'tips': []}

    # 3. Executar Múltiplas Análises em Paralelo (Focadas no Setor)
    
    # Executa as funções existentes, mas no DataFrame filtrado (df_setor)
    tasks_to_run = [
        asyncio.to_thread(run_quality_analysis, df_setor, f"qualidade do setor {setor_nome}"),
        asyncio.to_thread(run_root_cause_analysis, df_setor, f"causas do setor {setor_nome}"),
        asyncio.to_thread(run_individual_performance_analysis, df_setor, f"top operadores do setor {setor_nome}"),
    ]
    
    # Inclui NLP se houver dados de observação suficientes
    if len(df_setor) > 30 and 'observacao_combinada' in df_setor.columns and df_setor['observacao_combinada'].dropna().any():
        tasks_to_run.append(run_nlp_analysis(df_setor, f"análise de tópicos das observações do setor {setor_nome}"))

    executed_results = await asyncio.gather(*tasks_to_run, return_exceptions=True)

    # 4. Consolidar Resultados
    
    combined_summary = f"**Análise Setorial Detalhada: {setor_nome}** ({len(df_setor)} registros)\n\n"
    combined_vis = []
    combined_tips = []
    llm_raw_analysis_data = {}
    
    for result in executed_results:
        # 1. VERIFICAÇÃO DE EXCEÇÃO
        if isinstance(result, Exception):
            print(f"Aviso: Uma das análises setoriais falhou com Exceção: {result}")
            continue
            
        # 2. VERIFICAÇÃO DE TIPO (CORREÇÃO CRÍTICA)
        if not isinstance(result, dict):
            # Captura o objeto 'coroutine' ou outro tipo inesperado
            print(f"Aviso: Resultado inesperado durante a consolidação (não é um dicionário): {type(result)}")
            continue
            
        # 3. FILTRO DE STATUS
        if result.get('status') in ('OK', 'INFO'): 
            combined_summary += result.get('summary', '') + "\n\n"
            combined_vis.extend(result.get('visualization_data', []))
            combined_tips.extend(result.get('tips', []))
            llm_raw_analysis_data.update(result.get('llm_raw_analysis', {}))

    return {
        'status': 'OK',
        'summary': combined_summary,
        'visualization_data': combined_vis,
        'tips': combined_tips,
        'llm_raw_analysis': llm_raw_analysis_data 
    }
# --- FIM DAS FUNÇÕES NOVAS ---


# --- FUNÇÕES DE ANÁLISE EXISTENTES (MANUTENÇÃO) ---

def _extract_person_or_machine_id(query: str) -> Optional[str]:
    match_id = re.search(r'(ID|Cod|Operador|Pessoa|Maquina)\s*[:]?\s*(\w+)', query, re.IGNORECASE)
    if match_id:
        return match_id.group(2).strip()
   
    match_quote = re.search(r"['\"](\w+)['\"]", query)
    if match_quote:
        return match_quote.group(1).strip()
        
    return None

def _simple_keyword_parser(query: str) -> str:
    query_lower = query.lower()
    
    if any(word in query_lower for word in ["solda", "smt", "estêncil", "pasta", "fluxo"]):
        return "SMT / Solda"
    if any(word in query_lower for word in ["produto", "placa", "componente", "item", "modelo"]):
        return "Produto / Componente"
    if any(word in query_lower for word in ["setor", "área", "origem", "detecção"]):
        return "Setor"
    if any(word in query_lower for word in ["desvio", "rejeição", "dppm", "taxa", "qualidade"]):
        return "Qualidade / Métrica"
    
    return "Foco Geral (Não Classificado)"

def forecast_next_period(df: pd.DataFrame) -> Optional[float]:
    df['periodo'] = df['data_registro'].dt.to_period('M').astype(str)
    df['quantidade'] = pd.to_numeric(df['quantidade'], errors='coerce').fillna(0)
    series = df.groupby('periodo')['quantidade'].sum().values
    
    if len(series) > 3:
        try:
            trend = np.polyfit(range(len(series)), series, 1)[0]
            next_val = series[-1] + trend
            return float(max(0, next_val)) 
        except Exception:
            return None
    return None

def _extract_origin_sector(causa_raiz: str) -> str:
    if not isinstance(causa_raiz, str):
        return 'Setor Desconhecido'
    
    match = re.search(r'\((.*?)\)', causa_raiz)
    if match:
        origin_str = match.group(1)
        return origin_str.split('/')[0].strip()
    return 'Geral/Outros'

def run_quality_analysis(df: pd.DataFrame, query: str) -> Dict[str, Any]:
    period, specific_date, granularity_name = extract_period_and_date(query)
    df_filtered = df.copy()

    if specific_date:
        if period == 'D' or ('day' in str(specific_date) and period == 'G'): 
            df_filtered = df_filtered[df_filtered['data_registro'].dt.date == specific_date.date()]
            granularity_name = "Diária"
            period = 'D'
        elif period == 'M' or ('day' not in str(specific_date)):
            df_filtered = df_filtered[
                (df_filtered['data_registro'].dt.year == specific_date.year) & 
                (df_filtered['data_registro'].dt.month == specific_date.month)
            ]
            granularity_name = f"Mensal ({specific_date.strftime('%m/%Y')})"
            period = 'D' 

    if df_filtered.empty:
        return {'status': 'FAIL', 'summary': f"Não há dados para o período solicitado: **{granularity_name}**.", 'visualization_data': [], 'tips': []}

    if period == 'G':
        period = 'M' 
        granularity_name = 'Mensal'

    df_filtered = df_filtered.reset_index(drop=True) 
    df_filtered['periodo'] = df_filtered['data_registro'].dt.to_period(period).astype(str)

    df_filtered['quantidade'] = pd.to_numeric(df_filtered['quantidade'], errors='coerce').fillna(0)
    df_filtered['quantidade_produzida'] = pd.to_numeric(df_filtered['quantidade_produzida'], errors='coerce').fillna(0)
    
    summary_data = df_filtered.groupby('periodo').agg(
        total_falhas_periodo=('quantidade', 'sum'),
        total_producao_periodo=('quantidade_produzida', 'sum') 
    ).reset_index()

    summary_data['total_producao_safe'] = summary_data['total_producao_periodo'].apply(lambda x: x if x > 0 else 1)
    summary_data['rejeicao_percentual'] = (summary_data['total_falhas_periodo'] / summary_data['total_producao_safe']) * 100

    media_rejeicao = summary_data['rejeicao_percentual'].mean()
    top_period_rejeicao = summary_data.sort_values(by='rejeicao_percentual', ascending=False).iloc[0] if not summary_data.empty else {'periodo': 'N/A', 'rejeicao_percentual': 0}

    df_pico = df_filtered[df_filtered['periodo'] == top_period_rejeicao['periodo']]
    top_falha_pico = df_pico['falha_individual'].mode().iat[0] if not df_pico.empty and not df_pico['falha_individual'].mode().empty else "N/A"
    
    resumo = f"""
        **Análise de Qualidade: Taxa de Rejeição e Tendência ({granularity_name})**
        
        A **média de Rejeição** no período analisado é de **{media_rejeicao:.2f}%**.
        O período com a **maior taxa de rejeição** foi **{top_period_rejeicao['periodo']}**, com **{top_period_rejeicao['rejeicao_percentual']:.2f}%**.
        
        **Foco:** A principal falha neste período de pico foi: **{top_falha_pico}**.
    """
    
    vis_data = ChartData(
        title=f"Tendência da Taxa de Rejeição (%) - Agregação {granularity_name}",
        labels=summary_data['periodo'].tolist(),
        datasets=[
            {"label": "Taxa de Rejeição (%)", "data": summary_data['rejeicao_percentual'].tolist(), "type": 'line', "borderColor": 'rgb(255, 99, 132)', "backgroundColor": 'rgba(255, 99, 132, 0.5)'}
        ],
        chart_type='line' 
    )
    
    dicas = [
        Tip(title="Foco no Desvio", detail=f"O processo de controle de qualidade deve analisar o período de pico ({top_period_rejeicao['periodo']}) e a falha **{top_falha_pico}**."),
        Tip(title="Meta Estratégica", detail=f"Busque reduzir a taxa média geral para **{(media_rejeicao * 0.9):.2f}%** no próximo ciclo."),
    ]

    del df_filtered
    gc.collect()

    return {
        'status': 'OK', 
        'summary': resumo, 
        'visualization_data': [vis_data.model_dump()], 
        'tips': dicas,
        'llm_raw_analysis': {
            'summary': resumo,
            'topics_data': [{'nome': 'Rejeição Alta', 'contagem': 1}]
        }
    }

def run_root_cause_analysis(df: pd.DataFrame, query: str) -> Dict[str, Any]:
    item_col = 'produto' if 'produto' in df.columns else ('produto_id' if 'produto_id' in df.columns else 'falha_individual') 
    
    if any(word in query.lower() for word in ["placa", "produto", "componente", "item", "modelo"]):

        item_counts = df[item_col].value_counts(normalize=True).mul(100).round(2).reset_index(name='percentual').head(5)
        item_counts.columns = [item_col, 'percentual']
        
        if not item_counts.empty:
            top_item = item_counts.iloc[0][item_col]
            top_item_perc = item_counts.iloc[0]['percentual']
            
            resumo_item = f"""
                **Análise de Prioridade de Produto (Causa Raiz)**
                O produto/item com maior incidência de falhas é **{top_item}**, representando **{top_item_perc:.2f}%** do total.
                Recomenda-se uma investigação aprofundada neste item específico.
            """
            
            vis_data_item = ChartData(
                title=f"Distribuição Top 5 de Falhas por Produto/Item ({item_col})",
                labels=item_counts[item_col].tolist(),
                datasets=[
                    {"label": "Percentual de Falhas", "data": item_counts['percentual'].tolist(), "type": 'pie', "backgroundColor": ['#00b37c', '#24a4ff', '#ffcd56', '#9966ff', '#ff9f40']}
                ],
                chart_type='pie'
            )
            
            dicas_item = [
                Tip(title="Foco de Engenharia", detail=f"Concentre a investigação no design e montagem do item **{top_item}**."),
                Tip(title="Auditoria de Processo", detail=f"Verifique o processo produtivo que leva à falha deste item."),
            ]

            return {
                'status': 'OK', 
                'summary': resumo_item, 
                'visualization_data': [vis_data_item.model_dump()], 
                'tips': dicas_item,
                'llm_raw_analysis': {
                    'summary': resumo_item,
                    'topics_data': [{'nome': f'Falha do Produto {top_item}', 'contagem': df[item_col].value_counts().iloc[0]}]
                }
            }

    causa_col_to_use = 'causa_raiz_detalhada' if 'causa_raiz_detalhada' in df.columns else 'causa_raiz_processo'

    causa_raiz_counts = df[causa_col_to_use].value_counts(normalize=True).mul(100).round(2).reset_index(name='percentual').head(5)
    causa_raiz_counts.columns = ['causa_raiz', 'percentual']
    
    top_causa = causa_raiz_counts.iloc[0]['causa_raiz'] if not causa_raiz_counts.empty else "N/A"
    top_causa_perc = causa_raiz_counts.iloc[0]['percentual'] if not causa_raiz_counts.empty else 0.0

    linha_counts = df['linha_produto'].value_counts(normalize=True).mul(100).round(2).reset_index(name='percentual').head(5)
    linha_counts.columns = ['linha_produto', 'percentual']
    top_linha = linha_counts.iloc[0]['linha_produto'] if not linha_counts.empty else "N/A"
    top_linha_perc = linha_counts.iloc[0]['percentual'] if not linha_counts.empty else 0.0

    resumo = f"""
        **Análise de Prioridade (Conhecimento de Processo)**
        A principal causa de falhas é **{top_causa}**, representando **{top_causa_perc}%** do total.
        A linha de produtos com maior incidência de falhas é a **Linha {top_linha}** (**{top_linha_perc}%** das ocorrências).
    """
    
    vis_data_causa = ChartData(
        title=f"Distribuição Top 5 de Falhas por Causa Raiz ({causa_col_to_use.replace('_', ' ').title()})",
        labels=causa_raiz_counts['causa_raiz'].tolist(),
        datasets=[
            {"label": "Percentual de Falhas", "data": causa_raiz_counts['percentual'].tolist(), "type": 'bar', "backgroundColor": 'rgba(75, 192, 192, 0.7)'}
        ],
        chart_type='bar'
    )
    
    vis_data_linha = ChartData(
        title="Distribuição Top 5 de Falhas por Linha de Produto",
        labels=linha_counts['linha_produto'].tolist(),
        datasets=[
            {"label": "Percentual de Falhas", "data": linha_counts['percentual'].tolist(), "type": 'bar', "backgroundColor": 'rgba(255, 159, 64, 0.7)'}
        ],
        chart_type='bar'
    )
    
    dicas = [
        Tip(title=f"Ação Prioritária (Causa)", detail=f"Concentre esforços na causa **'{top_causa}'**."),
        Tip(title=f"Ação Prioritária (Produto)", detail="Realize auditorias nos procedimentos de montagem e teste dos produtos da Linha de " + top_linha + "."),
    ]
    
    return {
        'status': 'OK', 
        'summary': resumo, 
        'visualization_data': [vis_data_causa.model_dump(), vis_data_linha.model_dump()], 
        'tips': dicas,
        'llm_raw_analysis': {
            'summary': resumo,
            'topics_data': [{'nome': top_causa, 'contagem': causa_raiz_counts.iloc[0]['percentual']}]
        }
    }

def run_individual_performance_analysis(df: pd.DataFrame, query: str) -> Dict[str, Any]:
    """
    Analisa o desempenho individual de revisores ou operadores.
    Se o texto da query indicar 'revisão' ou 'revisores', mostra somente as pessoas da revisão.
    """
    print(Fore.CYAN + "📊 Iniciando análise individual..." + Style.RESET_ALL)

    # 🔹 Detectar se a consulta é sobre revisão
    query_lower = query.lower()
    is_revisao_query = any(w in query_lower for w in ["revisão", "revisores", "pessoas da revisão"])

    # 🔹 Identificar coluna base
    possible_cols = ['pessoa_id', 'maquina_id', 'responsavel_falha', 'setor_falha_individual', 'linha_produto']
    col_name = next((col for col in possible_cols if col in df.columns), None)

    if not col_name:
        return {
            'status': 'FAIL',
            'summary': "Análise Individual não executada: coluna de identificação não encontrada.",
            'visualization_data': [],
            'tips': []
        }

    df_filtered = df.copy()

    # 🔹 Filtro especial: se for revisão, mantém apenas revisores definidos
    revisores_validos = [
        'Revisão - Sylmara', 'Revisão - Cryslainy', 'Revisão - Venâncio',
        'Revisão - Evilla', 'Revisão - Evelin', 'Revisão - Outros'
    ]
    if is_revisao_query and 'setor_falha_individual' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['setor_falha_individual'].isin(revisores_validos)]
        col_name = 'setor_falha_individual'  # força o agrupamento pelos revisores
        print(Fore.GREEN + f"🎯 Consulta de revisão detectada — filtrando apenas revisores: {revisores_validos}" + Style.RESET_ALL)

    if df_filtered.empty:
        return {
            'status': 'FAIL',
            'summary': "Nenhum dado encontrado para revisão ou colaboradores.",
            'visualization_data': [],
            'tips': []
        }

    # 🔹 Normaliza e agrupa
    df_filtered['quantidade'] = pd.to_numeric(df_filtered['quantidade'], errors='coerce').fillna(0)
    perf_counts = df_filtered.groupby(col_name)['quantidade'].sum().sort_values(ascending=False).reset_index(name='Total_Falhas')

    if perf_counts.empty:
        return {'status': 'FAIL', 'summary': "Dados insuficientes para cálculo.", 'visualization_data': [], 'tips': []}

    vis_data = ChartData(
        title=f"Falhas por {col_name.replace('_', ' ').title()} da Revisão" if is_revisao_query else f"Falhas por {col_name.title()}",
        labels=perf_counts[col_name].tolist(),
        datasets=[{
            "label": "Total de Falhas",
            "data": perf_counts['Total_Falhas'].tolist(),
            "type": 'bar',
            "backgroundColor": 'rgba(255, 99, 132, 0.7)'
        }],
        chart_type='bar'
    )

    resumo = (
        f"**Análise de Performance Individual**\n\n"
        f"{'Foco: Revisores da Revisão' if is_revisao_query else 'Foco: Operadores/Setores'}.\n"
        f"O total de falhas foi somado por {col_name.replace('_', ' ')}. "
        f"O principal destaque é **{perf_counts.iloc[0][col_name]}** com **{int(perf_counts.iloc[0]['Total_Falhas'])}** falhas registradas."
    )

    dicas = [
        Tip(title="Atenção ao Top Falhador", detail=f"Verifique o processo do revisor {perf_counts.iloc[0][col_name]}."),
        Tip(title="Comparativo", detail="Analise o desempenho dos demais revisores como referência.")
    ]

    return {
        'status': 'OK',
        'summary': resumo,
        'visualization_data': [vis_data.model_dump()],
        'tips': dicas,
        'llm_raw_analysis': {
            'summary': resumo,
            'topics_data': [
                {'nome': perf_counts.iloc[i][col_name], 'contagem': int(perf_counts.iloc[i]['Total_Falhas'])}
                for i in range(len(perf_counts))
            ]
        }
    }


def run_smt_trend_analysis(df: pd.DataFrame, query: str) -> Dict[str, Any]:
    if 'causa_raiz_detalhada' not in df.columns:
        return {'status': 'FAIL', 'summary': "Análise SMT não executada: Coluna 'causa_raiz_detalhada' não encontrada.", 'visualization_data': [], 'tips': []}

    df_smt = df[df['causa_raiz_detalhada'].str.contains('SMT|Solda|Stencil|Pasta|Fluxo', case=False, na=False)].copy()

    if df_smt.empty:
        return {'status': 'INFO', 'summary': "Nenhuma falha de solda ou SMT encontrada no dataset atual para análise detalhada.", 'visualization_data': [], 'tips': []}

    df_smt['periodo'] = df_smt['data_registro'].dt.to_period('M').astype(str)

    df_smt['quantidade'] = pd.to_numeric(df_smt['quantidade'], errors='coerce').fillna(0)
    
    trend_data = df_smt.groupby('periodo')['quantidade'].sum().reset_index(name='Total_Falhas_SMT')

    if trend_data.empty:
          return {'status': 'INFO', 'summary': "Dados de tendência SMT insuficientes.", 'visualization_data': [], 'tips': []}

    top_falhas_smt = df_smt['falha_individual'].value_counts().head(5).reset_index(name='Contagem')
    top_falhas_smt.columns = ['Falha_Individual', 'Contagem']
    
    top_falha_nome = top_falhas_smt.iloc[0]['Falha_Individual'] if not top_falhas_smt.empty else "N/A"
    total_falhas_smt = trend_data['Total_Falhas_SMT'].sum()

    resumo = f"""
        **Análise Focada: Risco e Tendência de Falhas de Solda/SMT**
        
        No período, foram registradas **{int(total_falhas_smt)} falhas** relacionadas diretamente a processos SMT ou Solda.
        
        **Tendência:** A tendência de falhas por mês é mostrada no gráfico de linha abaixo. Uma investigação é necessária se a tendência for crescente.
        
        **Principal Causa Tática:** A falha mais comum neste grupo é **'{top_falha_nome}'**. 
    """

    vis_tendencia = ChartData(
        title="1. Tendência Mensal de Falhas SMT/Solda (Contagem)",
        labels=trend_data['periodo'].tolist(),
        datasets=[
            {"label": "Total de Falhas SMT", "data": trend_data['Total_Falhas_SMT'].tolist(), "type": 'line', "borderColor": 'rgb(255, 165, 0)', "backgroundColor": 'rgba(255, 165, 0, 0.5)'}
        ],
        chart_type='line' 
    )

    vis_top_falhas = ChartData(
        title="2. Top 5 Falhas Individuais dentro do Processo SMT",
        labels=top_falhas_smt['Falha_Individual'].tolist(),
        datasets=[
            {"label": "Contagem", "data": top_falhas_smt['Contagem'].tolist(), "type": 'bar', "backgroundColor": 'rgba(0, 150, 255, 0.7)'}
        ],
        chart_type='bar' 
    )

    dicas = [
        Tip(title="Foco de Processo SMT", detail=f"O time de SMT deve auditar imediatamente o processo e insumos relacionados à falha **'{top_falha_nome}'**."),
        Tip(title="Monitoramento", detail="Use o gráfico de tendência para determinar se as ações corretivas recentes estão surtindo efeito."),
    ]
    
    del df_smt
    gc.collect()
    
    return {
        'status': 'OK', 
        'summary': resumo, 
        'visualization_data': [vis_tendencia.model_dump(), vis_top_falhas.model_dump()], 
        'tips': dicas,
        'llm_raw_analysis': {
            'summary': resumo,
            'topics_data': [{'nome': f'Foco SMT: {top_falha_nome}', 'contagem': top_falhas_smt.iloc[0]['Contagem']}]
        }
    }

async def run_nlp_analysis(df: pd.DataFrame, query: str) -> Dict[str, Any]:
    if df['observacao_combinada'].isnull().all() or df['observacao_combinada'].str.strip().eq('').all():
        return {'status': 'FAIL', 'summary': "Análise de Tópicos não executada: A maioria das observações está vazia ou nula.", 'visualization_data': [], 'tips': []}

    analysis_result = {'status': 'FAIL', 'error': 'Inicializado'} # Resultado default em caso de erro

    try:
        # Adiciona um TEMPO LIMITE de 25 segundos para a chamada ao LLM
        analysis_result = await asyncio.wait_for(
            analyze_observations_with_gemini(df, query),
            timeout=25.0
        )
    except asyncio.TimeoutError:
        print("⚠️ Tempo limite (25s) excedido na análise de NLP. Ativando fallback.")
        analysis_result = {'status': 'FAIL', 'error': 'Timeout de 25s no LLM'}
    except Exception as e:
        print(f"Erro na API do LLM: {e}")
        analysis_result = {'status': 'FAIL', 'error': str(e)}

    
    if analysis_result['status'] != 'OK':
        # --- CAMADA DE FALLBACK INTELIGENTE (Será ativada em caso de Timeout) ---
        
        keyword_focus = _simple_keyword_parser(query)
        error_detail = analysis_result.get('error', 'Erro desconhecido de API/Parsing')
        
        # Executa uma análise estatística simples no foco identificado
        falha_col = 'falha_individual' if 'falha_individual' in df.columns else 'falha'
        top_falha = df[falha_col].dropna().mode().iat[0] if not df[falha_col].dropna().empty else "N/A"
        
        fallback_summary = f"""
        **Análise de Tópicos (Modo de Resiliência Ativado)**
        
        A análise avançada da IA falhou devido a um erro de comunicação ou parsing ({error_detail}).
        
        **Foco Detectado:** Sua pergunta sugere foco em **{keyword_focus}**.
        
        Como alternativa, o dado mais relevante no momento é a falha **'{top_falha}'**, que domina a incidência no conjunto de dados.
        
        Por favor, ajuste a granularidade da pergunta (ex: "apenas as observações de hoje") e tente novamente.
        """

        return {
            'status': 'INFO', # Muda para INFO para que o resultado seja agregado no composite
            'summary': fallback_summary, 
            'visualization_data': [],
            'tips': [Tip(title="Resiliência da IA", detail=f"O LLM falhou, mas o sistema sugeriu foco em: **{keyword_focus}** e na falha **{top_falha}**.")],
            'llm_raw_analysis': {'summary': fallback_summary, 'topics_data': []}
        }

    topicos = analysis_result['topics_data']
    
    if not topicos:
        return {
            'status': 'FAIL', 
            'summary': "O Gemini não retornou tópicos válidos para visualização.", 
            'visualization_data': [], 
            'tips': []
        }

    vis_data = ChartData(
        title="Contagem por Causa Raiz (Análise IA do Texto via Gemini)",
        labels=[t['nome'] for t in topicos],
        datasets=[
            {"data": [t['contagem'] for t in topicos], "type": 'pie', "backgroundColor": ["#4bc0c0", "#ff6384", "#ffcd56", "#36a2eb", "#9966ff"]}
        ],
        chart_type='pie' 
    )
    
    top_causa_ia = topicos[0]['nome']
    
    resumo = f"**Análise de Tópicos via Gemini:**\n{analysis_result['summary']}"
    
    dicas = [
        Tip(title="Ação por Tópico", detail=f"Se a principal causa é '{top_causa_ia}', revise as instruções ou materiais para a prevenção."),
        Tip(title="Validação", detail="O Gemini validou a classificação. Use esta informação para refinar processos.")
    ]

    return {
        'status': 'OK', 
        'summary': resumo, 
        'visualization_data': [vis_data.model_dump()], 
        'tips': dicas,
        'llm_raw_analysis': analysis_result
    }

def run_sector_analysis(df: pd.DataFrame, query: str) -> Dict[str, Any]:
    """
    Executa análise de falhas por setor, mostrando gráfico e tabela completa de todos os setores.
    """
    print(Fore.MAGENTA + "📊 Iniciando análise de falhas por setor..." + Style.RESET_ALL)

    # Colunas possíveis que indicam setor
    sector_cols = ["setor_falha_individual", "linha_produto", "setor", "departamento"]
    col_sector = next((col for col in sector_cols if col in df.columns), None)

    if not col_sector:
        return {
            'status': 'FAIL',
            'summary': "Coluna de setor não encontrada para análise.",
            'visualization_data': [],
            'tips': []
        }

    # Converte quantidade e agrupa por setor
    df['quantidade'] = pd.to_numeric(df.get('quantidade', 1), errors='coerce').fillna(0)
    grouped = df.groupby(col_sector)['quantidade'].sum().sort_values(ascending=False).reset_index()

    if grouped.empty:
        return {
            'status': 'FAIL',
            'summary': "Nenhum dado disponível para análise por setor.",
            'visualization_data': [],
            'tips': []
        }

    # --- VISUALIZAÇÃO: Gráfico de Barras ---
    chart = ChartData(
        title="Distribuição de Falhas por Setor",
        labels=grouped[col_sector].tolist(),
        datasets=[{
            "label": "Total de Falhas",
            "data": grouped['quantidade'].tolist(),
            "type": 'bar',
            "backgroundColor": 'rgba(54, 162, 235, 0.7)',
        }],
        chart_type='bar'
    )

    # --- VISUALIZAÇÃO: Tabela com todos os setores ---
    table = ChartData(
        title="Tabela de Falhas por Setor",
        chart_type='table',
        labels=["Setor", "Total de Falhas"],
        datasets=[{
            "label": "Falhas",
            "data": [
                [row[col_sector], int(row['quantidade'])] for _, row in grouped.iterrows()
            ]
        }]
    )

    # --- RESUMO ---
    top_sector = grouped.iloc[0][col_sector]
    top_value = int(grouped.iloc[0]['quantidade'])
    total_failures = int(grouped['quantidade'].sum())
    resumo = (
        f"**Análise de Falhas por Setor**\n\n"
        f"O setor com maior incidência de falhas é **{top_sector}**, "
        f"com **{top_value}** registros, representando aproximadamente "
        f"{(top_value / total_failures * 100):.1f}% do total de {total_failures} falhas."
    )

    dicas = [
        Tip(title="Foco de Melhoria", detail=f"Investigue as causas no setor **{top_sector}**."),
        Tip(title="Análise Completa", detail="Considere os setores de menor falha como referência de boas práticas."),
    ]

    return {
        'status': 'OK',
        'summary': resumo,
        'visualization_data': [chart.model_dump(), table.model_dump()],
        'tips': dicas,
        'llm_raw_analysis': {
            'summary': resumo,
            'topics_data': [
                {'nome': row[col_sector], 'contagem': int(row['quantidade'])}
                for _, row in grouped.iterrows()
            ]
        }
    }


def run_structured_default_analysis(df: pd.DataFrame, query: str) -> Dict[str, Any]:
    if df.empty:
        return {
            'status': 'FAIL', 
            'summary': "Não há dados válidos para realizar a Análise Estruturada Padrão.", 
            'visualization_data': [], 
            'tips': []
        }
    
    falha_col = 'falha_individual' if 'falha_individual' in df.columns else 'falha'

    causa_col_to_use = 'causa_raiz_detalhada' if 'causa_raiz_detalhada' in df.columns else 'causa_raiz_processo'

    causa_counts = df[causa_col_to_use].value_counts().head(3)
    top_causa = causa_counts.index[0] if not causa_counts.empty else "N/A"

    linha_counts = df['linha_produto'].value_counts().head(3)
    top_linha = linha_counts.index[0] if not linha_counts.empty else "N/A"
    
    top_falha = df[falha_col].dropna().mode().iat[0] if not df[falha_col].dropna().empty else "N/A"

    summary = f"""
        **Análise Estruturada Padrão (Fallback Robusto)**
        A IA compilou as principais prioridades de foco com base nos dados brutos ({len(df)} registros):
        
        1.  **Prioridade de Processo (Causa Raiz):** A causa mais comum é **'{top_causa}'**.
        2.  **Prioridade de Produção (Linha):** A linha de produto **'{top_linha}'** tem a maior incidência de falhas.
        3.  **Falha de Componente:** A falha mais registrada é **'{top_falha}'**.
    """
    
    vis_data = []

    if not causa_counts.empty:
        vis_data.append(ChartData(
            title="Top 3 Causas Raiz de Processo",
            labels=causa_counts.index.tolist(),
            datasets=[{"data": causa_counts.tolist(), "type": 'pie', "backgroundColor": ["#4bc0c0", "#ff6384", "#ffcd56"]}],
            chart_type='pie'
        ).model_dump())

    if not linha_counts.empty:
        vis_data.append(ChartData(
            title="Top 3 Linhas de Produto com Falha",
            labels=linha_counts.index.tolist(),
            datasets=[{"data": linha_counts.tolist(), "type": 'pie', "backgroundColor": ["#36a2eb", "#9966ff", "#ff9f40"]}],
            chart_type='pie'
        ).model_dump())

    dicas = [
        Tip(title="Foco Imediato", detail=f"Concentre a investigação na causa **'{top_causa}'**."),
        Tip(title="Sugestão de Busca", detail="Para detalhes, pergunte: 'Qual a tendência de rejeição mensal?' ou 'Análise do tópico das observações.'")
    ]
    
    return {
        'status': 'OK', 
        'summary': summary, 
        'visualization_data': vis_data, 
        'tips': dicas,
        'llm_raw_analysis': {
            'summary': summary,
            'topics_data': [{'nome': 'Análise Estruturada Padrão', 'contagem': len(df)}] 
        }
    }

def run_dppm_definition() -> Dict[str, Any]:
    summary = """
        **DPPM** significa **Defeitos Por Milhão**.
        
        É uma métrica de qualidade que mede a quantidade de peças defeituosas ou falhas encontradas a cada 1 milhão de unidades produzidas.
        
        **Por que é importante?**
        - **Padronização:** Permite comparar a qualidade de diferentes processos ou fábricas.
        - **Alta Precisão:** É ideal para processos de alta qualidade, onde a taxa de defeitos é muito baixa (ex: 0,05%).
    """
    
    dicas = [
        Tip(title="Cálculo Rápido", detail="DPPM = (Total de Defeitos / Total Produzido) * 1.000.000"),
        Tip(title="Meta 6 Sigma", detail="O objetivo final da metodologia 6 Sigma é atingir um DPPM de apenas 3,4."),
    ]
    
    return {
        'status': 'OK',
        'summary': summary,
        'visualization_data': [],
        'tips': dicas,
        'llm_raw_analysis': {'summary': summary, 'topics_data': []}
    }

def run_greeting_analysis(query: str) -> Dict[str, Any]:
    current_hour = datetime.now().hour
    if 5 <= current_hour < 12:
        greeting = "Bom dia!"
    elif 12 <= current_hour < 18:
        greeting = "Boa tarde!"
    else:
        greeting = "Boa noite!"

    summary = (
        f"{greeting} Eu sou a **Cortex**, a inteligência artificial do Tron Analytics. Em que posso te ajudar hoje?"
        "\n\nPara começar, você pode me perguntar sobre:"
    )

    dicas = [
        Tip(title="Tendência de Qualidade", detail="Pergunte: 'Qual é a taxa de rejeição mensal da produção?'"),
        Tip(title="Foco Geográfico", detail="Pergunte: 'Quais são as falhas mais comuns no setor de SMT?'"),
        Tip(title="Desvio de Processo", detail="Pergunte: 'Análise do tópico das observações.'"),
    ]
    
    return {
        'status': 'OK', 
        'summary': summary, 
        'visualization_data': [], 
        'tips': dicas,
        'llm_raw_analysis': {'summary': summary, 'topics_data': []}
    }

async def run_domain_analysis_composite(query: str, data_to_analyze: List[Dict]) -> Dict[str, Any]:
    """
    Motor de Análise de Domínio Composta.
    Detecta intenções, executa análises estatísticas em paralelo e as consolida.
    """
    from colorama import Fore, Style
    query_lower = query.lower().strip()

    # --- TRATAMENTO DE SAUDAÇÃO ---
    greeting_patterns = r'^(oi|olá|ola|bom dia|boa tarde|boa noite|tudo bem|e aí)[\s.,!?]*$'
    if re.match(greeting_patterns, query_lower):
        return run_greeting_analysis(query)

    # --- TRATAMENTO DE DEFINIÇÕES ---
    definition_patterns = [
        'o que é dppm', 'dppm o que é', 'o que significa dppm',
        'definição de dppm', 'o que e dppm'
    ]
    if any(pattern in query_lower for pattern in definition_patterns):
        return run_dppm_definition()

    # --- PREPARAÇÃO DE DADOS ---
    df = prepare_dataframe(data_to_analyze, flatten_multifalha=True)
    if df.empty or len(data_to_analyze) == 0:
        return {
            "status": "FAIL",
            "summary": "Nenhum dado encontrado para análise ou dados inválidos. Verifique a seleção de dados.",
            "tips": [Tip(title="Base de Dados Vazia", detail="Verifique a fonte de dados e o filtro inicial.")],
            "visualization_data": [],
            "llm_raw_analysis": {'summary': '', 'topics_data': []}
        }

    # --- CONTINUIDADE DE CONTEXTO ---
    if "continuar" in query_lower or "agora me mostre" in query_lower:
        last_context = memory.last()
        if last_context:
            return {
                "status": "INFO",
                "summary": f"Continuando a partir da última análise ({last_context[-1]['query']}):\n\n{last_context[-1]['summary']}",
                "tips": [Tip(title="Contexto", detail="Reutilizei o resumo da análise anterior para dar continuidade à sua exploração.")],
                "visualization_data": [],
                "llm_raw_analysis": {'summary': '', 'topics_data': []}
            }

    # --- DETECÇÃO DE INTENÇÃO (LLM + LOCAL) ---
    active_intents = await detect_intents(query)

    # --- HEURÍSTICA ADICIONAL (revisores/pessoas) ---
    if any(word in query_lower for word in [
        "revisor", "revisores", "pessoa da revisão",
        "funcionário", "colaborador", "avaliador"
    ]):
        print("🧩 Heurística: Detecção de intenção 'individual' pela palavra-chave.")
        if "individual" not in active_intents:
            active_intents.append("individual")

    # --- NORMALIZAÇÃO DE INTENÇÕES ---
    def normalize_intents(intents):
        normalized = []
        for intent in intents:
            intent = normalize_text(intent).lower()
            if intent in ["setor", "sector", "linha", "producao", "produção", "setores"]:
                normalized.append("sector")
            elif intent in ["causa", "root_cause", "processo", "process", "causas"]:
                normalized.append("root_cause")
            elif intent in ["qualidade", "quality", "rejeição", "rejeicao"]:
                normalized.append("quality")
            elif intent in ["smt", "smt_foco", "solda", "stencil", "pasta", "fluxo"]:
                normalized.append("smt_foco")
            elif intent in [
                "individual", "pessoa", "operador", "colaborador", 
                "funcionário", "funcionarios", "revisor", "revisores", 
                "responsável", "responsaveis", "avaliador", "avaliadores"
            ]:
                normalized.append("individual")
        return list(set(normalized))

    active_intents = normalize_intents(active_intents)
    print(Fore.CYAN + f"🔍 Intenções normalizadas: {active_intents}" + Style.RESET_ALL)

    # --- DEFINIÇÃO DE TAREFAS ---
    tasks_to_run = []

    # --- ANÁLISE INDIVIDUAL (PRIORIDADE MÁXIMA) ---
    if "individual" in active_intents:
        print(Fore.GREEN + "🧩 Acionando análise INDIVIDUAL (Operadores/Revisores)" + Style.RESET_ALL)
        tasks_to_run.append(asyncio.to_thread(run_individual_performance_analysis, df, query))

    # --- ANÁLISE SETORIAL (PRIORIDADE SECUNDÁRIA) ---
    specific_sector_found = await run_in_threadpool(_find_sector_in_query, query)
    if not tasks_to_run and ("sector" in active_intents or specific_sector_found):
        print(Fore.MAGENTA + "🧩 Acionando análise de SETOR" + Style.RESET_ALL)
        if specific_sector_found:
            tasks_to_run.append(run_sector_specific_analysis(df, query))
        else:
            tasks_to_run.append(asyncio.to_thread(run_sector_analysis, df, query))

    # --- OUTRAS ANÁLISES (PARALELAS) ---
    if not tasks_to_run:
        if "quality" in active_intents:
            print(Fore.BLUE + "🧩 Acionando análise de QUALIDADE" + Style.RESET_ALL)
            tasks_to_run.append(asyncio.to_thread(run_quality_analysis, df, query))

        if "root_cause" in active_intents:
            print(Fore.YELLOW + "🧩 Acionando análise de CAUSA RAIZ" + Style.RESET_ALL)
            tasks_to_run.append(asyncio.to_thread(run_root_cause_analysis, df, query))

        if "smt_foco" in active_intents or any(w in query_lower for w in ["smt", "solda", "stencil", "pasta", "fluxo"]):
            print(Fore.CYAN + "🧩 Acionando análise SMT / Solda" + Style.RESET_ALL)
            tasks_to_run.append(asyncio.to_thread(run_smt_trend_analysis, df, query))

        if "nlp" in active_intents:
            print(Fore.LIGHTBLACK_EX + "🧩 Acionando análise NLP (texto livre)" + Style.RESET_ALL)
            tasks_to_run.append(run_nlp_analysis(df, query))

        # --- FALLBACK PADRÃO ---
        if not tasks_to_run or "default" in active_intents or "general" in active_intents:
            if "general" in active_intents:
                print(Fore.LIGHTYELLOW_EX + "⚙️ Ativando Fallback Inteligente (General Intent)" + Style.RESET_ALL)
                tasks_to_run.append(asyncio.to_thread(fallback_analysis, query, df, local_intent_classifier))
            else:
                print(Fore.LIGHTRED_EX + "⚙️ Ativando Fallback Estruturado Padrão" + Style.RESET_ALL)
                tasks_to_run.append(asyncio.to_thread(run_structured_default_analysis, df, query))

    # --- EXECUÇÃO DAS ANÁLISES ---
    executed_results = await asyncio.gather(*tasks_to_run, return_exceptions=True)
    successful_results = []
    for result in executed_results:
        if isinstance(result, Exception):
            print(f"⚠️ Erro durante a execução de uma tarefa de análise: {result}")
        elif result.get('status') in ('OK', 'INFO'):
            successful_results.append(result)

    # --- FALLBACK FINAL ---
    if not successful_results:
        print(Fore.RED + "🚨 Todas as análises falharam. Ativando Fallback Semântico Local Final." + Style.RESET_ALL)
        final_result = fallback_analysis(query, df, local_intent_classifier)
        memory.add(query, final_result.get('summary', ''))
        return final_result

    # --- CONSOLIDAÇÃO DE RESULTADOS ---
    llm_raw_analysis_data = next(
        (r['llm_raw_analysis'] for r in successful_results 
         if r.get('llm_raw_analysis') and r.get('llm_raw_analysis', {}).get('topics_data')), 
        None
    )

    combined_summary = "\n\n".join(r["summary"] for r in successful_results)
    combined_vis = [v for r in successful_results for v in r.get("visualization_data", [])]
    combined_tips = [t for r in successful_results for t in r.get("tips", [])]

    # --- PREVISÃO ---
    next_forecast = await run_in_threadpool(forecast_next_period, df)
    if next_forecast is not None and next_forecast > 0:
        forecast_summary = (
            f"\n\n📈 **Previsão de Risco:** Se o padrão se mantiver, "
            f"o próximo período pode registrar cerca de **{next_forecast:.0f} falhas**."
        )
        combined_summary += forecast_summary
        combined_tips.append(Tip(title="Ação Preventiva", detail="Avalie planos de mitigação para o próximo ciclo."))

    # --- INSIGHT ESTRATÉGICO (GEMINI) ---
    gemini_insight_text = ""
    if llm_raw_analysis_data:
        try:
            strategic_insight_result = await asyncio.wait_for(
                summarize_analysis_with_gemini(llm_raw_analysis_data),
                timeout=40.0
            )
            if strategic_insight_result['status'] == 'OK':
                gemini_insight_text = strategic_insight_result.get('strategic_insight', '')
        except asyncio.TimeoutError:
            print(Fore.YELLOW + "⚠️ Tempo limite excedido ao gerar Insight Estratégico." + Style.RESET_ALL)
        except Exception as e:
            print(Fore.RED + f"Erro ao gerar Insight Estratégico com Gemini: {e}" + Style.RESET_ALL)

    if gemini_insight_text:
        combined_tips.append(Tip(title="Insight Estratégico da IA", detail=gemini_insight_text))

    memory.add(query, combined_summary)

    del df
    gc.collect()

    return {
        "status": "OK",
        "query": query,
        "summary": combined_summary,
        "tips": combined_tips,
        "visualization_data": combined_vis
    }
