# services/analyst.py (CÓDIGO COMPLETO E CORRIGIDO - ESTRATÉGIA HÍBRIDA)

import json
import pandas as pd
import numpy as np
import re
from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncio 
from collections import deque

# Importações de módulos locais (Ajuste se o seu caminho for diferente)
import schemas 
from .preprocessing import prepare_dataframe, extract_period_and_date
# Assumindo que essas duas funções async estão no llm_core.py
from .llm_core import analyze_observations_with_gemini, summarize_analysis_with_gemini

AnalysisResponse = schemas.AnalysisResponse
Tip = schemas.Tip
ChartData = schemas.ChartData 

# --- MEMÓRIA DE CONTEXTO (Ideia 1) ---
class AnalysisMemory:
    def __init__(self, max_history=5):
        self.history = deque(maxlen=max_history)

    def add(self, query, summary):
        self.history.append({"query": query, "summary": summary})

    def last(self):
        return list(self.history)

memory = AnalysisMemory()

# --- FUNÇÃO AUXILIAR DE INTENÇÃO (Ideia 2) ---
intent_map = {
    "qualidade": ["dppm", "rejeição", "falha", "taxa de falha", "defeito", "qualidade", "tendência", "defeitos"],
    "setor": ["origem", "detecção", "departamento", "área", "setor"],
    "causa_raiz": ["causa raiz", "processo", "linha", "produto", "raiz"],
    "nlp": ["observação", "tópico", "comentário", "texto", "relato", "nlp"],
}

def detect_intents(query_lower: str) -> List[str]:
    """Identifica todas as intenções ativas na consulta."""
    active_intents = []
    for intent, words in intent_map.items():
        if any(word in query_lower for word in words):
            active_intents.append(intent)
    return active_intents if active_intents else ["default"]

# --- FUNÇÃO DE FORECAST SIMPLES (Ideia 5) ---
def forecast_next_period(df: pd.DataFrame) -> Optional[float]:
    """Previsão simples de falhas para o próximo período usando regressão linear."""
    df['periodo'] = df['data_registro'].dt.to_period('M').astype(str)
    series = df.groupby('periodo')['quantidade'].sum().values
    
    # Requer pelo menos 3 pontos de dados para uma regressão minimamente válida
    if len(series) > 3:
        try:
            # Encaixa uma linha (polinômio de grau 1) nos dados
            trend = np.polyfit(range(len(series)), series, 1)[0]
            next_val = series[-1] + trend
            return float(max(0, next_val)) # Garante que o forecast não seja negativo
        except Exception:
            return None
    return None

# --- FUNÇÕES EXISTENTES DE ANÁLISE (MANTIDAS) ---

def _extract_origin_sector(causa_raiz: str) -> str:
# ... (função _extract_origin_sector mantida)
    if not isinstance(causa_raiz, str):
        return 'Setor Desconhecido'
    
    match = re.search(r'\((.*?)\)', causa_raiz)
    if match:
        origin_str = match.group(1)
        return origin_str.split('/')[0].strip()
    return 'Geral/Outros'

def run_quality_analysis(df: pd.DataFrame, query: str) -> Dict[str, Any]:
# ... (função run_quality_analysis mantida)
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
        return {'status': 'FAIL', 'summary': f"Não há dados para o período solicitado: **{granularity_name}**.", 'visualization_data': []}

    if period == 'G':
        period = 'M' 
        granularity_name = 'Mensal'

    df_filtered = df_filtered.reset_index(drop=True) 
    df_filtered['periodo'] = df_filtered['data_registro'].dt.to_period(period).astype(str)
    
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
    
    return {'status': 'OK', 'summary': resumo, 'visualization_data': [vis_data.model_dump()], 'tips': dicas}

def run_root_cause_analysis(df: pd.DataFrame, query: str) -> Dict[str, Any]:
# ... (função run_root_cause_analysis mantida)
    causa_raiz_counts = df['causa_raiz_processo'].value_counts(normalize=True).mul(100).round(2).reset_index(name='percentual').head(5)
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
        title="Distribuição Top 5 de Falhas por Causa Raiz (Processo)",
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
    
    return {'status': 'OK', 'summary': resumo, 'visualization_data': [vis_data_causa.model_dump(), vis_data_linha.model_dump()], 'tips': dicas}

async def run_nlp_analysis(df: pd.DataFrame, query: str) -> Dict[str, Any]:
# ... (função run_nlp_analysis mantida)
    
    if df['observacao_combinada'].isnull().all() or df['observacao_combinada'].str.strip().eq('').all():
        return {'status': 'FAIL', 'summary': "Análise de Tópicos não executada: A maioria das observações está vazia ou nula.", 'visualization_data': []}

    analysis_result = await analyze_observations_with_gemini(df, query)
    
    if analysis_result['status'] != 'OK':
        return {
            'status': 'FAIL', 
            'summary': f"Falha na análise de tópicos com Gemini: {analysis_result.get('error', 'Erro desconhecido')}", 
            'visualization_data': [],
            'tips': [Tip(title="Erro de API/Contexto", detail="Verifique a chave de API ou se os dados de observação são relevantes.")],
        }

    topicos = analysis_result['topics_data']
    
    if not topicos:
        return {
            'status': 'FAIL', 
            'summary': "O Gemini não retornou tópicos válidos para visualização.", 
            'visualization_data': [], 
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
    
    return {'status': 'OK', 'summary': resumo, 'visualization_data': [vis_data.model_dump()], 'tips': dicas}

def run_sector_analysis(df: pd.DataFrame, query: str) -> Dict[str, Any]:
# ... (função run_sector_analysis mantida)
    
    if df.empty:
        return {'status': 'FAIL', 'summary': "Análise de Setor não executada: DataFrame vazio.", 'visualization_data': []}

    falhas_por_setor_deteccao = df.groupby('setor_falha_individual')['falha_individual'] \
                                 .count() \
                                 .sort_values(ascending=False) \
                                 .reset_index(name='Contagem').head(5)
    
    if falhas_por_setor_deteccao.empty:
        return {'status': 'FAIL', 'summary': "Análise de Setor não executada: Dados de setor de falha ausentes.", 'visualization_data': []}
             
    top_setor_deteccao = falhas_por_setor_deteccao.iloc[0]['setor_falha_individual']

    chart_deteccao = ChartData(
        title="1. Volume de Falhas por Setor de DETECÇÃO (Top 5)",
        labels=falhas_por_setor_deteccao['setor_falha_individual'].tolist(),
        datasets=[
            {"label": "Total de Falhas", "data": falhas_por_setor_deteccao['Contagem'].tolist(), "backgroundColor": "rgba(255, 99, 132, 0.7)" }
        ],
        chart_type='bar'
    )

    df['setor_origem'] = df['causa_raiz_processo'].apply(_extract_origin_sector)
    falhas_por_setor_origem = df.groupby('setor_origem')['documento_id'].count().sort_values(ascending=False).reset_index(name='Contagem').head(5)
    
    top_origem = falhas_por_setor_origem.iloc[0]['setor_origem'] if not falhas_por_setor_origem.empty else "N/A"

    chart_origem = ChartData(
        title="2. Volume de Falhas por Setor de ORIGEM (Top 5)",
        labels=falhas_por_setor_origem['setor_origem'].tolist(),
        datasets=[
            {"label": "Total de Falhas", "data": falhas_por_setor_origem['Contagem'].tolist(), "backgroundColor": "rgba(54, 162, 235, 0.7)" }
        ],
        chart_type='bar'
    )
    
    falhas_detalhadas = ""
    if top_origem != "N/A":
        falhas_do_top_origem = df[df['setor_origem'] == top_origem]
        top_falhas_no_origem = falhas_do_top_origem['falha_individual'].value_counts().head(3).reset_index()
        top_falhas_no_origem.columns = ['falha', 'contagem']

        if not top_falhas_no_origem.empty:
            falhas_detalhadas = "As falhas mais comuns neste setor de origem são:\n"
            for _, row in top_falhas_no_origem.iterrows():
                falhas_detalhadas += f"- **{row['falha']}** ({row['contagem']} ocorrências)\n"
    
    resumo = f"""
        **Análise de Setor (Detecção vs. Origem)**
        
        O setor de **DETECÇÃO** com o maior volume de falhas é **{top_setor_deteccao}** ({falhas_por_setor_deteccao.iloc[0]['Contagem']} ocorrências).
        
        No entanto, o setor de **ORIGEM** mais provável das falhas é **{top_origem}**. Isso indica que a maioria dos problemas está sendo criada lá.
        
        {falhas_detalhadas}
        
        **Ação Recomendada:** Concentre a investigação de **Causa Raiz** no setor de **ORIGEM** ({top_origem}) para corrigir os processos que geram as falhas listadas.
    """
    
    dicas = [
        Tip(title="Ação Estratégica", detail=f"O foco deve ser a melhoria contínua dos processos do setor de **ORIGEM** ({top_origem})."),
        Tip(title="Ponto de Controle", detail=f"O setor de **DETECÇÃO** ({top_setor_deteccao}) deve ser mantido como o principal ponto de controle de qualidade."),
    ]
    
    return {
        'status': 'OK', 
        'summary': resumo, 
        'visualization_data': [chart_deteccao.model_dump(), chart_origem.model_dump()], 
        'tips': dicas
    }

def run_structured_default_analysis(df: pd.DataFrame, query: str) -> Dict[str, Any]:
# ... (função run_structured_default_analysis mantida)
    
    if df.empty:
        return {
            'status': 'FAIL', 
            'summary': "Não há dados válidos para realizar a Análise Estruturada Padrão.", 
            'visualization_data': [], 
            'tips': []
        }
    
    falha_col = 'falha_individual' if 'falha_individual' in df.columns else 'falha'
    
    causa_counts = df['causa_raiz_processo'].value_counts().head(3)
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
    
    return {'status': 'OK', 'summary': summary, 'visualization_data': vis_data, 'tips': dicas}

def run_dppm_definition() -> Dict[str, Any]:
# ... (função run_dppm_definition mantida)
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
        'tips': dicas
    }

def run_greeting_analysis(query: str) -> Dict[str, Any]:
# ... (função run_greeting_analysis mantida)
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
        Tip(title="Desvio de Processo", detail="Pergunte: 'Análise do tópico das observações'"),
    ]
    
    return {
        'status': 'OK', 
        'summary': summary, 
        'visualization_data': [], 
        'tips': dicas
    }

# --- FUNÇÃO PRINCIPAL DE ORQUESTRAÇÃO AVANÇADA (ASSÍNCRONA) ---

async def handle_query_analysis(query: str, data_to_analyze: List[Dict]) -> AnalysisResponse:
    """
    IA Cortex Inteligente+: 
    - Usa memória para contexto (Ideia 1)
    - Detecta intenções com motor semântico (Ideia 2)
    - Executa análises em paralelo/compostas (Ideia 3)
    - Gera previsões (Ideia 5) e insights de LLM (Ideia 4)
    """
    query_lower = query.lower().strip()

    # 0. TRATAMENTO DE CUMPRIMENTOS e DEFINIÇÕES
    greeting_patterns = r'^(oi|olá|bom dia|boa tarde|boa noite|tudo bem|e aí)[\s.,!?]*$'
    if re.match(greeting_patterns, query_lower):
        return AnalysisResponse(query=query, **run_greeting_analysis(query))

    definition_patterns = ['o que é dppm', 'dppm o que é', 'o que significa dppm', 'definição de dppm', 'o que e dppm']
    if any(pattern in query_lower for pattern in definition_patterns):
        return AnalysisResponse(query=query, **run_dppm_definition())

    # 1. Pré-processamento e Contexto (Ideia 1)
    df = prepare_dataframe(data_to_analyze, flatten_multifalha=True) 

    if df.empty or len(data_to_analyze) == 0:
        return AnalysisResponse(
            query=query,
            summary="Nenhum dado encontrado para análise ou dados inválidos. Verifique a seleção de dados.",
            tips=[Tip(title="Base de Dados Vazia", detail="Verifique a fonte de dados e o filtro inicial.")],
            visualization_data=[]
        )
        
    # Tratamento de Continuação (Ideia 1)
    if "continuar" in query_lower or "agora me mostre" in query_lower:
        last_context = memory.last()
        if last_context:
            return AnalysisResponse(
                query=query,
                summary=f"Continuando a partir da última análise ({last_context[-1]['query']}):\n\n{last_context[-1]['summary']}",
                tips=[Tip(title="Contexto", detail="Reutilizei o resumo da análise anterior para dar continuidade à sua exploração.")]
            )

    # 2. Detecta Intenções (Ideia 2)
    active_intents = detect_intents(query_lower)
    
    # 3. Execução Paralela (Ideia 3)
    tasks_to_run = []
    
    if "qualidade" in active_intents:
        tasks_to_run.append(asyncio.to_thread(run_quality_analysis, df, query))
    if "setor" in active_intents:
        tasks_to_run.append(asyncio.to_thread(run_sector_analysis, df, query))
    if "causa_raiz" in active_intents:
        tasks_to_run.append(asyncio.to_thread(run_root_cause_analysis, df, query))
    # NLP é a única análise que precisa de 'await' direto
    if "nlp" in active_intents:
        tasks_to_run.append(run_nlp_analysis(df, query))
        
    # Fallback/Default se nenhuma intenção específica foi encontrada
    if not tasks_to_run:
        tasks_to_run.append(asyncio.to_thread(run_structured_default_analysis, df, query))

    # Executa todas as tarefas simultaneamente (Ideia 3)
    executed_results = await asyncio.gather(*tasks_to_run)
    executed_results = [r for r in executed_results if r.get("status") == "OK"]

    # Caso todas as análises específicas falhem
    if not executed_results:
        # Garante o Fallback Estruturado, mesmo que a primeira chamada tenha falhado
        default_analysis = run_structured_default_analysis(df, query)
        memory.add(query, default_analysis.get('summary', ''))
        return AnalysisResponse(query=query, **default_analysis)

    # 4. Combina Múltiplos Resultados
    combined_summary = "\n\n".join(r["summary"] for r in executed_results)
    combined_vis = [v for r in executed_results for v in r.get("visualization_data", [])]
    combined_tips = [t for r in executed_results for t in r.get("tips", [])]

    # 5. Forecast Automático (Ideia 5)
    next_forecast = forecast_next_period(df)
    if next_forecast is not None and next_forecast > 0:
        combined_summary += f"\n\n📈 **Previsão de Risco:** Se o padrão se mantiver, o próximo período pode registrar cerca de **{next_forecast:.0f} falhas** (baseado na tendência dos dados de falha)."
        combined_tips.append(Tip(title="Ação Preventiva", detail="Avalie planos de mitigação para o próximo ciclo, dado o risco de aumento de falhas."))
    
    # 6. Insights Automáticos com Gemini (Ideia 4)
    # Chama o LLM para sumarizar e dar um insight estratégico sobre os resultados combinados
    try:
        gemini_insight = await summarize_analysis_with_gemini({
            "query": query,
            "summary": combined_summary,
            "tips": [t.detail for t in combined_tips],
        })
        if gemini_insight and len(gemini_insight) > 20:
            combined_summary += "\n\n🧠 **Insight Estratégico da IA:** " + gemini_insight
    except Exception:
        # Se o LLM falhar, a análise de Pandas/Python ainda é retornada
        pass 

    # 7. Atualiza Memória e Retorna (Ideia 1)
    memory.add(query, combined_summary)

    return AnalysisResponse(
        query=query,
        summary=combined_summary,
        tips=combined_tips,
        visualization_data=combined_vis
    )