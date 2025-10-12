import json
import pandas as pd
import numpy as np
import re
from typing import Dict, List, Any
from datetime import datetime

# Importações de schemas (Assumindo que schemas é um módulo local ou está na raiz)
import schemas 

# Importações dos módulos modulares (mantidas as originais)
from .preprocessing import prepare_dataframe, extract_period_and_date
# Importa a função de NLP para a Lógica 3
from .ia_core import classificar_observacao_topico 

AnalysisResponse = schemas.AnalysisResponse
Tip = schemas.Tip

# Assumindo que o ChartData é o schema de Pydantic que você definiu para um único gráfico
ChartData = schemas.ChartData 

# --- FUNÇÕES DE ANÁLISE DETALHADA (Auxiliares para Orquestração) ---

def _extract_origin_sector(causa_raiz: str) -> str:
    """Extrai o setor de origem da string de Causa Raiz de Processo."""
    if not isinstance(causa_raiz, str):
        return 'Setor Desconhecido'
    
    # Ex: 'Falha no Processo (Máquina de Solda/Revisão)' -> 'Máquina de Solda'
    match = re.search(r'\((.*?)\)', causa_raiz)
    if match:
        # Simplifica para o primeiro setor (ex: Máquina de Solda)
        origin_str = match.group(1)
        return origin_str.split('/')[0].strip()
    return 'Geral/Outros'

def run_quality_analysis(df: pd.DataFrame, query: str) -> Dict[str, Any]:
    """LÓGICA 1: Calcula e visualiza o DPPM (Defeitos por Milhão) ao longo do tempo."""
    
    period, specific_date, granularity_name = extract_period_and_date(query)
    df_filtered = df.copy()
    
    # 1. Aplica o filtro de data (Se necessário, aprimorar a lógica de filtragem de data relativa aqui)
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

    # 2. Agregação e Cálculo de DPPM (Melhoria 1)
    if period == 'G':
        period = 'M' 
        granularity_name = 'Mensal'

    # Reseta o índice de falhas individuais para a contagem correta
    df_filtered = df_filtered.reset_index(drop=True) 

    df_filtered['periodo'] = df_filtered['data_registro'].dt.to_period(period).astype(str)
    
    summary_data = df_filtered.groupby('periodo').agg(
        total_falhas_periodo=('quantidade', 'sum'),
        total_producao_periodo=('quantidade_produzida', 'sum') 
    ).reset_index()

    summary_data['total_producao_safe'] = summary_data['total_producao_periodo'].apply(lambda x: x if x > 0 else 1)
    # Cálculo do DPPM (Defeitos por Milhão)
    summary_data['dppm_periodo'] = (summary_data['total_falhas_periodo'] / summary_data['total_producao_safe']) * 1_000_000

    # 3. Resumo
    media_geral = summary_data['dppm_periodo'].mean()
    top_period_dppm = summary_data.sort_values(by='dppm_periodo', ascending=False).iloc[0] if not summary_data.empty else {'periodo': 'N/A', 'dppm_periodo': 0}

    resumo = f"""
        **Análise de Qualidade ({granularity_name})**
        A **média de DPPM** no período analisado é de **{media_geral:.2f}**.
        O período com o **maior DPPM** foi **{top_period_dppm['periodo']}**, com **{top_period_dppm['dppm_periodo']:.2f}**.
        O controle de qualidade deve focar em reduzir o DPPM médio e analisar o período de pico.
    """
    
    # ATENÇÃO: vis_data (um único gráfico) deve ser retornado como uma lista de um item.
    vis_data = ChartData(
        title=f"Tendência do DPPM (Defeitos por Milhão) - Agregação {granularity_name}",
        labels=summary_data['periodo'].tolist(),
        datasets=[
            {"label": "DPPM", "data": summary_data['dppm_periodo'].tolist(), "type": 'line', "borderColor": 'rgb(255, 99, 132)', "backgroundColor": 'rgba(255, 99, 132, 0.5)'}
        ],
        chart_type='line' # Adicionado o tipo de gráfico
    )
    
    dicas = [
        Tip(title="Foco no Desvio", detail=f"Analise o período de pico ({top_period_dppm['periodo']}) para identificar a causa do alto DPPM."),
        Tip(title="Meta", detail=f"O acompanhamento é crucial. Tente reduzir a média geral para **{(media_geral * 0.9):.2f}** DPPM."),
        Tip(title="Métrica", detail="DPPM é a métrica padrão da indústria para defeitos de qualidade de produção."),
    ]
    
    return {'status': 'OK', 'summary': resumo, 'visualization_data': [vis_data.model_dump()], 'tips': dicas}

def run_root_cause_analysis(df: pd.DataFrame, query: str) -> Dict[str, Any]:
    """LÓGICA 2: Analisa a distribuição de falhas por Causa Raiz e Linha de Produto."""
    
    causa_raiz_counts = df['causa_raiz_processo'].value_counts(normalize=True).mul(100).round(2).reset_index(name='percentual')
    causa_raiz_counts.columns = ['causa_raiz', 'percentual']
    
    top_causa = causa_raiz_counts.iloc[0]['causa_raiz'] if not causa_raiz_counts.empty else "N/A"
    top_causa_perc = causa_raiz_counts.iloc[0]['percentual'] if not causa_raiz_counts.empty else 0.0

    linha_counts = df['linha_produto'].value_counts(normalize=True).mul(100).round(2).reset_index(name='percentual')
    linha_counts.columns = ['linha_produto', 'percentual']
    top_linha = linha_counts.iloc[0]['linha_produto'] if not linha_counts.empty else "N/A"
    top_linha_perc = linha_counts.iloc[0]['percentual'] if not linha_counts.empty else 0.0

    resumo = f"""
        **Análise de Prioridade (Conhecimento de Processo)**
        A principal causa de falhas é **{top_causa}**, representando **{top_causa_perc}%** do total.
        A linha de produtos com maior incidência de falhas é a **Linha {top_linha}** (**{top_linha_perc}%** das ocorrências).
    """
    
    # ATENÇÃO: vis_data (um único gráfico) deve ser retornado como uma lista de um item.
    vis_data = ChartData(
        title="Distribuição de Falhas por Causa Raiz (Processo)",
        labels=causa_raiz_counts['causa_raiz'].tolist(),
        datasets=[
            {"label": "Percentual de Falhas", "data": causa_raiz_counts['percentual'].tolist(), "type": 'bar'}
        ],
        chart_type='bar'
    )
    
    dicas = [
        Tip(title=f"Ação Prioritária (Causa)", detail=f"Concentre esforços na causa **'{top_causa}'**."),
        Tip(title=f"Ação Prioritária (Produto)", detail="Realize auditorias nos procedimentos de montagem e teste dos produtos da Linha de " + top_linha + "."),
    ]
    
    return {'status': 'OK', 'summary': resumo, 'visualization_data': [vis_data.model_dump()], 'tips': dicas}

def run_nlp_analysis(df: pd.DataFrame, query: str) -> Dict[str, Any]:
    """LÓGICA 3: Analisa a distribuição de falhas por Tópico (NLP do texto livre)."""
    
    if df['observacao_combinada'].isnull().all() or df['observacao_combinada'].str.strip().eq('').all():
        return {'status': 'FAIL', 'summary': "Análise de Tópicos não executada: A maioria das observações está vazia ou nula.", 'visualization_data': []}
    
    # Aplica a função de NLP do ia_core
    df['causa_raiz_ia'] = df['observacao_combinada'].apply(classificar_observacao_topico)
    
    analise_topico = df.groupby('causa_raiz_ia')['documento_id'].count().sort_values(ascending=False).reset_index(name='Contagem')
    
    if analise_topico.empty or analise_topico.iloc[0]['causa_raiz_ia'].startswith("N/A"):
        return {'status': 'FAIL', 'summary': "A análise de observações não retornou resultados válidos (modelo NLP indisponível ou textos muito curtos).", 'visualization_data': []}
            
    # ATENÇÃO: vis_data (um único gráfico) deve ser retornado como uma lista de um item.
    vis_data = ChartData(
        title="Contagem por Causa Raiz (Análise IA do Texto)",
        labels=analise_topico['causa_raiz_ia'].tolist(),
        datasets=[
            {"label": "Contagem", "data": analise_topico['Contagem'].tolist(), "backgroundColor": ["#4bc0c0", "#ff6384", "#ffcd56", "#36a2eb", "#9966ff"]}
        ],
        chart_type='pie' # Sugerido Pie para distribuição de tópicos
    )
    
    top_causa_ia = analise_topico.iloc[0]['causa_raiz_ia']
    
    resumo = f"A principal causa raiz identificada (via texto livre - NLP) é **{top_causa_ia}**, com {analise_topico.iloc[0]['Contagem']} ocorrências."
    
    dicas = [
        Tip(title="Ação por Tópico", detail=f"Se a principal causa é '{top_causa_ia}', revise as instruções ou materiais para a prevenção."),
        Tip(title="Validação", detail="Compare esta classificação de NLP com a classificação tabular para validar a precisão.")
    ]
    
    return {'status': 'OK', 'summary': resumo, 'visualization_data': [vis_data.model_dump()], 'tips': dicas}

def run_sector_analysis(df: pd.DataFrame, query: str) -> Dict[str, Any]:
    """LÓGICA 4: Analisa o setor de DETECÇÃO e o setor de ORIGEM, retornando DOIS gráficos."""
    
    if df.empty:
        return {'status': 'FAIL', 'summary': "Análise de Setor não executada: DataFrame vazio.", 'visualization_data': []}

    # --- 1. Análise de Setor de DETECÇÃO (Onde a falha foi encontrada) ---
    falhas_por_setor_deteccao = df.groupby('setor_falha_individual')['falha_individual'] \
                            .count() \
                            .sort_values(ascending=False) \
                            .reset_index(name='Contagem')
    
    if falhas_por_setor_deteccao.empty:
        return {'status': 'FAIL', 'summary': "Análise de Setor não executada: Dados de setor de falha ausentes.", 'visualization_data': []}
            
    top_setor_deteccao = falhas_por_setor_deteccao.iloc[0]['setor_falha_individual']

    # Gráfico 1: Setor de Detecção
    chart_deteccao = ChartData(
        title="1. Volume de Falhas por Setor de DETECÇÃO",
        labels=falhas_por_setor_deteccao['setor_falha_individual'].tolist(),
        datasets=[
            {"label": "Total de Falhas", "data": falhas_por_setor_deteccao['Contagem'].tolist(), "backgroundColor": "rgba(255, 99, 132, 0.7)" }
        ],
        chart_type='bar'
    )

    # --- 2. Análise de Setor de ORIGEM (Onde o problema foi causado) ---
    df['setor_origem'] = df['causa_raiz_processo'].apply(_extract_origin_sector)
    falhas_por_setor_origem = df.groupby('setor_origem')['documento_id'].count().sort_values(ascending=False).reset_index(name='Contagem')
    
    top_origem = falhas_por_setor_origem.iloc[0]['setor_origem'] if not falhas_por_setor_origem.empty else "N/A"

    # Gráfico 2: Setor de Origem
    chart_origem = ChartData(
        title="2. Volume de Falhas por Setor de ORIGEM",
        labels=falhas_por_setor_origem['setor_origem'].tolist(),
        datasets=[
            {"label": "Total de Falhas", "data": falhas_por_setor_origem['Contagem'].tolist(), "backgroundColor": "rgba(54, 162, 235, 0.7)" }
        ],
        chart_type='bar'
    )
    
    # 💡 NOVO: Análise detalhada das falhas no setor de origem principal
    falhas_do_top_origem = df[df['setor_origem'] == top_origem]
    top_falhas_no_origem = falhas_do_top_origem['falha_individual'].value_counts().head(3).reset_index()
    top_falhas_no_origem.columns = ['falha', 'contagem']

    falhas_detalhadas = ""
    if not top_falhas_no_origem.empty:
        falhas_detalhadas = "As falhas mais comuns neste setor de origem são:\n"
        for _, row in top_falhas_no_origem.iterrows():
            falhas_detalhadas += f"- **{row['falha']}** ({row['contagem']} ocorrências)\n"
    
    # --- 3. Resumo Combinado e Melhorado ---
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
    
    # 4. Retorna uma LISTA de gráficos, convertendo os Pydantic models para dicionários
    return {
        'status': 'OK', 
        'summary': resumo, 
        'visualization_data': [chart_deteccao.model_dump(), chart_origem.model_dump()], 
        'tips': dicas
    }

def default_analysis(df: pd.DataFrame, query: str) -> AnalysisResponse:
    """Análise de fallback se nenhuma query específica for encontrada."""
    try:
        falha_col = 'falha_individual' if 'falha_individual' in df.columns else 'falha'
        top_falha = df[falha_col].dropna().mode().iat[0]
    except Exception:
        top_falha = "N/A (Coluna de falha vazia)"
        
    return AnalysisResponse(
        query=query,
        summary=f"A IA realizou uma análise geral sobre **{len(df)}** registros. A falha mais comum é **'{top_falha}'**. Nenhuma consulta específica foi detectada.",
        tips=[
            Tip(title="Sugestão de Busca Avançada", detail="Tente buscar por **'taxa de falha diária'**, **'causa raiz'** ou **'tópico das observações'**.")
        ],
        visualization_data=[] # Garante que o fallback retorne uma lista vazia
    )

def run_dppm_definition() -> Dict[str, Any]:
    """LÓGICA DEFINITION: Explica o que é DPPM."""
    
    summary = """
        **DPPM** significa **Defeitos Por Milhão**.
        
        É uma métrica de qualidade que mede a quantidade de peças defeituosas ou falhas encontradas a cada 1 milhão de unidades produzidas.
        
        **Por que é importante?**
        - **Padronização:** Permite comparar a qualidade de diferentes processos ou fábricas.
        - **Alta Precisão:** É ideal para processos de alta qualidade, onde a taxa de defeitos é muito baixa (ex: 0,05%).
        - **Análise de Tendência:** Seu acompanhamento ao longo do tempo indica se o processo de produção está melhorando (DPPM diminuindo) ou piorando (DPPM aumentando).
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

# NOVA FUNÇÃO: Resposta a cumprimentos simples
def run_greeting_analysis(query: str) -> Dict[str, Any]:
    """LÓGICA GREETING: Responde a cumprimentos e oferece sugestões de consulta."""
    
    # Determina o cumprimento baseado na hora do dia (melhora a personalização)
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

    # Dicas de consulta não óbvias
    dicas = [
        Tip(title="Tendência de Qualidade", detail="Pergunte: 'Qual é o DPPM mensal da produção?'"),
        Tip(title="Foco Geográfico", detail="Pergunte: 'Quais são as falhas mais comuns no lado A da placa?'"),
        Tip(title="Desvio de Processo", detail="Pergunte: 'A principal causa raiz tem relação com o setor de SMT?'"),
        Tip(title="Análise Preditiva", detail="Pergunte: 'Existe alguma observação que indique problemas com o processo de solda?'"),
    ]
    
    # Retorna uma resposta estruturada sem dados de visualização
    return {
        'status': 'OK', 
        'summary': summary, 
        'visualization_data': [], # Lista vazia, pois não há gráfico
        'tips': dicas
    }

# --- FUNÇÃO PRINCIPAL DE ORQUESTRAÇÃO DE ANÁLISE ---

def handle_query_analysis(query: str, data_to_analyze: List[Dict]) -> AnalysisResponse:
    """
    Função principal de IA que orquestra as análises estatísticas/domínio
    com base na consulta do usuário.
    """
    query_lower = query.lower()
    analysis_results = {}
    
    # 0. NOVO: TRATAMENTO DE CUMPRIMENTOS (Prioridade Máxima)
    # Define padrões que indicam apenas um cumprimento simples
    greeting_patterns = r'^(oi|olá|bom dia|boa tarde|boa noite|tudo bem|e aí)[\s.,!?]*$'
    if re.match(greeting_patterns, query_lower):
        return AnalysisResponse(
            query=query,
            **run_greeting_analysis(query)
        )

    # NOVO: TRATAMENTO DE DEFINIÇÕES (Prioridade Alta)
    definition_patterns = [
        'o que é dppm', 'dppm o que é', 'o que significa dppm', 
        'definição de dppm', 'o que e dppm', 

        'o que é ddpm', 'ddpm o que é', 'o que significa ddpm', 
    ]
    if any(pattern in query_lower for pattern in definition_patterns):
        return AnalysisResponse(
            query=query,
            **run_dppm_definition()
        )

    # 1. Pré-processamento e Flatten Crítico
    df = prepare_dataframe(data_to_analyze, flatten_multifalha=True) 

    if df.empty or len(data_to_analyze) == 0:
        # Resposta de fallback para base de dados vazia
        return AnalysisResponse(
            query=query,
            summary="Nenhum dado encontrado para análise ou dados inválidos após o pré-processamento. Tente consultar dados em um período diferente.",
            tips=[Tip(title="Base de Dados Vazia", detail="Verifique a fonte de dados e o filtro inicial.")],
            visualization_data=[]
        )

    # 2. Mapeamento de Funções de Análise e Execução (mantido o resto da lógica)
    
    # Tenta executar a análise mais específica correspondente à query
    if 'taxa de falha' in query_lower or 'dppm' in query_lower or 'qualidade' in query_lower:
        analysis_results = run_quality_analysis(df, query)
    elif 'causa raiz' in query_lower or 'linha' in query_lower:
        analysis_results = run_root_cause_analysis(df, query)
    elif 'tópico' in query_lower or 'nlp' in query_lower or 'observações' in query_lower:
        analysis_results = run_nlp_analysis(df, query)
    elif 'setor' in query_lower or 'origem' in query_lower:
        analysis_results = run_sector_analysis(df, query)
    
    # 3. Consolidação dos Resultados
    if analysis_results and analysis_results.get('status') == 'OK':
        return AnalysisResponse(
            query=query,
            summary=analysis_results['summary'],
            tips=analysis_results['tips'],
            visualization_data=analysis_results['visualization_data']
        )
    
    # 4. Fallback para Análise Geral
    return default_analysis(df, query)