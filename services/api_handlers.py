# services/api_handlers.py
import json
import pandas as pd
from typing import Dict, List, Any
from datetime import datetime
import schemas 

# Importações dos módulos modulares
from .preprocessing import prepare_dataframe, extract_period_and_date
# Importa diretamente as funções de análise, sem a necessidade de uma camada extra aqui
from .ia_core import analisar_checklist, analisar_checklist_multifalha, classificar_observacao_topico

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
    Função de IA que usa o Transformer, Scikit-learn (implícito) e a lógica de Processo
    para análise de dados tabulares baseada em query.
    """
    # Pre-processamento inicial do DataFrame
    df = prepare_dataframe(data_to_analyze)

    if df.empty or len(data_to_analyze) == 0:
        return AnalysisResponse(
            query=query,
            summary="Nenhum dado encontrado para análise ou dados inválidos após o pré-processamento.",
            tips=[Tip(title="Base de Dados Vazia", detail="Verifique a fonte de dados e o filtro inicial.")]
        )

    query_lower = query.lower()

    # ----------------------------------------------------
    # LÓGICA 1: TAXA DE FALHA / QUALIDADE
    # ----------------------------------------------------
    if 'taxa de falha' in query_lower or 'qualidade' in query_lower or 'produção' in query_lower or 'compare' in query_lower:
        
        period, specific_date, granularity_name = extract_period_and_date(query)
        
        df_filtered = df.copy()
        
        # Filtro por Data Específica
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
                return AnalysisResponse(query=query, summary=f"Não há dados de falha para o período solicitado: **{granularity_name}**.", tips=[Tip(title="Verificar Intervalo", detail="Ajuste a data ou verifique se o filtro de dados inicial (data_to_analyze) está correto.")])
        
        # Agregação
        if period == 'G':
            period = 'M' 
            granularity_name = 'Mensal'

        if period == 'D':
            df_filtered['periodo'] = df_filtered['data_registro'].dt.strftime('%Y-%m-%d')
        elif period == 'M':
            df_filtered['periodo'] = df_filtered['data_registro'].dt.strftime('%Y-%m')
        else:
            # Fallback seguro
            df_filtered['periodo'] = df_filtered['data_registro'].dt.strftime('%Y-%m') # Agrupa por Mês por padrão

        summary = df_filtered.groupby('periodo').agg(
            total_falhas_periodo=('quantidade', 'sum'),
            total_producao_periodo=('quantidade_produzida', 'sum')
        ).reset_index()

        # Cálculo da Taxa de Falha
        summary['producao_safe'] = summary['total_producao_periodo'].apply(lambda x: x if x > 0 else 1)
        summary['taxa_falha_periodo'] = (summary['total_falhas_periodo'] / summary['producao_safe']) * 100

        # Resumo, Visualização e Dicas
        media_geral = summary['taxa_falha_periodo'].mean()
        top_period_taxa = summary.sort_values(by='taxa_falha_periodo', ascending=False).iloc[0] if not summary.empty else {'periodo': 'N/A', 'taxa_falha_periodo': 0}

        resumo = f"""
            **Análise de Qualidade ({granularity_name})**
            
            A **média da taxa de falha** no período analisado é de **{media_geral:.3f}%**.
            O período com a **maior taxa de falha** foi **{top_period_taxa['periodo']}**, com **{top_period_taxa['taxa_falha_periodo']:.3f}%**.
            O controle de qualidade deve focar em reduzir a média.
        """
        
        vis_data = {
            "title": f"Tendência da Taxa de Falha (%) - Agregação {granularity_name}",
            "labels": summary['periodo'].tolist(),
            "datasets": [
                {"label": "Taxa de Falha Média (%)", "data": summary['taxa_falha_periodo'].tolist(), "type": 'line', "borderColor": 'rgb(255, 99, 132)', "backgroundColor": 'rgba(255, 99, 132, 0.5)'}
            ]
        }
        
        dicas = [
            Tip(title="Foco na Qualidade", detail=f"Analise o período de pico ({top_period_taxa['periodo']}) para identificar o motivo do desvio."),
            Tip(title="Monitoramento", detail=f"O acompanhamento é crucial. Tente reduzir a média geral para **{(media_geral * 0.9):.3f}%**.")
        ]
        
        return AnalysisResponse(query=query, summary=resumo, visualization_data=vis_data, tips=dicas)


    # ----------------------------------------------------
    # LÓGICA 2: ANÁLISE DE CAUSA RAIZ E LINHA DE PRODUTO 
    # ----------------------------------------------------
    elif 'causa raiz' in query_lower or 'falhas por linha' in query_lower:
        causa_raiz_counts = df['causa_raiz_processo'].value_counts(normalize=True).mul(100).round(2).reset_index()
        causa_raiz_counts.columns = ['causa_raiz', 'percentual']
        top_causa = causa_raiz_counts.iloc[0]['causa_raiz'] if not causa_raiz_counts.empty else "N/A"
        
        linha_counts = df['linha_produto'].value_counts(normalize=True).mul(100).round(2).reset_index()
        linha_counts.columns = ['linha_produto', 'percentual']
        top_linha = linha_counts.iloc[0]['linha_produto'] if not linha_counts.empty else "N/A"

        resumo = f"""
            **Análise de Prioridade (Conhecimento de Processo)**
            
            A principal causa de falhas, baseada na sua classificação, é **{top_causa}**, representando **{causa_raiz_counts.iloc[0]['percentual'] if not causa_raiz_counts.empty else 0.0}%** do total.
            
            A linha de produtos com maior incidência de falhas é a **Linha {top_linha}** (**{linha_counts.iloc[0]['percentual'] if not linha_counts.empty else 0.0}%** das ocorrências). Isso aponta para um foco imediato nesta linha.
        """
        
        vis_data = {
            "title": "Distribuição de Falhas por Causa Raiz (Processo)",
            "labels": causa_raiz_counts['causa_raiz'].tolist(),
            "datasets": [
                {"label": "Percentual de Falhas", "data": causa_raiz_counts['percentual'].tolist(), "type": 'bar'}
            ]
        }
        
        dicas = [
            Tip(title=f"Ação para {top_causa}", detail=f"Conforme seu mapeamento, foque na resolução do problema de 'Componente/Processo' associado a esta causa."),
            Tip(title=f"Ação na Linha {top_linha}", detail="Realize auditorias nos procedimentos de montagem e teste dos produtos da Linha de " + top_linha + "."),
        ]
        
        return AnalysisResponse(query=query, summary=resumo, visualization_data=vis_data, tips=dicas)


    # ----------------------------------------------------
    # LÓGICA 3: ANÁLISE DE OBSERVAÇÕES (Transformer/NLP)
    # ----------------------------------------------------
    elif 'tópico' in query_lower or 'nlp' in query_lower or 'observações' in query_lower:
        
        df['causa_raiz_ia'] = df['observacao_combinada'].apply(classificar_observacao_topico)
        
        analise_topico = df.groupby('causa_raiz_ia')['documento_id'].count().sort_values(ascending=False).reset_index(name='Contagem')
        
        if analise_topico.empty:
             return AnalysisResponse(query=query, summary="A análise de observações não retornou resultados válidos.", tips=[])
             
        vis_data = {
            "title": "Contagem por Causa Raiz (Análise IA do Texto)",
            "labels": analise_topico['causa_raiz_ia'].tolist(),
            "datasets": [
                {"label": "Contagem", "data": analise_topico['Contagem'].tolist(), "backgroundColor": ["#4bc0c0", "#ff6384", "#ffcd56", "#36a2eb", "#9966ff"]}
            ]
        }
        
        top_causa = analise_topico.iloc[0]['causa_raiz_ia']
        
        resumo = f"A **Análise de Tópicos (Transformer)** nas observações revelou que a principal causa raiz identificada (via texto livre) é **{top_causa}**, com {analise_topico.iloc[0]['Contagem']} ocorrências."
        
        dicas = [
            Tip(title="Ação por Tópico", detail=f"Se a principal causa é '{top_causa}', revise as instruções ou materiais para a prevenção."),
            Tip(title="Validação", detail="Compare esta classificação de NLP com a classificação tabular de 'causa_raiz_processo' para validar a precisão.")
        ]
        
        return AnalysisResponse(query=query, summary=resumo, visualization_data=vis_data, tips=dicas)
        

    # ----------------------------------------------------
    # LÓGICA 4: ANÁLISE DE DESVIO POR SETOR
    # ----------------------------------------------------
    elif 'setor' in query_lower or 'revisão' in query_lower:
        
        falhas_por_setor = df.groupby('setor')['falha'].count().sort_values(ascending=False).reset_index(name='Contagem')
        
        if falhas_por_setor.empty:
            return AnalysisResponse(query=query, summary="A análise de falhas por setor não retornou resultados.", tips=[])
            
        vis_data = {
            "title": "Total de Falhas Registradas por Setor",
            "labels": falhas_por_setor['setor'].tolist(),
            "datasets": [
                {"label": "Total de Falhas", "data": falhas_por_setor['Contagem'].tolist(), "backgroundColor": "rgba(255, 99, 132, 0.6)" }
            ]
        }
        
        top_setor = falhas_por_setor.iloc[0]['setor']
        
        resumo = f"O setor **{top_setor}** registrou o maior volume de falhas ({falhas_por_setor.iloc[0]['Contagem']} ocorrências). Isso indica que este é o ponto focal de detecção."
        
        dicas = [
            Tip(title="Foco de Detecção", detail=f"Concentrar auditorias de processo no setor '{top_setor}' para entender por que as falhas são detectadas ali."),
        ]
        
        return AnalysisResponse(query=query, summary=resumo, visualization_data=vis_data, tips=dicas)

    # Resposta Padrão/Geral
    try:
        # Pega a falha mais comum considerando as falhas individuais (se 'flattened') ou a coluna 'falha'
        falha_col = 'falha_individual' if 'falha_individual' in df.columns else 'falha'
        top_falha = df[falha_col].dropna().mode().iat[0]
    except Exception:
        top_falha = "N/A (Coluna de falha vazia)"
        
    return AnalysisResponse(
        query=query,
        summary=f"A IA realizou uma análise geral sobre **{len(df)}** registros. A falha mais comum é **'{top_falha}'**.",
        tips=[
            Tip(title="Sugestão de Busca Avançada", detail="Tente buscar por **'taxa de falha diária'**, **'taxa de falha mensal'** ou **'taxa de falha 05/2025'** para análises temporais específicas."),
            Tip(title="Filtro de Dados", detail="Se a análise não for a esperada, verifique os filtros de data aplicados antes de enviar os dados.")
        ]
    )