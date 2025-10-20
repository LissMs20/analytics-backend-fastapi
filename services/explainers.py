# services/explainers.py
import pandas as pd
from typing import Tuple, List, Dict, Any

# Supondo que você tenha o schema ChartData aqui ou uma representação simples
# Usaremos uma representação simples Dict[str, Any] para compatibilidade

def generate_explanation(df: pd.DataFrame, tipo: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Transforma dados de falhas em uma explicação técnica e lista de gráficos/dados
    sem usar IA generativa.
    """
    if df.empty or 'quantidade' not in df.columns or df['quantidade'].sum() == 0:
        return "Sem dados válidos para análise no período selecionado.", []

    df['quantidade'] = pd.to_numeric(df['quantidade'], errors='coerce').fillna(0)
    charts = []
    
    # Colunas que você pode querer usar (ajuste se necessário)
    falha_col = 'falha_individual' if 'falha_individual' in df.columns else 'falha'
    setor_col = 'setor_origem' if 'setor_origem' in df.columns else 'setor_falha_individual'
    causa_col = 'causa_raiz_detalhada' if 'causa_raiz_detalhada' in df.columns else 'causa_raiz_processo'


    if tipo in ["falhas", "general"] and falha_col in df.columns:
        summary_df = df.groupby(falha_col)["quantidade"].sum().reset_index()
        top = summary_df.sort_values("quantidade", ascending=False).head(3)
        if not top.empty:
            top_list = top[falha_col].tolist()
            texto = f"As falhas mais frequentes são **{', '.join(top_list)}**, totalizando **{top['quantidade'].sum():.0f}** ocorrências neste período."
            
            charts.append({
                "title": f"Top 3 Falhas ({falha_col})",
                "labels": top[falha_col].tolist(),
                "datasets": [{"label": "Contagem", "data": top["quantidade"].tolist(), "type": "pie", "backgroundColor": ["#ff6384", "#36a2eb", "#ffcd56"]}]
            })
            if tipo == "falhas": return texto, charts
            
    if tipo in ["setores", "general"] and setor_col in df.columns:
        summary_df = df.groupby(setor_col)["quantidade"].sum().reset_index()
        top = summary_df.sort_values("quantidade", ascending=False).head(3)
        if not top.empty:
            top_list = top[setor_col].tolist()
            texto = f"Os setores com mais falhas reportadas são **{', '.join(top_list)}**. Concentre a auditoria de processo nestas áreas."
            charts.append({
                "title": f"Top 3 Setores ({setor_col})",
                "labels": top[setor_col].tolist(),
                "datasets": [{"label": "Contagem", "data": top["quantidade"].tolist(), "type": "bar", "backgroundColor": ["#4bc0c0", "#ff9f40", "#9966ff"]}]
            })
            if tipo == "setores": return texto, charts

    if tipo in ["causas", "general"] and causa_col in df.columns:
        summary_df = df.groupby(causa_col)["quantidade"].sum().reset_index()
        top = summary_df.sort_values("quantidade", ascending=False).head(3)
        if not top.empty:
            top_list = top[causa_col].tolist()
            texto = f"As principais causas-raiz no período são **{', '.join(top_list)}**. Isso requer uma ação corretiva imediata do time de Engenharia."
            charts.append({
                "title": f"Top 3 Causas ({causa_col})",
                "labels": top[causa_col].tolist(),
                "datasets": [{"label": "Contagem", "data": top["quantidade"].tolist(), "type": "pie", "backgroundColor": ["#ff6384", "#36a2eb", "#ffcd56"]}]
            })
            if tipo == "causas": return texto, charts

    # Fallback para "general" ou se as colunas primárias não existirem
    texto_geral = f"A análise estatística padrão mostra **{df['quantidade'].sum():.0f}** falhas no total. O sistema está exibindo os dados primários encontrados para o período."
    return texto_geral, charts