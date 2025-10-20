# routers/analysis.py (CÓDIGO COMPLETO E CORRIGIDO)

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Dict, List, Any
import pandas as pd
import json
import logging

import schemas
from models import DadoIA, RegistroProducao
from database import get_db
from auth import get_current_user
import models 

from services.api_handlers import processar_analise_checklist
# CORREÇÃO CHAVE: Importa o Orquestrador Central
from services.intelligence import get_strategic_analysis 

router = APIRouter(tags=["Análise de IA"])
logger = logging.getLogger(__name__)

# --- Endpoint de Análise de Múltiplas Falhas (SÍNCRONO) ---
@router.post("/multifalha", response_model=schemas.AnalysisResponse)
def testar_analise_multifalha(
    dados: schemas.ChecklistCreateMulti,
    current_user: models.Usuario = Depends(get_current_user)
):
    """
    Endpoint de teste para simular a análise de IA para múltiplas falhas.
    """
    try:
        dados_dict = dados.model_dump(exclude={"falhas"})
        falhas_lista = dados.falhas

        json_string = processar_analise_checklist(dados_dict, falhas_lista)
        analise = json.loads(json_string)
        
        # 2. Mapeia o resultado para o schema AnalysisResponse
        summary = analise.get("resumo_geral", "Análise concluída.")
        
        # Cria as Dicas
        tips = [
            schemas.Tip(
                title=f"Falha: {a.get('falha', 'N/A')} - {a.get('status', 'N/D')}", 
                detail=a.get('mensagem', a.get('recomendacao', 'Detalhes na análise JSON.'))
            )
            for a in analise.get('analises_individuais', [])
        ]
        
        # 3. Cria dados de visualização
        alert_count = sum(1 for a in analise.get('analises_individuais', []) if "ALERTA" in a.get('status', ''))
        rule_count = sum(1 for a in analise.get('analises_individuais', []) if "Recomendação Encontrada" in a.get('status', ''))
        
        vis_data = schemas.ChartData(
             title="Status da Análise por Falha",
             labels=["Alertas/Risco (ML)", "Recomendação (Regra)"],
             datasets=[
                 {"label": "Contagem", "data": [alert_count, rule_count], "type": 'bar', "backgroundColor": ['#E53935', '#FB8C00']}
             ],
             chart_type='bar' # Adicionado chart_type para ChartsData, se necessário
        ).model_dump()
        
        return schemas.AnalysisResponse(
            query="Multi-Falha Teste Direto",
            summary=summary,
            tips=tips,
            visualization_data=[vis_data]
        )
    except Exception as e:
        logger.exception(f"Falha na análise multi-falha: {e}")
        raise HTTPException(status_code=500, detail=f"Falha na análise multi-falha: {e}")

# --- ENDPOINT DE ANÁLISE AVANÇADA DA IA (/analyze) - CORRIGIDO ---
@router.post("/analyze", response_model=schemas.AnalysisResponse)
async def analyze_data_endpoint(
    analysis_query: schemas.AnalysisQuery,
    db: Session = Depends(get_db),
    current_user: models.Usuario = Depends(get_current_user)
):
    """
    Endpoint que recebe uma query do usuário, busca checklists, 
    combina-os e usa a IA Orquestradora (intelligence.py) para gerar análise.
    """
    
    # 1. Busca e Combina Dados (Lógica de Orquestração de Dados) - MANTIDA
    try:
        # Lógica de busca e merge de dados (SÍNCRONA)
        checklists_db = db.query(DadoIA).filter(
            DadoIA.status == 'COMPLETO',
            DadoIA.data_finalizacao.isnot(None)
        ).all()

        producao_db = db.query(RegistroProducao).filter(
            RegistroProducao.tipo_registro == 'D'
        ).all()
        
        df_checklists = pd.DataFrame([c.__dict__ for c in checklists_db]) if checklists_db else pd.DataFrame()
        df_producao = pd.DataFrame([r.__dict__ for r in producao_db]) if producao_db else pd.DataFrame()
        
        if not df_checklists.empty:
            df_checklists = df_checklists.drop(columns=['_sa_instance_state'], errors='ignore')
        if not df_producao.empty:
            df_producao = df_producao.drop(columns=['_sa_instance_state'], errors='ignore')
            
        if not df_checklists.empty and not df_producao.empty:
            df_checklists['data_finalizacao_date'] = pd.to_datetime(df_checklists['data_finalizacao']).dt.date
            df_producao['data_registro_date'] = pd.to_datetime(df_producao['data_registro']).dt.date
            
            df_merged = pd.merge(
                df_checklists,
                df_producao,
                left_on="data_finalizacao_date",
                right_on="data_registro_date",
                how="left",
                suffixes=('_falha', '_prod')
            ).fillna({"quantidade_diaria": 0})
            
            df_merged['quantidade_produzida'] = df_merged['quantidade_diaria']
            df_merged['data_registro'] = df_merged['data_finalizacao']

        elif not df_checklists.empty:
            df_merged = df_checklists.copy()
            df_merged['quantidade_produzida'] = 0
            df_merged['data_registro'] = df_merged['data_finalizacao']
        
        else:
            df_merged = pd.DataFrame() 

        # 2. Prepara dados para a IA Orquestradora
        data_to_analyze: List[Dict] = df_merged.to_dict('records') # Lista de Dicts para o Analyst/LLM
        
        # O DataFrame também é passado, pois o ML Predictor usa o DataFrame pronto
        df_for_ml = df_merged 
        
        # 3. CHAMA A IA ORQUESTRADORA (get_strategic_analysis)
        # Ela decide se usa ML (df_for_ml), Worker ou Análise Composta (data_to_analyze)
        analysis_dict_result = await get_strategic_analysis(
            df=df_for_ml, # DataFrame otimizado para ML/Pandas
            query=analysis_query.query, 
            data_to_analyze=data_to_analyze # Lista de Dicts para o Analyst (se necessário)
        )

        # 4. Converte o resultado (Dict) para o Pydantic AnalysisResponse
        # Garante que as Tips sejam convertidas corretamente de volta.
        return schemas.AnalysisResponse(
            query=analysis_query.query,
            summary=analysis_dict_result.get('summary', analysis_dict_result.get('message', 'Erro na análise')),
            tips=[schemas.Tip(**t) if isinstance(t, dict) else t for t in analysis_dict_result.get('tips', [])],
            visualization_data=analysis_dict_result.get('visualization_data', [])
        )

    except Exception as e:
        logger.exception(f"Erro na orquestração de dados ou na análise da IA: {e}")
        # Retorna erro no formato AnalysisResponse (ou levanta HTTPException)
        raise HTTPException(status_code=500, detail=f"Falha na análise avançada de dados da IA: {e}")