from fastapi import APIRouter, Depends, HTTPException, status, Query, BackgroundTasks # Importar BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Optional, Dict
from datetime import datetime
import json
from sqlalchemy import or_, cast, String

import schemas
import models
from database import get_db, SessionLocal
from auth import get_current_user, assistencia_required
from services.ia_core import analisar_checklist

router = APIRouter()

# --- Endpoint: Adicionar um Checklist ao BD (POST) ---
def run_ia_in_background(dado_ia_id: int, falhas_a_analisar: List[dict]):
    """Função síncrona que executa a IA e atualiza o campo resultado_ia em uma nova sessão."""
   
    with SessionLocal() as db_in_thread:
        # Busca o objeto db_dado dentro desta nova sessão
        db_dado = db_in_thread.query(models.DadoIA).filter(models.DadoIA.id == dado_ia_id).first()
        
        if not db_dado:
            print(f"Erro: DadoIA com ID {dado_ia_id} não encontrado na thread de IA.")
            return

        try:
            resultados = []
            for f in falhas_a_analisar:

                resultado = analisar_checklist(f) 
                resultados.append(resultado)
                
            json_string = json.dumps(resultados, ensure_ascii=False)
            
            db_dado.resultado_ia = json_string
            db_in_thread.commit()
            db_in_thread.refresh(db_dado)
            
        except Exception as e:
            # Garante que qualquer erro seja revertido
            db_in_thread.rollback() 
            print(f"Erro na análise de IA em background para {db_dado.documento_id}: {e}")


@router.post("/checklists/", response_model=schemas.Checklist) 
def criar_checklist(
    dado: dict,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: models.Usuario = Depends(get_current_user)
):
    vai_para_assistencia = dado.get("vai_para_assistencia", False)

    # Define status e IA
    status_inicial = 'PENDENTE' if vai_para_assistencia else 'COMPLETO'
    data_finalizacao_inicial = datetime.now() if status_inicial == 'COMPLETO' else None
    responsavel_assistencia_inicial = None
    
    # Prepara os dados do checklist e a lista de falhas para o BD
    is_multi = "falhas" in dado
    falhas_lista = dado.get("falhas", [])
    
    # --- 1. PREPARAÇÃO DO OBJETO DE DB (Sem o resultado_ia) ---
    if is_multi:
        db_dado = models.DadoIA(
            produto=dado["produto"],
            quantidade=dado["quantidade"],
            observacao_producao=dado.get("observacao_producao"),
            
            responsavel=dado["responsavel"],
            
            falhas_json=falhas_lista if falhas_lista else None,
            
            status=status_inicial,
            resultado_ia=None,
            documento_id="NC-TEMP",
            data_finalizacao=data_finalizacao_inicial,
            responsavel_assistencia=responsavel_assistencia_inicial,
        )
    else:
        # Modo de uma única falha - Usa o Pydantic para validar e extrair dados
        dado_schema = schemas.ChecklistCreate(**dado)
        
        # O Pydantic cria um dicionário, mas precisamos lidar com falhas_json
        # e o status/finalização manualmente para o SQLAlchemy
        dados_para_db = dado_schema.model_dump(
            exclude={'vai_para_assistencia', 'falha', 'setor', 
                     'localizacao_componente', 'lado_placa'}
        )

        # Para consistência, crie a lista de falhas para a IA e o campo falhas_json (se necessário)
        if status_inicial == 'COMPLETO' and dado_schema.falha:
            falha_unica = {
                "falha": dado_schema.falha,
                "setor": dado_schema.setor,
                "localizacao_componente": dado_schema.localizacao_componente,
                "lado_placa": dado_schema.lado_placa,
                "observacao_producao": dado_schema.observacao_producao
            }
            falhas_lista.append(falha_unica)
            
        db_dado = models.DadoIA(
            **dados_para_db,
            # Se for modo single, as falhas são salvas nos campos diretos do modelo
            falha=dado_schema.falha if dado_schema.falha else None,
            setor=dado_schema.setor if dado_schema.setor else None,
            localizacao_componente=dado_schema.localizacao_componente if dado_schema.localizacao_componente else None,
            lado_placa=dado_schema.lado_placa if dado_schema.lado_placa else None,
            
            falhas_json=None,
            resultado_ia=None,
            status=status_inicial,
            documento_id="NC-TEMP",
            data_finalizacao=data_finalizacao_inicial,
            responsavel_assistencia=responsavel_assistencia_inicial,
        )

    # --- 2. SALVA NO BANCO (Resposta Rápida) ---
    db.add(db_dado)
    db.commit()
    db.refresh(db_dado)

    # Gera documento_id
    db_dado.documento_id = f"NC{db_dado.id:05d}"
    db.commit()
    db.refresh(db_dado)

    # --- 3. DISPARA A IA EM BACKGROUND (SÓ SE COMPLETO) ---
    if status_inicial == "COMPLETO" and falhas_lista:

        background_tasks.add_task(run_ia_in_background, db_dado.id, falhas_lista)

    return db_dado

# --- Endpoint: Listar Dados com Filtro de Status (GET) ---

@router.get("/checklists/", response_model=schemas.PaginatedChecklists)
def listar_dados(
    db: Session = Depends(get_db),
    status: Optional[str] = Query(None, description="Filtrar por status: COMPLETO ou PENDENTE."),
    search: Optional[str] = Query(None, description="Pesquisar por parte do Documento ID ou número interno."),
    page: Optional[int] = Query(None, ge=1, description="Número da página (começando em 1)."),
    limit: Optional[int] = Query(None, ge=1, le=100, description="Número de itens por página."),
    current_user: models.Usuario = Depends(get_current_user)
):
    # OTIMIZAÇÃO CRÍTICA: Projeção Parcial de Colunas para Listagem Rápida
    COLUNAS_SELECIONADAS = [
        models.DadoIA.id,
        models.DadoIA.documento_id,
        models.DadoIA.produto,
        models.DadoIA.quantidade,
        models.DadoIA.responsavel,
        models.DadoIA.data_criacao,
        models.DadoIA.data_finalizacao,
        models.DadoIA.status,
        models.DadoIA.falhas_json,
        models.DadoIA.falha,
        models.DadoIA.setor,
        models.DadoIA.localizacao_componente,
        models.DadoIA.lado_placa,
        models.DadoIA.observacao_producao,
        models.DadoIA.observacao_assistencia,
        models.DadoIA.resultado_ia,
        models.DadoIA.responsavel_assistencia
    ]
    
    # Query base para contagem e aplicação de filtros
    query_count = db.query(models.DadoIA)

    # Filtro de status
    if status:
        query_count = query_count.filter(models.DadoIA.status == status.upper())

    # Filtro de pesquisa
    if search:
        search_term = search.strip().upper()
        search_number = search_term[2:] if search_term.startswith("NC") else search_term

        try:
            search_number_int = int(search_number)
            number_filter = cast(models.DadoIA.id, String).ilike(f"%{search_number_int}%")
        except ValueError:
            number_filter = None

        filters = [models.DadoIA.documento_id.ilike(f"%{search_term}%")]
        if number_filter is not None:
            filters.append(number_filter)

        query_count = query_count.filter(or_(*filters))
    
    # Contagem total de resultados (aplicada após todos os filtros)
    total_count = query_count.count()
    
    # ----------------------------------------------------
    # Criação da Query Final (com Projeção e Paginação)
    
    # Inicia a query final com projeção de colunas
    query_final = db.query(*COLUNAS_SELECIONADAS)
    
    # Reaplica os filtros
    if status:
        query_final = query_final.filter(models.DadoIA.status == status.upper())
    if search:

        query_final = query_final.filter(or_(*filters))

    # 1. Lógica para o Dashboard (Lista Completa)
    if page is None and limit is None and not status and not search:
        # Define um limite menor para o dashboard para garantir a velocidade
        dashboard_limit = 100 
        
        items_data = query_final.order_by(models.DadoIA.data_criacao.desc()).limit(dashboard_limit).all()
        # total_count será o len(items_data) neste caso
        
    # 2. Lógica de Paginação Padrão (Tabela)
    else:
        current_page = page if page is not None else 1
        current_limit = limit if limit is not None else 10
        skip = (current_page - 1) * current_limit
        
        # Ordenação e paginação
        items_data = query_final.order_by(models.DadoIA.data_criacao.desc()).offset(skip).limit(current_limit).all()
        # Se for paginação, o total_count é o valor real calculado acima
        
    # Mapeia os resultados da tupla para o Schema ChecklistResumo
    # CORREÇÃO CRÍTICA: Use o schema correto (ChecklistResumo)
    column_keys = [c.key for c in COLUNAS_SELECIONADAS]
    items = [schemas.ChecklistResumo.model_validate(dict(zip(column_keys, item))) for item in items_data]

    # Retorna o objeto paginado
    return {"items": items, "total_count": total_count}

# --- Endpoint: Buscar um Checklist Específico por Documento ID (GET) ---
@router.get("/checklists/{documento_id}", response_model=schemas.Checklist) 
def buscar_checklist(
    documento_id: str, 
    db: Session = Depends(get_db),
    current_user: models.Usuario = Depends(get_current_user)
):
    db_dado = db.query(models.DadoIA).filter(
        models.DadoIA.documento_id == documento_id
    ).first()
    
    if not db_dado:
        raise HTTPException(status_code=404, detail="Documento não encontrado.")
        
    return db_dado

# --- Endpoint: Edição de Checklist pela Assistência (PATCH) ---
@router.patch("/checklists/{documento_id}", response_model=schemas.Checklist)
def atualizar_checklist(
    documento_id: str, 
    dado_update: schemas.ChecklistUpdate, 
    db: Session = Depends(get_db),
    current_user: models.Usuario = Depends(assistencia_required) 
):
    # 1. Busca o documento
    db_dado = db.query(models.DadoIA).filter(
        models.DadoIA.documento_id == documento_id
    ).first()
    
    if not db_dado:
        raise HTTPException(status_code=404, detail="Documento não encontrado.")

    # 2. Prepara e aplica as mudanças
    update_data = dado_update.model_dump(exclude_unset=True) 
    is_now_complete = update_data.get('status') == 'COMPLETO' and db_dado.status != 'COMPLETO'
    
    for key, value in update_data.items():
        setattr(db_dado, key, value)
    
    # 3. Gerencia Data de Finalização e Responsável da Assistência
    if is_now_complete:
        db_dado.data_finalizacao = datetime.now() 
        db_dado.responsavel_assistencia = current_user.username 
    elif 'status' in update_data and update_data['status'] != 'COMPLETO':
        db_dado.data_finalizacao = None
        db_dado.responsavel_assistencia = None

    # 4. RE-RODA A IA (se COMPLETO e com dados necessários)
    if db_dado.status == 'COMPLETO' and db_dado.falha and db_dado.setor:
        dados_atuais = {
            "produto": db_dado.produto, 
            "falha": db_dado.falha,
            "localizacao_componente": db_dado.localizacao_componente,
            "lado_placa": db_dado.lado_placa,
            "setor": db_dado.setor,
            "quantidade": db_dado.quantidade,
        }
        # A IA retorna uma string JSON, que é salva diretamente no campo resultado_ia
        db_dado.resultado_ia = analisar_checklist(dados_atuais)
    
    # 5. Salva no banco e retorna
    db.commit()
    db.refresh(db_dado)
    
    return db_dado


# --- Endpoint: Re-Rodar ou Obter a Previsão de IA (GET) ---
@router.get("/checklists/{documento_id}/analise-ia", response_model=Dict)
def obter_analise_ia(
    documento_id: str,
    db: Session = Depends(get_db),
    current_user: models.Usuario = Depends(get_current_user)
):
    """
    Retorna a última análise de IA (resultado_ia) para o documento.
    Se o status for COMPLETO, garante que a IA seja rodada (analisar_checklist)
    antes de retornar o resultado, atualizando o cache no BD.
    """
    db_dado = db.query(models.DadoIA).filter(
        models.DadoIA.documento_id == documento_id
    ).first()
    
    if not db_dado:
        raise HTTPException(status_code=404, detail="Documento não encontrado.")

    # 1. Checa se precisa rodar a IA (apenas se COMPLETO e com campos chave preenchidos)
    if db_dado.status == 'COMPLETO' and db_dado.falha and db_dado.setor:
        
        # Cria o dicionário de entrada com todos os campos necessários para analisar_checklist
        dados_atuais = {
            "produto": db_dado.produto, 
            "falha": db_dado.falha,
            "localizacao_componente": db_dado.localizacao_componente,
            "lado_placa": db_dado.lado_placa,
            "setor": db_dado.setor,
            "quantidade": db_dado.quantidade,
        }
        
        # A função 'analisar_checklist' retorna uma string JSON
        try:
            # Roda a IA
            json_string = analisar_checklist(dados_atuais)
            
            # Se a IA rodou, salva o novo resultado no BD (atualiza o cache)
            db_dado.resultado_ia = json_string
            db.commit()
            
            # Retorna o resultado parseado
            return json.loads(json_string)

        except Exception as e:
            # Em caso de falha da IA, tenta retornar o resultado anterior
            if db_dado.resultado_ia:
                return json.loads(db_dado.resultado_ia)
            
            # Se não houver resultado anterior, lança um erro 500
            raise HTTPException(status_code=500, detail=f"Falha ao rodar o modelo de IA: {e}")
    
    # 2. Se o resultado_ia já existir (mas o status não foi COMPLETO AGORA), apenas o retorna
    if db_dado.resultado_ia:
        return json.loads(db_dado.resultado_ia)

    # 3. Se estiver pendente, ou faltarem dados, retorna um aviso
    return {"status": "PENDENTE", "mensagem": "Análise de IA só está disponível para checklists com status 'COMPLETO' e dados preenchidos."}