from fastapi import APIRouter, Depends, HTTPException, status, Query, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Optional, Dict
from datetime import datetime
import json
from sqlalchemy import or_, cast, String
from sqlalchemy.exc import SQLAlchemyError
import traceback

import schemas
import models
from database import get_db, SessionLocal
from auth import get_current_user, assistencia_required
from services.ia_core import analisar_checklist, analisar_checklist_multifalha # 💡 Importado o multifalha

router = APIRouter()

def run_ia_in_background(dado_ia_id: int, falhas_a_analisar: List[dict]):
    """Função síncrona que executa a IA e atualiza o campo resultado_ia em uma nova sessão."""
    
    # 💡 Lógica corrigida para usar analisar_checklist_multifalha
    # O run_ia_in_background agora é para a análise do checklist, que usa a IA legada.
    
    # Prepara os dados para a IA (adiciona campos globais)
    lista_para_ia = []
    
    with SessionLocal() as db_in_thread:
        db_dado = db_in_thread.query(models.DadoIA).filter(models.DadoIA.id == dado_ia_id).first()
        
        if not db_dado:
            print(f"Erro: DadoIA com ID {dado_ia_id} não encontrado na thread de IA.")
            return

        # Prepara a lista de falhas com dados globais do checklist para a IA
        # Esta lógica está replicada do api_handlers.py, mas é necessária aqui
        produto_global = db_dado.produto
        quantidade_global = db_dado.quantidade
        obs_prod_global = db_dado.observacao_producao or ""
        obs_assist_global = db_dado.observacao_assistencia or ""
        
        for falha_data in falhas_a_analisar:
            dados_para_ia = {
                "produto": produto_global,
                "quantidade": quantidade_global,
                "observacao_producao": obs_prod_global,
                "observacao_assistencia": obs_assist_global,
                **falha_data 
            }
            lista_para_ia.append(dados_para_ia)


        try:
            # 💡 Chama o processador de múltiplos itens
            resultados = analisar_checklist_multifalha(lista_para_ia) 
            
            json_string = json.dumps(resultados, ensure_ascii=False)
            
            db_dado.resultado_ia = json_string
            db_in_thread.commit()
            db_in_thread.refresh(db_dado)
            print(f"✅ Análise de IA concluída para {db_dado.documento_id}.")
            
        except Exception as e:
            db_in_thread.rollback() 
            print(f"❌ Erro na análise de IA em background para {db_dado.documento_id}: {e}")
            traceback.print_exc()

@router.post("/checklists/", response_model=schemas.Checklist) 
def criar_checklist(
    dado: dict,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: models.Usuario = Depends(get_current_user)
):
    try:
        print("📥 Dados recebidos:", dado)
        vai_para_assistencia = dado.get("vai_para_assistencia", False)

        status_inicial = 'PENDENTE' if vai_para_assistencia else 'COMPLETO'
        data_finalizacao_inicial = datetime.now() if status_inicial == 'COMPLETO' else None
        responsavel_assistencia_inicial = None

        falhas_lista = dado.get("falhas", [])
        is_multi = "falhas" in dado and isinstance(dado["falhas"], list)

        if is_multi:
            db_dado = models.DadoIA(
                produto=dado["produto"],
                quantidade=dado["quantidade"],
                observacao_producao=dado.get("observacao_producao"),
                responsavel=dado["responsavel"],
                falhas_json=json.dumps(falhas_lista, ensure_ascii=False) if falhas_lista else None,
                status=status_inicial,
                resultado_ia=None,
                documento_id="NC-TEMP",
                data_finalizacao=data_finalizacao_inicial,
                responsavel_assistencia=responsavel_assistencia_inicial,
            )
        else:
            dado_schema = schemas.ChecklistCreate(**dado) 
            
            dados_para_db = dado_schema.model_dump(
                exclude={'vai_para_assistencia', 'falha', 'setor', 'localizacao_componente', 'lado_placa'}
            )

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
                falha=dado_schema.falha or None,
                setor=dado_schema.setor or None,
                localizacao_componente=dado_schema.localizacao_componente or None,
                lado_placa=dado_schema.lado_placa or None,
                falhas_json=None,
                resultado_ia=None,
                status=status_inicial,
                documento_id="NC-TEMP",
                data_finalizacao=data_finalizacao_inicial,
                responsavel_assistencia=responsavel_assistencia_inicial,
            )

        print("💾 Salvando no banco...")
        db.add(db_dado)
        db.commit()
        db.refresh(db_dado)

        db_dado.documento_id = f"NC{db_dado.id:05d}"
        db.commit()
        db.refresh(db_dado)

        # 💡 Prepara a lista de falhas com os dados globais ANTES de mandar para o background
        lista_para_bg_ia = []
        if status_inicial == "COMPLETO" and falhas_lista:
            produto_global = db_dado.produto
            quantidade_global = db_dado.quantidade
            obs_prod_global = db_dado.observacao_producao or ""
            obs_assist_global = db_dado.observacao_assistencia or ""
            
            for falha_data in falhas_lista:
                dados_para_ia = {
                    "produto": produto_global,
                    "quantidade": quantidade_global,
                    "observacao_producao": obs_prod_global,
                    "observacao_assistencia": obs_assist_global,
                    **falha_data 
                }
                lista_para_bg_ia.append(dados_para_ia)

        print(f"✅ Checklist criado: {db_dado.documento_id}")
        if status_inicial == "COMPLETO" and lista_para_bg_ia:
            print("🧠 Enviando IA em background...")
            background_tasks.add_task(run_ia_in_background, db_dado.id, lista_para_bg_ia)

        return db_dado

    except KeyError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Campo obrigatório ausente: {e.args[0]}"
        )
    except Exception as e:
        if "value is not a valid" in str(e) or "missing required field" in str(e):
             raise HTTPException(status_code=422, detail=f"Erro de validação de dados: {str(e)}")

        if isinstance(e, SQLAlchemyError):
            db.rollback()
            print("❌ Erro SQLAlchemy:", str(e))
            raise HTTPException(
                status_code=500,
                detail="Erro ao salvar checklist no banco de dados."
            )

        db.rollback()
        print("🔥 Erro inesperado:", e)
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Erro interno ao criar checklist: {str(e)}"
        )

# --- FUNÇÕES DE LISTAGEM E BUSCA (INALTERADAS) ---

@router.get("/checklists/", response_model=schemas.PaginatedChecklists)
def listar_dados(
    db: Session = Depends(get_db),
    status: Optional[str] = Query(None, description="Filtrar por status: COMPLETO ou PENDENTE."),
    search: Optional[str] = Query(None, description="Pesquisar por parte do Documento ID ou número interno."),
    page: Optional[int] = Query(None, ge=1, description="Número da página (começando em 1)."),
    limit: Optional[int] = Query(None, ge=1, le=100, description="Número de itens por página."),
    current_user: models.Usuario = Depends(get_current_user)
):
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

    query_count = db.query(models.DadoIA)

    if status:
        query_count = query_count.filter(models.DadoIA.status == status.upper())

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

    total_count = query_count.count()

    query_final = db.query(*COLUNAS_SELECIONADAS)

    if status:
        query_final = query_final.filter(models.DadoIA.status == status.upper())
    if search:
        query_final = query_final.filter(or_(*filters))

    if page is None and limit is None and not status and not search:
        dashboard_limit = 100 
        
        items_data = query_final.order_by(models.DadoIA.data_criacao.desc()).limit(dashboard_limit).all()

    else:
        current_page = page if page is not None else 1
        current_limit = limit if limit is not None else 10
        skip = (current_page - 1) * current_limit

        items_data = query_final.order_by(models.DadoIA.data_criacao.desc()).offset(skip).limit(current_limit).all()

    column_keys = [c.key for c in COLUNAS_SELECIONADAS]
    items = []
    
    for item_tuple in items_data:
        row = dict(zip(column_keys, item_tuple))
        
        if isinstance(row.get("falhas_json"), list) or isinstance(row.get("falhas_json"), dict):
            row["falhas_json"] = json.dumps(row["falhas_json"], ensure_ascii=False)
            
        items.append(schemas.ChecklistResumo.model_validate(row))

    return {"items": items, "total_count": total_count}

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

# --- FUNÇÃO DE ATUALIZAÇÃO (INALTERADA) ---

@router.patch("/checklists/{documento_id}", response_model=schemas.Checklist)
def atualizar_checklist(
    documento_id: str, 
    dado_update: schemas.ChecklistUpdate, 
    db: Session = Depends(get_db),
    current_user: models.Usuario = Depends(assistencia_required) 
):

    db_dado = db.query(models.DadoIA).filter(
        models.DadoIA.documento_id == documento_id
    ).first()
    
    if not db_dado:
        raise HTTPException(status_code=404, detail="Documento não encontrado.")

    update_data = dado_update.model_dump(exclude_unset=True) 

    is_now_complete = update_data.get('status') == 'COMPLETO' and db_dado.status != 'COMPLETO'
    
    for key, value in update_data.items():
        setattr(db_dado, key, value)

    if is_now_complete:
        db_dado.data_finalizacao = datetime.now() 
        db_dado.responsavel_assistencia = current_user.username 
    elif 'status' in update_data and update_data['status'] != 'COMPLETO':
        db_dado.data_finalizacao = None
        db_dado.responsavel_assistencia = None

    if db_dado.status == 'COMPLETO' and db_dado.falha and db_dado.setor and not db_dado.falhas_json:
        # 💡 Esta parte está correta para análise SÍNCRONA de item único
        dados_atuais = {
            "produto": db_dado.produto, 
            "falha": db_dado.falha,
            "localizacao_componente": db_dado.localizacao_componente,
            "lado_placa": db_dado.lado_placa,
            "setor": db_dado.setor,
            "quantidade": db_dado.quantidade,
        }
        try:
            # 💡 ANALISAR_CHECKLIST está esperando UM dicionário de falha/setor
            db_dado.resultado_ia = json.dumps(analisar_checklist(dados_atuais), ensure_ascii=False)
        except Exception as e:
            print(f"Aviso: Falha ao rodar IA síncrona em PATCH para {documento_id}: {e}")

    try:
        db.commit()
        db.refresh(db_dado)
    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao salvar atualização no banco: {str(e)}"
        )
    
    return db_dado


# --- FUNÇÃO DE OBTENÇÃO DA ANÁLISE DE IA (CORRIGIDA) ---

@router.get("/checklists/{documento_id}/analise-ia", response_model=Dict)
def obter_analise_ia(
    documento_id: str,
    db: Session = Depends(get_db),
    current_user: models.Usuario = Depends(get_current_user)
):
    """
    Retorna a última análise de IA (resultado_ia) para o documento.
    Se o status for COMPLETO, garante que a IA seja rodada de forma síncrona
    antes de retornar o resultado, atualizando o cache no BD.
    """
    db_dado = db.query(models.DadoIA).filter(
        models.DadoIA.documento_id == documento_id
    ).first()
    
    if not db_dado:
        raise HTTPException(status_code=404, detail="Documento não encontrado.")

    # Se já tem resultado, retorna imediatamente (cache)
    if db_dado.resultado_ia:
        try:
            return json.loads(db_dado.resultado_ia)
        except json.JSONDecodeError:
            # Continua para re-análise em caso de JSON inválido
            pass 

    falhas_a_analisar = []
    
    # 1. Tenta carregar dados de falhas_json (modo multi)
    if db_dado.falhas_json:
        try:
            falhas_a_analisar = json.loads(db_dado.falhas_json) if isinstance(db_dado.falhas_json, str) else db_dado.falhas_json
            
            # Adiciona dados globais para cada falha na lista (necessário para a IA)
            produto_global = db_dado.produto
            quantidade_global = db_dado.quantidade
            obs_prod_global = db_dado.observacao_producao or ""
            obs_assist_global = db_dado.observacao_assistencia or ""
            
            falhas_para_ia = []
            for f in falhas_a_analisar:
                 falhas_para_ia.append({
                    "produto": produto_global,
                    "quantidade": quantidade_global,
                    "observacao_producao": obs_prod_global,
                    "observacao_assistencia": obs_assist_global,
                    **f
                })
            falhas_a_analisar = falhas_para_ia # Substitui pela lista enriquecida

        except json.JSONDecodeError:
            print(f"Erro ao decodificar falhas_json para {documento_id}")
            falhas_a_analisar = []

    # 2. Tenta carregar dados de campos únicos (modo single)
    elif db_dado.falha and db_dado.setor:
        falhas_a_analisar = [{
            "produto": db_dado.produto, 
            "falha": db_dado.falha,
            "localizacao_componente": db_dado.localizacao_componente,
            "lado_placa": db_dado.lado_placa,
            "setor": db_dado.setor,
            "quantidade": db_dado.quantidade,
            "observacao_producao": db_dado.observacao_producao,
            "observacao_assistencia": db_dado.observacao_assistencia,
        }]

    # 3. Executa a IA (Síncrona) se estiver COMPLETO e tiver dados
    if db_dado.status == 'COMPLETO' and falhas_a_analisar:
        
        try:
            # 💡 Chama a função correta
            resultados = analisar_checklist_multifalha(falhas_a_analisar) 
            json_string = json.dumps(resultados, ensure_ascii=False)

            # 💡 Salva o novo resultado no cache
            db_dado.resultado_ia = json_string
            db.commit()

            return json.loads(json_string)

        except Exception as e:
            print(f"Erro crítico na execução SÍNCRONA da IA para {documento_id}: {e}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Falha ao rodar o modelo de IA: {e}")

    # 4. Fallback: Se não tem análise e não pode ser analisado (não COMPLETO ou sem dados)
    return {"status": "PENDENTE", "mensagem": "Análise de IA só está disponível para checklists com status 'COMPLETO' e dados preenchidos. Requisitado sem cache."}
    
    # 💡 Linha desnecessária removida: return future.result()