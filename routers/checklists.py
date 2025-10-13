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
from services.ia_core import analisar_checklist

router = APIRouter()

# --- Função de Background: Executar a IA e Salvar o Resultado ---
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
                # Chama a IA para cada falha na lista
                resultado = analisar_checklist(f) 
                resultados.append(resultado)
                
            json_string = json.dumps(resultados, ensure_ascii=False)
            
            db_dado.resultado_ia = json_string
            db_in_thread.commit()
            db_in_thread.refresh(db_dado)
            print(f"✅ Análise de IA concluída para {db_dado.documento_id}.")
            
        except Exception as e:
            # Garante que qualquer erro seja revertido
            db_in_thread.rollback() 
            print(f"❌ Erro na análise de IA em background para {db_dado.documento_id}: {e}")
            traceback.print_exc()


# --- Endpoint: Adicionar um Checklist ao BD (POST) ---
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

        # Define status inicial
        status_inicial = 'PENDENTE' if vai_para_assistencia else 'COMPLETO'
        data_finalizacao_inicial = datetime.now() if status_inicial == 'COMPLETO' else None
        responsavel_assistencia_inicial = None

        # Lista de falhas (multi ou single)
        falhas_lista = dado.get("falhas", [])
        is_multi = "falhas" in dado and isinstance(dado["falhas"], list)

        # --- 1️⃣ Criar o objeto para salvar no BD ---
        if is_multi:
            # Modo Multi-Falha
            db_dado = models.DadoIA(
                produto=dado["produto"],
                quantidade=dado["quantidade"],
                observacao_producao=dado.get("observacao_producao"),
                responsavel=dado["responsavel"],
                # CORREÇÃO: Usa json.dumps para salvar a lista de dicts
                falhas_json=json.dumps(falhas_lista, ensure_ascii=False) if falhas_lista else None,
                status=status_inicial,
                resultado_ia=None,
                documento_id="NC-TEMP",
                data_finalizacao=data_finalizacao_inicial,
                responsavel_assistencia=responsavel_assistencia_inicial,
            )
        else:
            # Modo Single-Falha: valida com Pydantic
            # Validação Pydantic (lanca erro 422 se falhar)
            dado_schema = schemas.ChecklistCreate(**dado) 
            
            dados_para_db = dado_schema.model_dump(
                exclude={'vai_para_assistencia', 'falha', 'setor', 'localizacao_componente', 'lado_placa'}
            )

            # Se for COMPLETO e tiver dados de falha, cria a lista para a IA
            if status_inicial == 'COMPLETO' and dado_schema.falha:
                falha_unica = {
                    "falha": dado_schema.falha,
                    "setor": dado_schema.setor,
                    "localizacao_componente": dado_schema.localizacao_componente,
                    "lado_placa": dado_schema.lado_placa,
                    "observacao_producao": dado_schema.observacao_producao
                }
                falhas_lista.append(falha_unica) # Adiciona à lista que será usada pela IA

            db_dado = models.DadoIA(
                **dados_para_db,
                # Salva os campos da falha única diretamente no modelo
                falha=dado_schema.falha or None,
                setor=dado_schema.setor or None,
                localizacao_componente=dado_schema.localizacao_componente or None,
                lado_placa=dado_schema.lado_placa or None,
                # No modo single, falhas_json é None
                falhas_json=None,
                resultado_ia=None,
                status=status_inicial,
                documento_id="NC-TEMP",
                data_finalizacao=data_finalizacao_inicial,
                responsavel_assistencia=responsavel_assistencia_inicial,
            )

        # --- 2️⃣ Salvar no banco ---
        print("💾 Salvando no banco...")
        db.add(db_dado)
        db.commit()
        db.refresh(db_dado)

        # --- 3️⃣ Gerar documento_id ---
        db_dado.documento_id = f"NC{db_dado.id:05d}"
        db.commit()
        db.refresh(db_dado)
        print(f"✅ Checklist criado: {db_dado.documento_id}")

        # --- 4️⃣ Executar IA em background (se completo e com falhas) ---
        if status_inicial == "COMPLETO" and falhas_lista:
            print("🧠 Enviando IA em background...")
            # Envia a lista de falhas (que é a mesma em modo single ou multi)
            background_tasks.add_task(run_ia_in_background, db_dado.id, falhas_lista)

        return db_dado

    # 🧱 ERROS DE VALIDAÇÃO DE DADOS (Pydantic/KeyError)
    except KeyError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Campo obrigatório ausente: {e.args[0]}"
        )
    except Exception as e: # Captura o erro de validação do Pydantic que o FastAPI não capturaria
        if "value is not a valid" in str(e) or "missing required field" in str(e):
             raise HTTPException(status_code=422, detail=f"Erro de validação de dados: {str(e)}")
        
        # 🧱 ERROS DO BANCO
        if isinstance(e, SQLAlchemyError):
            db.rollback()
            print("❌ Erro SQLAlchemy:", str(e))
            raise HTTPException(
                status_code=500,
                detail="Erro ao salvar checklist no banco de dados."
            )
        
        # 🧱 ERROS GERAIS
        db.rollback()
        print("🔥 Erro inesperado:", e)
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Erro interno ao criar checklist: {str(e)}"
        )


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
    # 🚨 INÍCIO DA CORREÇÃO: Converte falhas_json de volta para string se necessário 🚨
    column_keys = [c.key for c in COLUNAS_SELECIONADAS]
    items = []
    
    for item_tuple in items_data:
        row = dict(zip(column_keys, item_tuple))
        
        # Se o SQLAlchemy retornou uma lista ou dict (o objeto Python),
        # converte de volta para uma string JSON para que o Pydantic aceite
        if isinstance(row.get("falhas_json"), list) or isinstance(row.get("falhas_json"), dict):
            row["falhas_json"] = json.dumps(row["falhas_json"], ensure_ascii=False)
            
        items.append(schemas.ChecklistResumo.model_validate(row))
    # 🚨 FIM DA CORREÇÃO 🚨

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
    
    # Flag para saber se o status MUDOU para COMPLETO
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

    # 4. RE-RODA A IA (se COMPLETO e com dados necessários, apenas para modo single)
    if db_dado.status == 'COMPLETO' and db_dado.falha and db_dado.setor and not db_dado.falhas_json:
        dados_atuais = {
            "produto": db_dado.produto, 
            "falha": db_dado.falha,
            "localizacao_componente": db_dado.localizacao_componente,
            "lado_placa": db_dado.lado_placa,
            "setor": db_dado.setor,
            "quantidade": db_dado.quantidade,
        }
        try:
             # A IA retorna uma string JSON, que é salva diretamente no campo resultado_ia
            db_dado.resultado_ia = analisar_checklist(dados_atuais)
        except Exception as e:
            print(f"Aviso: Falha ao rodar IA síncrona em PATCH para {documento_id}: {e}")
    
    # 5. Salva no banco e retorna
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


# --- Endpoint: Re-Rodar ou Obter a Previsão de IA (GET) ---
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
    Funciona tanto para modo single (falha, setor) quanto multi (falhas_json).
    """
    db_dado = db.query(models.DadoIA).filter(
        models.DadoIA.documento_id == documento_id
    ).first()
    
    if not db_dado:
        raise HTTPException(status_code=404, detail="Documento não encontrado.")

    # 1. Prepara a lista de falhas a analisar (single ou multi)
    falhas_a_analisar = []
    if db_dado.falhas_json:
        try:
            # O SQLAlchemy pode ter retornado como string ou lista/dict dependendo da query
            if isinstance(db_dado.falhas_json, str):
                falhas_a_analisar = json.loads(db_dado.falhas_json)
            else: # Se já for lista/dict (tipo JSON nativo do driver)
                falhas_a_analisar = db_dado.falhas_json
                
        except json.JSONDecodeError:
            print(f"Erro ao decodificar falhas_json para {documento_id}")
    elif db_dado.falha and db_dado.setor:
        falhas_a_analisar = [{
            "produto": db_dado.produto, 
            "falha": db_dado.falha,
            "localizacao_componente": db_dado.localizacao_componente,
            "lado_placa": db_dado.lado_placa,
            "setor": db_dado.setor,
            "quantidade": db_dado.quantidade,
        }]


    # 2. Checa se precisa rodar a IA (apenas se COMPLETO e com campos chave preenchidos)
    if db_dado.status == 'COMPLETO' and falhas_a_analisar:
        
        resultados = []
        try:
            # Roda a IA de forma síncrona para garantir o resultado imediato
            for f in falhas_a_analisar:
                resultado = analisar_checklist(f) 
                resultados.append(resultado)
            
            json_string = json.dumps(resultados, ensure_ascii=False)
            
            # Se a IA rodou, salva o novo resultado no BD (atualiza o cache)
            db_dado.resultado_ia = json_string
            db.commit()
            
            # Retorna o resultado parseado (pode ser lista de resultados)
            return json.loads(json_string)

        except Exception as e:
            # Em caso de falha da IA, tenta retornar o resultado anterior
            if db_dado.resultado_ia:
                print(f"Aviso: Falha na re-execução da IA para {documento_id}. Retornando cache.")
                return json.loads(db_dado.resultado_ia)
            
            # Se não houver resultado anterior, lança um erro 500
            print(f"Erro crítico na IA para {documento_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Falha ao rodar o modelo de IA: {e}")
    
    # 3. Se o resultado_ia já existir (e não precisou re-rodar), apenas o retorna
    if db_dado.resultado_ia:
        return json.loads(db_dado.resultado_ia)

    # 4. Se estiver pendente, ou faltarem dados, retorna um aviso
    return {"status": "PENDENTE", "mensagem": "Análise de IA só está disponível para checklists com status 'COMPLETO' e dados preenchidos."}