# routers/producao.py

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from datetime import date

import schemas
import models
from database import get_db 
from auth import (
    get_current_user,
    admin_required
)

router = APIRouter(
    prefix="/producao",
    tags=["Produção"],
)

# ----------------------------------------------------------------------
# ROTA DE CRIAÇÃO (AGORA REQUER APENAS ADMIN)
# ----------------------------------------------------------------------

@router.post("/", response_model=schemas.Producao, status_code=status.HTTP_201_CREATED)
def create_producao_registro(
    registro: schemas.ProducaoCreate,
    db: Session = Depends(get_db),
    current_user: models.Usuario = Depends(admin_required)
):
    # Verifica se já existe registro da mesma data e tipo
    db_registro = db.query(models.RegistroProducao).filter(
        models.RegistroProducao.data_registro == registro.data_registro,
        models.RegistroProducao.tipo_registro == registro.tipo_registro
    ).first()

    if db_registro:
        tipo = "mensal" if registro.tipo_registro == 'M' else "diário"
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Já existe um registro {tipo} para a data: {registro.data_registro}"
        )

    if registro.tipo_registro == 'D':
        # Se for diário, define quantidade_diaria e zera mensal
        quantidade_diaria = registro.quantidade_diaria or 0
        quantidade_mensal = 0
    elif registro.tipo_registro == 'M':
        # Se for mensal, define quantidade_mensal e zera diária
        quantidade_mensal = registro.quantidade_mensal or 0
        quantidade_diaria = 0
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tipo de registro inválido. Use 'D' (Diário) ou 'M' (Mensal)."
        )

    db_data = models.RegistroProducao(
        data_registro=registro.data_registro,
        tipo_registro=registro.tipo_registro,
        quantidade_diaria=quantidade_diaria,
        quantidade_mensal=quantidade_mensal,
        observacao_mensal=registro.observacao_mensal,
        observacao_diaria=registro.observacao_diaria,
        responsavel=current_user.username
    )

    db.add(db_data)
    db.commit()
    db.refresh(db_data)
    return db_data

# ----------------------------------------------------------------------
# ROTA DE LEITURA/LISTAGEM (Mantida aberta para usuários logados)
# ----------------------------------------------------------------------

@router.get("/", response_model=List[schemas.Producao])
def list_producao_registros(
    tipo: str | None = None,
    data: date | None = None,
    db: Session = Depends(get_db),
    _: models.Usuario = Depends(get_current_user)
):
    query = db.query(models.RegistroProducao)
    if tipo:
        query = query.filter(models.RegistroProducao.tipo_registro == tipo)
    if data:
        query = query.filter(models.RegistroProducao.data_registro == data)
    return query.order_by(models.RegistroProducao.data_registro.desc()).all()


# ----------------------------------------------------------------------
# ROTA DE ATUALIZAÇÃO (PATCH) (AGORA REQUER APENAS ADMIN)
# ----------------------------------------------------------------------

@router.patch("/{registro_id}", response_model=schemas.Producao)
def update_producao_registro(
    registro_id: int,
    update_data: schemas.ProducaoUpdate,
    db: Session = Depends(get_db),
    # 💡 MUDANÇA: Apenas Admin pode atualizar a produção
    _: models.Usuario = Depends(admin_required)
):
    db_registro = db.query(models.RegistroProducao).filter(
        models.RegistroProducao.id == registro_id
    ).first()

    if not db_registro:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Registro não encontrado")

    # Atualiza apenas os campos fornecidos
    update_data_dict = update_data.model_dump(exclude_unset=True)

    for key, value in update_data_dict.items():
        setattr(db_registro, key, value)

    db.commit()
    db.refresh(db_registro)
    return db_registro

# ----------------------------------------------------------------------
# ROTA DE DELEÇÃO (JÁ REQUERIA ADMIN)
# ----------------------------------------------------------------------

@router.delete("/{registro_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_producao_registro(
    registro_id: int,
    db: Session = Depends(get_db),
    # Mantido: Deleção restrita apenas a Admin
    _: models.Usuario = Depends(admin_required) 
):
    db_registro = db.query(models.RegistroProducao).filter(
        models.RegistroProducao.id == registro_id
    ).first()

    if not db_registro:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Registro não encontrado")

    db.delete(db_registro)
    db.commit()
    return