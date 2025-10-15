# Arquivo: routers/user.py

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

import models 
import schemas
from database import get_db
import auth 

import auth

router = APIRouter(
    prefix="/users",
    tags=["Usuários e Autenticação"]
)

@router.post("/", response_model=schemas.User, status_code=status.HTTP_201_CREATED)
def create_user(
    user: schemas.UserCreate, 
    db: Session = Depends(get_db)
):
    db_user = auth.get_user(db, username=user.username)
    if db_user:
        raise HTTPException(status_code=400, detail="Nome de usuário já registrado")

    if not user.name or not user.name.strip():
        raise HTTPException(status_code=400, detail="O campo Nome não pode ser vazio.")

    hashed_password = auth.get_password_hash(user.password)

    new_user = models.Usuario(
        name=user.name.strip(),              
        username=user.username,
        hashed_password=hashed_password,
        role=user.role
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    return new_user

@router.get("/", response_model=List[schemas.User])
def read_users(
    db: Session = Depends(get_db), 
    current_user: models.Usuario = Depends(auth.admin_required)
):
    users = db.query(models.Usuario).all()
    return users

@router.patch("/{user_id}", response_model=schemas.User)
def update_user(
    user_id: int, 
    user_data: schemas.UserUpdate,
    db: Session = Depends(get_db), 
    current_user: models.Usuario = Depends(auth.admin_required)
):
    db_user = db.query(models.Usuario).filter(models.Usuario.id == user_id).first()
    if db_user is None:
        raise HTTPException(status_code=404, detail="Usuário não encontrado")

    update_data = user_data.model_dump(exclude_unset=True)

    if 'password' in update_data and update_data['password']:
        update_data['hashed_password'] = auth.get_password_hash(update_data.pop('password'))

    for key, value in update_data.items():
        setattr(db_user, key, value)
        
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_user(
    user_id: int, 
    db: Session = Depends(get_db),
    current_user: models.Usuario = Depends(auth.admin_required)
):
    db_user = db.query(models.Usuario).filter(models.Usuario.id == user_id).first()
    
    if db_user is None:
        raise HTTPException(status_code=404, detail="Usuário não encontrado")

    if db_user.id == current_user.id:
        raise HTTPException(status_code=400, detail="Você não pode excluir sua própria conta enquanto estiver logado.")

    db.delete(db_user)
    db.commit()

    return {}