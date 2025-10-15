from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from typing import List
import schemas
import models
from database import get_db 
from auth import (
    create_access_token, 
    verify_password, 
    get_password_hash, 
    get_user,
    get_current_user,
    admin_required 
)

router = APIRouter()

@router.get("/users/", response_model=List[schemas.User]) 
def list_users(
    db: Session = Depends(get_db), 
    _: models.Usuario = Depends(admin_required)
):
    users = db.query(models.Usuario).all()
    return users

@router.post("/users/", response_model=schemas.User, status_code=status.HTTP_201_CREATED)
def create_user(
    user: schemas.UserCreate, 
    db: Session = Depends(get_db),
    _: models.Usuario = Depends(admin_required) 
):
    db_user = get_user(db, username=user.username)
    if db_user:
        raise HTTPException(status_code=400, detail="Usuário já registrado.")
    
    hashed_password = get_password_hash(user.password)
    db_user = models.Usuario(
        username=user.username, 
        hashed_password=hashed_password, 
        role=user.role
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@router.patch("/users/{user_id}", response_model=schemas.User)
def update_user(
    user_id: int, 
    user_data: schemas.UserUpdate, 
    db: Session = Depends(get_db), 
    _: models.Usuario = Depends(admin_required)
):
    db_user = db.query(models.Usuario).filter(models.Usuario.id == user_id).first()
    if db_user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Usuário não encontrado")

    if user_data.role is not None:
        db_user.role = user_data.role
    
    if user_data.password:
        db_user.hashed_password = get_password_hash(user_data.password)
        
    db.commit()
    db.refresh(db_user)
    return db_user

@router.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_user(
    user_id: int, 
    db: Session = Depends(get_db), 
    current_user: models.Usuario = Depends(get_current_user)
):
    admin_required(current_user)

    if user_id == current_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Você não pode excluir sua própria conta enquanto está logado."
        )
        
    user_to_delete = db.query(models.Usuario).filter(models.Usuario.id == user_id).first()
    if user_to_delete is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Usuário não encontrado")
        
    db.delete(user_to_delete)
    db.commit()
    return 

@router.post("/token", response_model=schemas.Token)
def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(), 
    db: Session = Depends(get_db)
):
    user = get_user(db, form_data.username)
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Nome de usuário ou senha incorretos",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(
        data={"sub": user.username, "role": user.role}
    )
    return {"access_token": access_token, "token_type": "bearer", "role": user.role} 

@router.get("/users/me/", response_model=schemas.User)
def read_users_me(current_user: models.Usuario = Depends(get_current_user)):
    return current_user