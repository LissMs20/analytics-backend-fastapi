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

# ----------------------------------------------------------------------
# ROTAS DE GERENCIAMENTO DE USUÁRIOS (APENAS ADMIN)
# ----------------------------------------------------------------------

# Rota para LISTAR todos os usuários (GET /users/)
@router.get("/users/", response_model=List[schemas.User]) 
def list_users(
    db: Session = Depends(get_db), 
    _: models.Usuario = Depends(admin_required)
):
    users = db.query(models.Usuario).all()
    return users


# Rota de criação de usuário (POST /users/)
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

# Rota para ATUALIZAR um usuário (PATCH /users/{user_id})
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
    
    # Atualiza o Role, se fornecido
    if user_data.role is not None:
        db_user.role = user_data.role
    
    # Atualiza a senha, se fornecida
    if user_data.password:
        db_user.hashed_password = get_password_hash(user_data.password)
        
    db.commit()
    db.refresh(db_user)
    return db_user


# Rota para DELETAR um usuário (DELETE /users/{user_id})
@router.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_user(
    user_id: int, 
    db: Session = Depends(get_db), 
    current_user: models.Usuario = Depends(get_current_user)
):
    # Garante que o usuário logado é um admin antes de prosseguir
    admin_required(current_user)
    
    # Impede que o admin logado exclua a si mesmo
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
    # Retorna o status 204 (No Content) após exclusão bem-sucedida
    return 

# ----------------------------------------------------------------------
# ROTAS DE AUTENTICAÇÃO (MANTIDAS)
# ----------------------------------------------------------------------

# Rota de Login (Retorna o token)
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
    # Retorna o role para uso imediato no frontend
    return {"access_token": access_token, "token_type": "bearer", "role": user.role} 

# Rota para obter o usuário logado
@router.get("/users/me/", response_model=schemas.User)
def read_users_me(current_user: models.Usuario = Depends(get_current_user)):
    return current_user