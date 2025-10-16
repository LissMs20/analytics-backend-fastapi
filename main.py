# main.py

import os
from dotenv import load_dotenv

load_dotenv() 

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional 
import models
from database import engine
from routers import auth, checklists, analysis, producao, user, auth_routes

try:
    # A nova tabela RegistroProducao será criada aqui
    models.Base.metadata.create_all(bind=engine)
    # AQUI DEVERÁ APARECER "Cliente Gemini inicializado com sucesso."
    print("Tabelas do banco de dados criadas/verificadas com sucesso.")
except Exception as e:
    print(f"ERRO CRÍTICO ao inicializar o banco de dados: {e}") 
    pass 

app = FastAPI(
    title="Analytics AI Backend",
    description="API de Gestão de Checklists de Produção e Análise de IA."
)

# ----------------------------------------------------
# CONFIGURAÇÃO DO CORS (INALTERADA)
# ----------------------------------------------------

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8000",
    "https://analytics-frontend-react.vercel.app", 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------
# INCLUSÃO DOS ROUTERS
# ----------------------------------------------------

# Inclui os roteadores com prefixos e tags
app.include_router(auth_routes.router, tags=["Autenticação"], prefix="/api")
app.include_router(user.router, prefix="/api")
app.include_router(checklists.router, tags=["Checklists"], prefix="/api")
app.include_router(analysis.router, tags=["Análise de IA"], prefix="/api")
app.include_router(producao.router, tags=["Produção"], prefix="/api")


# Rota de Status da API
@app.get("/")
def read_root():
    return {"status": "ok", "message": "API Rodando com Sucesso! Versão 1.0"}