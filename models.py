# models.py

from sqlalchemy import Column, Integer, String, Float, DateTime, func, Date, ForeignKey, UniqueConstraint, JSON # <-- Adicione JSON aqui
from sqlalchemy.orm import relationship
from database import Base

class Usuario(Base):
    __tablename__ = "usuarios"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    role = Column(String, default='producao') 

class DadoIA(Base):
    __tablename__ = "dado_ia"

    # Campos de Identificação e Fluxo
    id = Column(Integer, primary_key=True, index=True)
    documento_id = Column(String, unique=True, index=True)
    data_criacao = Column(DateTime, default=func.now(), index=True)
    responsavel = Column(String, nullable=False) # Responsável pela CRIAÇÃO
    
    # NOVO CAMPO: Data de Finalização/Revisão
    data_finalizacao = Column(DateTime, nullable=True) # Será preenchido na Assistência

    # NOVO CAMPO: Responsável pela Assistência
    responsavel_assistencia = Column(String, nullable=True) # Usuário logado que finalizou

    status = Column(String, default='COMPLETO', index=True) # COMPLETO ou PENDENTE

    # Campos do Checklist (Inputs do Usuário)
    produto = Column(String)
    quantidade = Column(Integer)

    observacao_producao = Column(String, nullable=True) 

    # Campos Detalhados (Opcionais na Criação, Obrigatórios na Assistência)
    falha = Column(String, nullable=True) 

    observacao_assistencia = Column(String, nullable=True)

    # RENOMEADO: 'localizacao' -> 'localizacao_componente'
    localizacao_componente = Column(String, nullable=True) 
    # RENOMEADO: 'lado' -> 'lado_placa'
    lado_placa = Column(String, nullable=True) 
    setor = Column(String, nullable=True)
    observacao = Column(String, nullable=True) 

    # Campo de Resultado da IA
    resultado_ia = Column(String, nullable=True)

    falhas_json = Column(JSON, nullable=True)

class RegistroProducao(Base):
    __tablename__ = "registros_producao"
    
    # 💡 CORREÇÃO 1: Definir a restrição de unicidade composta
    __table_args__ = (
        UniqueConstraint('data_registro', 'tipo_registro', name='uq_data_tipo_registro'),
    )

    id = Column(Integer, primary_key=True, index=True)
    
    # 💡 CORREÇÃO 2: Remover unique=True daqui
    data_registro = Column(Date, index=True, nullable=False) 
    
    # 💡 ADIÇÃO 1: Coluna de Tipo de Registro
    # 'M' (Mensal) ou 'D' (Diário). Defina o max_length apropriado.
    tipo_registro = Column(String(1), nullable=False) 
    
    quantidade_diaria = Column(Integer, nullable=True) 
    quantidade_mensal = Column(Integer, nullable=False) 
    
    # ADIÇÃO: Campos de Observação (mantidos)
    observacao_mensal = Column(String, nullable=True)
    observacao_diaria = Column(String, nullable=True)
    
    responsavel = Column(String, nullable=False)