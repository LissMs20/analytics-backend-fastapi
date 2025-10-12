from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, List, Dict, Any # Importamos Dict e Any para os datasets
from datetime import datetime, date # 💡 Importamos 'date' para a produção

# --------------------------------------------------
# SCHEMAS DE AUTENTICAÇÃO E USUÁRIO (MANTIDOS)
# --------------------------------------------------

# Esquema base para UserCreate e UserUpdate
class UserBase(BaseModel):
    # Garante que o role seja um dos valores permitidos e define o padrão se necessário
    name: str 
    username: str
    role: str = Field(..., pattern=r'^(admin|assistencia|producao)$') 

# Esquema para criação de usuário (Admin/Inicial)
class UserCreate(UserBase):
    password: str

# 💡 NOVO: Esquema para atualização de usuário (senha e role são opcionais)
class UserUpdate(BaseModel):
    name: Optional[str] = None
    password: Optional[str] = Field(None, min_length=6) 
    role: Optional[str] = Field(None, pattern=r'^(admin|assistencia|producao)$')
    model_config = ConfigDict(from_attributes=True)

# 💡 ATUALIZADO: Esquema de saída para o usuário (inclui ID para listagem)
class User(UserBase):
    id: int # Adicionado o ID
    is_active: bool = True # Adicionado para completar o modelo do DB
    
    model_config = ConfigDict(from_attributes=True)

# Esquema para o Login
class Token(BaseModel):
    access_token: str
    token_type: str
    role: str
    name: str 

class TokenData(BaseModel):
    username: Optional[str] = None

# --------------------------------------------------
# SCHEMAS DE CHECKLIST (MANTIDOS)
# --------------------------------------------------

# 1. Esquema BASE (Campos Mínimos na Criação)
class ChecklistBase(BaseModel):
    produto: str
    quantidade: int
    observacao_producao: Optional[str] = None
    observacao_assistencia: Optional[str] = None
    vai_para_assistencia: bool = False

# 2. Esquema de CRIAÇÃO (Checklist Completo/Inicial)
class ChecklistCreate(ChecklistBase):
    responsavel: str 
    falha: Optional[str] = None
    localizacao_componente: Optional[str] = None 
    lado_placa: Optional[str] = None 
    setor: Optional[str] = None
    observacao: Optional[str] = None

class Falha(BaseModel):
    falha: str
    setor: Optional[str] = None
    localizacao_componente: Optional[str] = None
    lado_placa: Optional[str] = None
    observacao_producao: Optional[str] = None

class ChecklistCreateMulti(ChecklistBase):
    responsavel: str
    falhas: List[Falha]

# 3. Esquema de EDIÇÃO (Usado pela Assistência Técnica)
class ChecklistUpdate(BaseModel):
    falha: Optional[str] = None
    localizacao_componente: Optional[str] = None
    lado_placa: Optional[str] = None
    setor: Optional[str] = None
    quantidade: Optional[int] = None 
    status: Optional[str] = None
    observacao_producao: Optional[str] = None 
    observacao_assistencia: Optional[str] = None 
    # Adicionado para ser usado na finalização de assistência
    responsavel_assistencia: Optional[str] = None
    falhas_json: Optional[str] = None

# 4. Esquema de SAÍDA (Retorno da API) - Usado no GET por ID
class Checklist(ChecklistCreate):
    id: int
    documento_id: str
    data_criacao: datetime
    data_finalizacao: Optional[datetime] = None
    responsavel_assistencia: Optional[str] = None
    status: str
    resultado_ia: Optional[str] = None
    falhas_json: Optional[str] = None 
    
    model_config = ConfigDict(from_attributes=True) 

# 5. Esquema de RESUMO - Usado no GET de Listagem (listar_dados)
class ChecklistResumo(BaseModel):
    id: int
    documento_id: str
    produto: Optional[str] = None
    quantidade: Optional[int] = None
    responsavel: Optional[str] = None
    data_criacao: Optional[datetime] = None
    status: Optional[str] = None
    
    # Campos que devem ser incluídos para visualização rápida e Assistência
    falha: Optional[str] = None
    setor: Optional[str] = None
    falhas_json: Optional[str] = None
    observacao_producao: Optional[str] = None
    observacao_assistencia: Optional[str] = None

    responsavel_assistencia: Optional[str] = None 
    data_finalizacao: Optional[datetime] = None 
    resultado_ia: Optional[str] = None 

    model_config = ConfigDict(from_attributes=True)


# 6. Esquema de SAÍDA PAGINADA
class PaginatedChecklists(BaseModel):
    """Schema para retorno paginado de listas de checklists."""
    items: List[ChecklistResumo] 
    total_count: int

# --------------------------------------------------
# SCHEMAS DE ANÁLISE DA IA (CORRIGIDOS PARA MULTI-GRÁFICO)
# --------------------------------------------------

class AnalysisQuery(BaseModel):
    query: str

class Tip(BaseModel):
    title: str
    detail: str

# 💡 NOVO: Esquema para estruturar os dados de um único gráfico
class ChartData(BaseModel):
    title: str
    labels: List[str]
    datasets: List[Dict[str, Any]]
    chart_type: str = Field('bar', description="Tipo de gráfico (ex: 'bar', 'line', 'pie')")


class AnalysisResponse(BaseModel):
    query: str
    summary: str
    # ✅ CORREÇÃO CRÍTICA: Agora aceita uma LISTA de gráficos
    visualization_data: List[ChartData] = Field(default_factory=list)
    tips: List[Tip]

# --------------------------------------------------
# SCHEMAS DE REGISTRO DE PRODUÇÃO (MANTIDOS)
# --------------------------------------------------

class ProducaoBase(BaseModel):
    """
    Schema base para dados de produção.
    """
    # Usamos date em vez de datetime para simplificar a entrada (AAAA-MM-DD)
    data_registro: date = Field(..., description="Data do registro de produção (AAAA-MM-DD).")

class ProducaoCreate(ProducaoBase):
    """
    Campos de entrada para criar um registro de produção.
    """
    
    # ⭐️ CORREÇÃO: Torne ambos opcionais para evitar erros de validação Pydantic
    # quando apenas um campo é fornecido (e para aceitar 'None' antes de ser tratado)
    quantidade_diaria: Optional[int] = Field(None, ge=0, description="Quantidade de placas produzidas no dia.")
    quantidade_mensal: Optional[int] = Field(None, ge=0, description="Quantidade de placas produzidas no mês.")
    
    tipo_registro: str = Field(..., max_length=1, description="Tipo de registro: 'M' (Mensal) ou 'D' (Diário).")
    
    # 💡 ADIÇÃO DOS CAMPOS DE OBSERVAÇÃO QUE FALTAVAM AQUI
    observacao_mensal: Optional[str] = None
    observacao_diaria: Optional[str] = None
    
    responsavel: str = Field(..., description="Nome do usuário responsável pelo registro.")

# Esquema para atualização (PATCH) de registro de produção
class ProducaoUpdate(BaseModel):
    """
    Campos opcionais para atualizar um registro existente.
    """
    quantidade_diaria: Optional[int] = Field(None, ge=0)
    quantidade_mensal: Optional[int] = Field(None, ge=0)
    
    # 💡 ADIÇÃO: Observações
    observacao_mensal: Optional[str] = None
    observacao_diaria: Optional[str] = None

# Esquema de saída (Resposta da API)
class Producao(ProducaoBase):
    """
    Schema de saída que inclui o ID, o responsável e as observações.
    """
    id: int
    tipo_registro: str 
    quantidade_diaria: Optional[int] = None
    quantidade_mensal: Optional[int] = None # Corrigido para ser Optional, pois depende do tipo_registro

    observacao_mensal: Optional[str] = None
    observacao_diaria: Optional[str] = None

    responsavel: str

    model_config = ConfigDict(from_attributes=True)