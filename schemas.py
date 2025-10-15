from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, List, Dict, Any
from datetime import datetime, date

class UserBase(BaseModel):
    name: str 
    username: str
    role: str = Field(..., pattern=r'^(admin|assistencia|producao)$') 

class UserCreate(UserBase):
    password: str

class UserUpdate(BaseModel):
    name: Optional[str] = None
    password: Optional[str] = Field(None, min_length=6) 
    role: Optional[str] = Field(None, pattern=r'^(admin|assistencia|producao)$')
    model_config = ConfigDict(from_attributes=True)

class User(UserBase):
    id: int
    is_active: bool = True
    
    model_config = ConfigDict(from_attributes=True)

class Token(BaseModel):
    access_token: str
    token_type: str
    role: str
    name: str 

class TokenData(BaseModel):
    username: Optional[str] = None

class ChecklistBase(BaseModel):
    produto: str
    quantidade: int
    observacao_producao: Optional[str] = None
    observacao_assistencia: Optional[str] = None
    vai_para_assistencia: bool = False

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

class ChecklistUpdate(BaseModel):
    falha: Optional[str] = None
    localizacao_componente: Optional[str] = None
    lado_placa: Optional[str] = None
    setor: Optional[str] = None
    quantidade: Optional[int] = None 
    status: Optional[str] = None
    observacao_producao: Optional[str] = None 
    observacao_assistencia: Optional[str] = None 
    responsavel_assistencia: Optional[str] = None
    falhas_json: Optional[str] = None

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

class ChecklistResumo(BaseModel):
    id: int
    documento_id: str
    produto: Optional[str] = None
    quantidade: Optional[int] = None
    responsavel: Optional[str] = None
    data_criacao: Optional[datetime] = None
    status: Optional[str] = None
    falha: Optional[str] = None
    setor: Optional[str] = None
    falhas_json: Optional[str] = None
    observacao_producao: Optional[str] = None
    observacao_assistencia: Optional[str] = None

    responsavel_assistencia: Optional[str] = None 
    data_finalizacao: Optional[datetime] = None 
    resultado_ia: Optional[str] = None 

    model_config = ConfigDict(from_attributes=True)

class PaginatedChecklists(BaseModel):
    """Schema para retorno paginado de listas de checklists."""
    items: List[ChecklistResumo] 
    total_count: int

class AnalysisQuery(BaseModel):
    query: str

class Tip(BaseModel):
    title: str
    detail: str

class ChartData(BaseModel):
    title: str
    labels: List[str]
    datasets: List[Dict[str, Any]]
    chart_type: str = Field('bar', description="Tipo de gráfico (ex: 'bar', 'line', 'pie')")


class AnalysisResponse(BaseModel):
    query: str
    summary: str
    visualization_data: List[ChartData] = Field(default_factory=list)
    tips: List[Tip]

class ProducaoBase(BaseModel):
    """
    Schema base para dados de produção.
    """
    data_registro: date = Field(..., description="Data do registro de produção (AAAA-MM-DD).")

class ProducaoCreate(ProducaoBase):
    """
    Campos de entrada para criar um registro de produção.
    """

    quantidade_diaria: Optional[int] = Field(None, ge=0, description="Quantidade de placas produzidas no dia.")
    quantidade_mensal: Optional[int] = Field(None, ge=0, description="Quantidade de placas produzidas no mês.")
    
    tipo_registro: str = Field(..., max_length=1, description="Tipo de registro: 'M' (Mensal) ou 'D' (Diário).")
    
    observacao_mensal: Optional[str] = None
    observacao_diaria: Optional[str] = None
    
    responsavel: str = Field(..., description="Nome do usuário responsável pelo registro.")

class ProducaoUpdate(BaseModel):
    """
    Campos opcionais para atualizar um registro existente.
    """
    quantidade_diaria: Optional[int] = Field(None, ge=0)
    quantidade_mensal: Optional[int] = Field(None, ge=0)
    observacao_mensal: Optional[str] = None
    observacao_diaria: Optional[str] = None

class Producao(ProducaoBase):
    """
    Schema de saída que inclui o ID, o responsável e as observações.
    """
    id: int
    tipo_registro: str 
    quantidade_diaria: Optional[int] = None
    quantidade_mensal: Optional[int] = None
    observacao_mensal: Optional[str] = None
    observacao_diaria: Optional[str] = None

    responsavel: str

    model_config = ConfigDict(from_attributes=True)