import json
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, date
import numpy as np
import re 
from dateutil.parser import parse, ParserError 
from functools import lru_cache

# ----------------------------------------------------
# DADOS DE REFERÊNCIA (Domínio - Usados por IA e Análise)
# ----------------------------------------------------

# (As constantes PLACAS_TEMPO, PLACAS_PROTECAO_1, etc. permanecem inalteradas)
# ... (Funções clean_product_name, classify_product_line, extract_period_and_date, safe_json_load permanecem inalteradas)

@lru_cache(maxsize=None)
def clean_product_name(product: str) -> str:
    """Remove o código P0000 e excesso de espaços do nome da placa."""
    if not isinstance(product, str):
        return ""
    # Remove código do produto (P seguido de 4 ou 5 dígitos) e espaços adjacentes
    return re.sub(r'P\d{4,5}\s+', '', product).strip()

# As listas longas de produtos foram mantidas aqui para o mapeamento
PLACAS_TEMPO = [
    'P2340 PLACA MONTADA (SMD + PTH) 7311V22 TCS MULTIESCALA - 01-03 MK 12VCA-VCC',
    'P2341 PLACA MONTADA (SMD + PTH) 7311V22 RTP 01 MK 24 A 240VCA - 24 A 240VCC',
    'P0939 PLACA MONTADA (SMD + PTH) 7348V2 TRD 2016 MODELO 01 MK 110VCA',
    'P0940 PLACA MONTADA (SMD + PTH) 7348V2 TRD 2016 MODELO 01 MK 220VCA',
    'P0941 PLACA MONTADA (SMD + PTH) 7348V2 TRD 2016 MODELO 01 MK 24VCA-VCC',
    'P0943 PLACA MONTADA (SMD + PTH) 7348V2 TRD 2016 MODELO 02 MK 125VCC',
    'P0945 PLACA MONTADA (SMD + PTH) 7348V2 TRD 2016 MODELO 02 MK 220VCA',
    'P0946 PLACA MONTADA (SMD + PTH) 7348V2 TRD 2016 MODELO 02 MK 24VCA-VCC',
    'P2313 PLACA MONTADA (SMD + PTH) 7311V22 TCS 02-04 MULTIESCALA MK 24 A 240VCA - 24 A 240VCC',
    'P2315 PLACA MONTADA (SMD + PTH) 7311V22 TCS 01-03 MULTIESCALA MK 24 A 240VCA - 24 A 240VCC - V01',
    'P2338 PLACA MONTADA (SMD + PTH) 7311V22 RDR MULTIESCALA - MK 24 A 240VCA/VCC',
    'P1041 PLACA MONTADA (SMD + PTH) 7311V21 RDR MULTIESCALA - MK 24 A 240VCA - 24 A 240VCC',
    'P1044 PLACA MONTADA (SMD + PTH) 7311V22 RVB - MK 24 A 240VCA - 24 A 240VCC - V02',
    'P2314 PLACA MONTADA (SMD + PTH) 7311V22 TMF-02 - MK 24 A 240VCA/VCC+ 12VCA/VCC',
    'P1051 PLACA MONTADA (SMD + PTH) 7335V21 RT 01 MKC 12VCA-VCC V03',
    'P1052 PLACA MONTADA (SMD + PTH) 7335V21 RT 01 MKC 24VCA-VCC V03',
    'P2185 PLACA MONTADA (SMD + PTH) 7335V21RT 01 MKC 380-440VCA  V03',
    'P2187 PLACA MONTADA (SMD + PTH) 7335V21RT 01 MKC 220-380VCA  V03',
    'P1053 PLACA MONTADA (SMD + PTH) 7335V21 RT 01 MKC 94 A 242VCA V03',
    'P1396 PLACA MONTADA (SMD + PTH) 7336V22 MULTICAMADAS RAX 02 MKC 24 A 240VCA',
    'P1397 PLACA MONTADA (SMD + PTH) 7336V22 MULTIESCALA RYD - MKC 24 A 240 VCA - VCC',
    'P1406 PLACA MONTADA (SMD + PTH) 7336V22 MULTIESCALA TEI 05 MKC 24 A 240 VCA - VCC',
    'P1398 PLACA MONTADA (SMD + PTH) 7336V22 MULTIESCALA TEI C/POT - MKC 24 A 240 VCA - VCC',
    'P1399 PLACA MONTADA (SMD + PTH) 7336V22 MULTIESCALA TEI 02-04 MKC 12VCA-VCC',
    'P1392 PLACA MONTADA (SMD + PTH) 7336V22 MULTIESCALA TEI 02-04 MKC 24 A 240 VCA - VCC',
    'P1377 PLACA MONTADA (SMD + PTH) 7344V23 RAX 01 MKC 24 A 240 VCA - VCC',
    'P2290 PLACA MONTADA (SMD + PTH) 7344V23 RBE 01-03 MKC 24 A 240VCA - 24 A 240VCC',
    'P1385 PLACA MONTADA (SMD + PTH) 7344V23 TEI MODELOS 01-03 MKC 12VCA-VCC',
    'P1387 PLACA MONTADA (SMD + PTH) 7344V23 MULTIESCALA TEI 01-03 / RPP MKC 24 A 240VCA - 24 A 240VCC V01',
    'P1375 PLACA MONTADA (SMD + PTH) 7344V23 TMF 01 MKC 24 A 240VCA - 24 A 240VCC V01',
    'P1381 PLACA MONTADA (SMD + PTH) 7344V23 TEI TEMPO FIXO - MKC 24 A 240VCA - 24 A 240VCC',
]
PLACAS_PROTECAO_1 = [
    'P1119 PLACA MONTADA (SMD + PTH) 7332V21 FFS 01 MKC 220-380VCA V03',
    'P1149 PLACA MONTADA (SMD + PTH) 7334V21 FSN - MK 110VCA',
    'P1150 PLACA MONTADA (SMD + PTH) 7334V21 FSN - MK 220VCA',
    'P1151 PLACA MONTADA (SMD + PTH) 7334V21 FSN - MK 380VCA V04',
    'P1152 PLACA MONTADA (SMD + PTH) 7334V21 FSN - MK 440VCA',
    'P1153 PLACA MONTADA (SMD + PTH) 7334V21 FSN - MK 480VCA',
    'P1156 PLACA MONTADA (SMD + PTH) 7334V31 FSN MULTICAMADAS - MK 220VCA',
    'P1157 PLACA MONTADA (SMD + PTH) 7334V31 FSN MULTICAMADAS - MK 380VCA',
    'P1158 PLACA MONTADA (SMD + PTH) 7334V31 FSN MULTICAMADAS - MK 440VCA',
    'P1336 PLACA MONTADA (SMD + PTH) 7506V2 PBM TRIFASICO FP/CA PADRAO - MK 220-380VCA',
    'P1338 PLACA MONTADA (SMD + PTH) 7506V2 PBM MONOFASICO CA PADRAO - 01 MK 220-380VCA',
]
PLACAS_PROTECAO_2 = [
    "P2443 PLACA MONTADA (SMD + PTH) 7370V21 RTM-07 MK 220VCA V01",
    "P1046 PLACA MONTADA (SMD + PTH) 7370V21 FIF 01 MK 220-440VCA - V03",
    "P1540 PLACA MONTADA (SMD + PTH) 7370V21 SST MK 480VCA",
    "P2525 PLACA MONTADA (SMD + PTH) 7370V22 RTM-07 MK 220VCA V0",
    "P1047 PLACA MONTADA (SMD + PTH) 7370V21 SST MK 110VCA",
    "P1048 PLACA MONTADA (SMD + PTH) 7370V21 SST MK 220VCA -",
    "P1049 PLACA MONTADA (SMD + PTH) 7370V21 SST MK 380VCA -",
    "P1050 PLACA MONTADA (SMD + PTH) 7370V21 SST MK 440VCA -",
    "P1078 PLACA MONTADA (SMD + PTH) 7314V21 RTC 12-14-16 MK 24VCC",
    "P1079 PLACA MONTADA (SMD + PTH) 7314V21 RTC 12-14-16 MK 48VCC",
    "P1080 PLACA MONTADA (SMD + PTH) 7314V21 RTC 12-14-16 MK 220VCC",
    "P1081 PLACA MONTADA (SMD + PTH) 7314V21 RTC 12-14-16 MK 250VCC",
    "P1082 PLACA MONTADA (SMD + PTH) 7314V21 RTC 12-14-16 MK 110/125VCC",
    "P1086 PLACA MONTADA (SMD + PTH) 7313V21 RST/RTT/MTP - MK 110VCA",
    "P1087 PLACA MONTADA (SMD + PTH) 7313V21 RST/RTT/MTP - MK 220VCA",
    "P1088 PLACA MONTADA (SMD + PTH) 7313V21 RST/RTT/MTP - MK 380VCA",
    "P1089 PLACA MONTADA (SMD + PTH) 7313V21 RST/RTT/MTP - MK 440VCA",
    "P1090 PLACA MONTADA (SMD + PTH) 7313V21 RST/RTT/MTP - MK 460VCA",
    "P1093 PLACA MONTADA (SMD + PTH) 7317V21 RCA MONOFASICO MODELOS 05-06 MM 220VCA -10A",
    "P1094 PLACA MONTADA (SMD + PTH) 7317V21 RCA MONOFASICO MODELOS 05-06 MK 110VCA -1A",
    "P1096 PLACA MONTADA (SMD + PTH) 7317V21 RCA MONOFASICO MODELOS 05-06 MK 110VCA - 10A",
    "P1097 PLACA MONTADA (SMD + PTH) 7317V21 RCA MONOFASICO MODELOS 05-06 MK 220VCA - 1A",
    "P1098 PLACA MONTADA (SMD + PTH) 7317V21 RCA MONOFASICO MODELOS 05-06 MK 220VCA - 5A",
    "P1099 PLACA MONTADA (SMD + PTH) 7317V21 RCA MONOFASICO MODELOS 03-04 MK 110VCA - 1A",
    "P2503 PLACA MONTADA (SMD + PTH) 7317V22 RCA MONOFASICO 01-02-03-04 MK 220VCA -10A",
    "P1106 PLACA MONTADA (SMD + PTH) 7317V21 RCA MONOFASICO 01-02-03-04 MK 110VCA - 5A",
    "P1108 PLACA MONTADA (SMD + PTH) 7317V22 RCA MONOFASICO 01-02-03-04 MK 220VCA -1A",
    "P1109 PLACA MONTADA (SMD + PTH) 7317V21 RCA MONOFASICO 01-02-03-04 MK 220VCA - 5A",
    "P1115 PLACA MONTADA (SMD + PTH) 7317V21 RCA TRIFASICO MODELOS 30-31 MK 220VCA -5A",
    "P1160 PLACA MONTADA (SMD + PTH) 7313V21 FIF/RSF - MK 110VCA",
    "P1161 PLACA MONTADA (SMD + PTH) 7313V21 FIF/RSF - MK 220VCA",
    "P1162 PLACA MONTADA (SMD + PTH) 7313V21 FIF/RSF - MK 380VCA",
    "P1163 PLACA MONTADA (SMD + PTH) 7313V21 FIF/RSF - MK 440VCA",
    "P1164 PLACA MONTADA (SMD + PTH) 7313V21 FIF/RSF - MK 480VCA",
    "P1168 PLACA MONTADA (SMD + PTH) 7313V21 RTM - MK 110VCA",
    "P1169 PLACA MONTADA (SMD + PTH) 7313V21 RTM - MK 220VCA",
    "P1170 PLACA MONTADA (SMD + PTH) 7313V21 RTM - MK 380VCA",
    "P1171 PLACA MONTADA (SMD + PTH) 7313V21 RTM - MK 460VCA",
    "P1172 PLACA MONTADA (SMD + PTH) 7313V21 RTM - MK 480VCA",
    "P1173 PLACA MONTADA (SMD + PTH) 7313V21 RTM - MK 254VCA",
    "P1174 PLACA MONTADA (SMD + PTH) 7313V21 RTM - MK 440VCA",
    "P1175 PLACA MONTADA (SMD + PTH) 7313V21 RTI - MK 110VCA - 48VCC",
    "P1180 PLACA MONTADA (SMD + PTH) 7313V21 RTI - MK 220VCA - 9VCC/12VCC",
    "P1182 PLACA MONTADA (SMD + PTH) 7313V21 RMV - MK 380-440VCA",
    "P1183 PLACA MONTADA (SMD + PTH) 7313V21 RMV - MK 220-380VCA",
    "P1192 PLACA MONTADA (SMD + PTH) 7317V21 RCC MONOFASICO MODELOS 05-06 MK 220VCC - 10A",
    "P1204 PLACA MONTADA (SMD + PTH) 7317V21 RCC MONOFASICO 01-02-03-04 MK 110VCA - 5A",
    "P1232 PLACA MONTADA (SMD + PTH) 7370V21 RTM-04 MK 220VCA V03"
]
PLACAS_NIVEL = [
    "P1066 PLACA MONTADA (SMD + PTH) 7412V21 REL 01-03 MKC 24VCC",
    "P1067 PLACA MONTADA (SMD + PTH) 7412V21 REL 01-03 MKC 24VCA",
    "P1068 PLACA MONTADA (SMD + PTH) 7412V21 REL 01-03 MKC 110VCA",
    "P1070 PLACA MONTADA (SMD + PTH) 7412V21 REL 01-03 MKC 220/380VCA V04",
    "P1071 PLACA MONTADA (SMD + PTH) 7412V21 REL 01-03 MKC 440VCA",
    "P1072 PLACA MONTADA (SMD + PTH) 7412V21 REL 01-03 MKC 254VCA",
    "P1074 PLACA MONTADA (SMD + PTH) 7412V21 REP 01-03 MKC 24VCA",
    "P1075 PLACA MONTADA (SMD + PTH) 7412V21 REP 01-03 MKC 110VCA",
    "P1077 PLACA MONTADA (SMD + PTH) 7412V21 REP 01-03 MKC 220/380VCA",
    "P1120 PLACA MONTADA (SMD + PTH) 7330V21 MULT. CNS 01 MK 24VCA",
    "P1122 PLACA MONTADA (SMD + PTH) 7330V21 MULT. CNS 01 MK 220VCA",
    "P1125 PLACA MONTADA (SMD + PTH) 7330V21 MULT. RDN 01 MK 24VCA",
    "P1127 PLACA MONTADA (SMD + PTH) 7330V21 MULT. RDN 01 MK 220VCA",
    "P1128 PLACA MONTADA (SMD + PTH) 7330V21 MULT. RDN 01 MK 380VCA",
    "P1129 PLACA MONTADA (SMD + PTH) 7330V21 MULT. RES 01 MK 24VCC",
    "P1130 PLACA MONTADA (SMD + PTH) 7330V21 MULT. RES 01 MK 24VCA",
    "P1132 PLACA MONTADA (SMD + PTH) 7330V21 MULT. RES 01 MK 220VCA",
    "P1133 PLACA MONTADA (SMD + PTH) 7331V21 REL/RN - 01-02-03 MKC 254VCA",
    "P1139 PLACA MONTADA (SMD + PTH) 7331V21 RN 01-02-03 MKC 220-380VCA V03",
    "P1141 PLACA MONTADA (SMD + PTH) 7346V21 RNF MODELOS 01-03 MK 220VCA",
    "P1142 PLACA MONTADA (SMD + PTH) 7346V21 RNF MODELOS 01-03 MK 380VCA V03",
    "P1143 PLACA MONTADA (SMD + PTH) 7346V21 RNF MODELOS 01-03 MK 440VCA",
    "P1167 PLACA MONTADA (SMD + PTH) MULTICAMADAS 7412V31 REL 01-03 MK 220-380VCA"
]

# SETs para busca rápida O(1)
PLACAS_TEMPO_SET = set([clean_product_name(p) for p in PLACAS_TEMPO])
PLACAS_PROTECAO_1_SET = set([clean_product_name(p) for p in PLACAS_PROTECAO_1])
PLACAS_PROTECAO_2_SET = set([clean_product_name(p) for p in PLACAS_PROTECAO_2])
PLACAS_NIVEL_SET = set([clean_product_name(p) for p in PLACAS_NIVEL])

@lru_cache(maxsize=None)
def classify_product_line(product_name: str) -> str:
    """Classifica a placa na linha de produção (Tempo, Proteção 1, Proteção 2, Nível ou Outros) usando SETs."""
    clean_name = clean_product_name(product_name)
    if clean_name in PLACAS_TEMPO_SET:
        return 'Tempo'
    if clean_name in PLACAS_PROTECAO_1_SET:
        return 'Proteção 1'
    if clean_name in PLACAS_PROTECAO_2_SET:
        return 'Proteção 2'
    if clean_name in PLACAS_NIVEL_SET:
        return 'Nível'
    return 'Outros'

# Mapeamento de Falha Bruta para Causa Raiz de Processo (Lógica de Domínio)
CAUSA_RAIZ_MAP = {
    'Curto de solda': 'Falha no Processo (Máquina de Solda/Revisão)',
    'Solda fria': 'Falha no Processo (Máquina de Solda/Pallet/Revisão)',
    'Falha de solda': 'Falha no Processo (Máquina de Solda/Pallet/Revisão)',
    'Defeito no componente': 'Componente (Qualidade do Fornecedor)',
    'Componente incorreto': 'Erro Humano (Linha de Montagem PTH/Revisão SMT)',
    'Componente faltando': 'Erro Humano (Linha de Montagem PTH) ou Falha de SMT',
    'Trilha rompida': 'Dano Físico (Acidente/Ajuste/Queima)',
    'Falha de gravação': 'Falha de Equipamento (Máquina de Gravação/Setup)',
    'Sem defeito': 'Desvio de Fluxo (Placa enviada incorretamente)',
    'Outros': 'Causa Indeterminada'
}

# ----------------------------------------------------
# FUNÇÃO DE REFINAMENTO DE CAUSA RAIZ (NOVA LÓGICA)
# ----------------------------------------------------

def refine_causa_raiz_smt(row: pd.Series) -> str:
    falha = str(row['falha_individual']).lower()
    setor_detecao = str(row['setor_falha_individual']).lower()
    causa_basica = str(row['causa_raiz_processo'])
    
    # 1. Refinamento para Curto ou Falha de Solda
    if 'solda' in falha or 'curto' in falha:
        # Se for Curto/Falha de Solda e detectado em SMT (ou IV)
        if 'smt' in setor_detecao or 'iv' in setor_detecao:
            if 'curto de solda' in falha:
                # Curto é quase sempre problema de PASTA/PRINTER/STENCIL
                return 'Falha Crítica SMT (Pasta/Stencil/Printer)' # <- MELHOR CLASSIFICAÇÃO
            elif 'falha de solda' in falha or 'solda fria' in falha:
                # Falha ou Solda Fria em SMT é mais ligada ao perfil do REFLOW ou a P&P desalinhada
                return 'Falha Processo SMT (Máquina Reflow/Perfil/P&P)' # <- CLASSIFICAÇÃO MAIS COMPLETA

    # 2. Refinamento para Componente Faltando/Desalinhado em SMT (Mantido)
    if ('componente faltando' in falha or 'desalinhado' in falha) and 'smt' in setor_detecao:
        return 'Falha Processo SMT (Máquina Pick & Place/Setup)'

    # Caso contrário, retorna a classificação básica
    return causa_basica

# ----------------------------------------------------
# FUNÇÕES DE PRÉ-PROCESSAMENTO E UTILS
# ----------------------------------------------------

def extract_period_and_date(query: str) -> Tuple[str, Optional[datetime], str]:
    """Extrai o nível de granularidade (D, M, Y, W, G) e uma data específica da query."""
    query_lower = query.lower()
    
    # 1. Extração do Período/Granularidade
    if 'diaria' in query_lower or 'dia' in query_lower or 'dias' in query_lower or 'hoje' in query_lower:
        period = 'D'
        granularity_name = 'Diária'
    elif 'mensal' in query_lower or 'mes' in query_lower or 'mês' in query_lower or 'este mes' in query_lower:
        period = 'M'
        granularity_name = 'Mensal'
    elif 'anual' in query_lower or 'ano' in query_lower or 'este ano' in query_lower:
        period = 'Y'
        granularity_name = 'Anual'
    elif 'semanal' in query_lower or 'semana' in query_lower:
        period = 'W'
        granularity_name = 'Semanal'
    else:
        period = 'G' # Geral
        granularity_name = 'Geral'

    # 2. Extração de Data Específica
    date_match = re.search(
        r'(\d{1,2}[/\-\\]\d{1,2}(?:[/\-\\]\d{2,4})?)|\b(janeiro|fevereiro|março|abril|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro)\b', 
        query_lower
    )
    
    parsed_date = None
    if date_match:
        date_str = date_match.group(0)
        try:
            # Garante que, se for apenas um mês, o ano seja o atual
            if re.match(r'^\b(janeiro|fevereiro|março|abril|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro)\b', date_str):
                date_str = f"{date_str} {datetime.now().year}"

            parsed_date = parse(date_str, fuzzy=True, dayfirst=True)
        except ParserError:
            pass 

    return period, parsed_date, granularity_name

def safe_json_load(value: Any) -> Any:
    """Utilitário para carregar JSON de forma segura."""
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value

# ====================================================================================================
# Função flatten_nested robusta
# ====================================================================================================
def flatten_nested(x):
    """Garante que um valor aninhado (list, ndarray, Series, etc.) vire uma string simples."""
    # Se for uma mini-Series (caso raro mas ocorre em merges/explodes)
    if isinstance(x, pd.Series):
        x = x.tolist()

    # Se for lista, array ou lista de listas → achata e junta tudo em string
    if isinstance(x, (list, np.ndarray)):
        # np.ravel achata qualquer dimensão (ex: [[1, 2], [3]] vira [1, 2, 3])
        return ' '.join(map(str, np.ravel(x)))

    # Se for NaN ou None → retorna string vazia
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ''

    # Caso geral → converte pra string simples
    return str(x)
# ====================================================================================================

def flatten_multi_failure_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Desanuvia (flatten) o campo 'falhas_json' em várias linhas, 
    criando um DataFrame onde cada linha é uma falha individual.
    """
    if 'falhas_json' not in df.columns:
        return df

    # 1. Aplica safe_json_load e prepara a coluna
    df['falhas_processadas'] = df['falhas_json'].apply(safe_json_load).apply(lambda x: x if isinstance(x, list) else [])
    
    # 2. Explode a lista de falhas
    df_flattened = df.explode('falhas_processadas')
    df_flattened = df_flattened.dropna(subset=['falhas_processadas'])

    if df_flattened.empty:
        return pd.DataFrame()

    # [PONTO 2] CORREÇÃO CRÍTICA: Reseta o índice para garantir unicidade após o .explode()
    df_flattened = df_flattened.reset_index(drop=True) 
    
    # 3. Normaliza as colunas aninhadas (falha, setor, etc.)
    falha_cols = pd.json_normalize(df_flattened['falhas_processadas'])
    
    falha_cols = falha_cols.rename(columns={
        'falha': 'falha_individual',
        'setor': 'setor_falha_individual',
        'localizacao_componente': 'localizacao_componente_individual',
        'lado_placa': 'lado_placa_individual'
    })

    # [PONTO 3] CORREÇÃO CRÍTICA: Limpa o índice do DataFrame de colunas normalizadas para concatenação segura
    falha_cols = falha_cols.reset_index(drop=True)

    # 4. Combina o DF original com as novas colunas
    df_final = df_flattened.drop(columns=['falhas_json', 'falhas_processadas', 'falha', 'setor'], errors='ignore')

    # Garantir que falha_cols e df_final tenham índices RangeIndex idênticos
    falha_cols.index = df_final.index

    df_final = pd.concat([df_final, falha_cols], axis=1)

    # 5. Adiciona a causa raiz básica
    df_final['causa_raiz_processo'] = df_final['falha_individual'].fillna('').apply(lambda x: CAUSA_RAIZ_MAP.get(x, 'Causa Indeterminada'))

    # 6. ✅ NOVO PASSO: Refina a causa raiz com lógica SMT/Setor
    df_final['causa_raiz_detalhada'] = df_final.apply(refine_causa_raiz_smt, axis=1)
    
    return df_final.reset_index(drop=True)

def prepare_dataframe(data: List[Dict], flatten_multifalha: bool = True) -> pd.DataFrame:
    """
    Centraliza o pré-processamento de dados de Checklist e Produção.
    CRÍTICO: Define o flatten como TRUE por padrão para análise.
    """
    df = pd.DataFrame(data)
    if df.empty:
        return df

    # [PONTO 1] Garante índice limpo e único
    df = df.reset_index(drop=True)
    df.index = pd.RangeIndex(len(df))

    # ====================================================================================================
    # 💡 Dica extra implementada: Limpa colunas duplicadas para evitar mini-Series na origem do problema
    # ====================================================================================================
    df = df.loc[:, ~df.columns.duplicated()]

    # ----------------------------
    # Pré-processamento inicial
    # ----------------------------
    if 'documento_id' not in df.columns:
        df['documento_id'] = df.get('id', pd.Series(range(len(df)))) 
    
    # Conversão de datas
    df['data_registro'] = pd.to_datetime(df.get('data_registro', df.get('data_finalizacao')), errors='coerce')
    df['data_finalizacao'] = pd.to_datetime(df.get('data_finalizacao'), errors='coerce')

    # Numéricos com preenchimento seguro
    for c in ['quantidade', 'quantidade_produzida', 'quantidade_diaria']:
        # Verifica se a coluna existe. Se não, cria uma série de zeros.
        if c not in df.columns:
            df[c] = 0
            
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(int)
    # ----------------------------
    # Flatten multifalha
    # ----------------------------
    if flatten_multifalha:
        df = flatten_multi_failure_data(df)
        if df.empty:
            return df

        # 1) Indice e colunas limpas
        df = df.reset_index(drop=True)
        df = df.loc[:, ~df.columns.duplicated()].copy()
        df.index = pd.RangeIndex(len(df))

        # (Logs de debug removidos para concisão, mas mantidos no código original se necessário)

        # 2) Função robusta para achatar valores aninhados (handling Series, list, ndarray, DataFrame)
        def flatten_nested_local(x): # Função renomeada para evitar conflito com a cacheada global
            if isinstance(x, pd.Series):
                x = x.tolist()
            if isinstance(x, pd.DataFrame):
                try:
                    return ' '.join(map(str, np.ravel(x.values)))
                except Exception:
                    return str(x)
            if isinstance(x, (list, tuple, np.ndarray)):
                return ' '.join(map(str, np.ravel(x)))
            if x is None or (isinstance(x, float) and pd.isna(x)):
                return ''
            return str(x)

        # 3) Garante que as colunas existam e sejam Series 1-D com strings
        def get_col_as_str_series(name):
            if name in df.columns:
                s = df[name].apply(flatten_nested_local)
            else:
                s = pd.Series([''] * len(df), index=df.index)
            
            vals = list(s)
            # Trunca ou completa para garantir mesmo length do df
            if len(vals) > len(df):
                vals = vals[:len(df)]
            elif len(vals) < len(df):
                vals = vals + [''] * (len(df) - len(vals))
            return pd.Series(vals, index=df.index).astype(str)

        obs_prod = get_col_as_str_series('observacao_producao')
        obs_ass = get_col_as_str_series('observacao_assistencia')

        # 4) Finalmente, cria a coluna combinada de forma segura
        df['observacao_combinada'] = (obs_prod.fillna('') + ' ' + obs_ass.fillna('')).str.strip()

        # Garantia final: remover qualquer coluna duplicada residual e reset index
        df = df.loc[:, ~df.columns.duplicated()].reset_index(drop=True)

    # ----------------------------
    # Features de Domínio 
    # ----------------------------
    df['linha_produto'] = df.get('produto', pd.Series('', index=df.index)).fillna('').apply(classify_product_line)

    # Se a função de flatten não rodou, criamos a coluna de causa raiz básica
    if 'causa_raiz_processo' not in df.columns and 'falha' in df.columns:
        df['causa_raiz_processo'] = df['falha'].fillna('').apply(lambda x: CAUSA_RAIZ_MAP.get(x, 'Causa Indeterminada'))

    # Se o flatten rodou, já temos 'causa_raiz_detalhada'. Se não, a detalhada é igual à básica
    if 'causa_raiz_detalhada' not in df.columns:
        df['causa_raiz_detalhada'] = df.get('causa_raiz_processo', 'Causa Indeterminada')


    # ----------------------------
    # Métricas (DPPM)
    # ----------------------------
    if (df['quantidade_produzida'] > 0).any():
        df['dppm_registro'] = (df['quantidade'] * 1_000_000) / df['quantidade_produzida']
    elif (df['quantidade_diaria'] > 0).any():
        df['dppm_registro'] = (df['quantidade'] * 1_000_000) / df['quantidade_diaria']
    else:
        df['dppm_registro'] = 0.0

    df = df.reset_index(drop=True)
    return df