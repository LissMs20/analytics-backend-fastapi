# train_models.py
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression 
import joblib
import json
import numpy as np

# --- Configurações de Arquivo ---
MODEL_FILE = 'checklist_predictor_model.joblib'
CLASSES_FILE = 'checklist_classes.json'
# --------------------------------

def create_initial_training_data():
    """
    Cria um DataFrame inicial de treinamento com base nos dados fornecidos pelo usuário.
    Estes dados simulam os registros 'COMPLETO' do banco de dados.
    """
    # Dados Categóricos Reais Fornecidos
    
    # Produtos (Simplificados para a parte do nome que não é o código PXXXX)
    produtos_origem = [
        'PLACA MONTADA (SMD + PTH) 7370V21 RTM-07 MK 220VCA V01',
        'PLACA MONTADA (SMD + PTH) 7311V22 TCS MULTIESCALA',
        'PLACA MONTADA (SMD + PTH) 7313V21 RST/RTT/MTP - MK 220VCA',
        'PLACA MONTADA (SMD + PTH) 7412V21 REP 01-03 MKC 24VCA',
        'PLACA MONTADA (SMD + PTH) 7344V23 TEI TEMPO FIXO',
    ]
    
    setores = [
        'SMT', 'Revisão - Sylmara', 'Revisão - Cryslainy', 
        'Proteção 1', 'Assistência', 'Tempo', 'Nível'
    ]
    
    falhas = [
        'Curto de solda', 'Solda fria', 'Falha de solda', 
        'Defeito no componente', 'Componente incorreto', 
        'Componente faltando', 'Trilha rompida', 'Falha de gravação', 
        'Sem defeito' 
    ]
    
    lado_placa = ['TOP', 'BOTTOM']
    
    # Localizações Comuns (Exemplos Genéricos, já que são relativos)
    localizacoes = ['U1', 'C10', 'R5', 'J1', 'Pino 3', 'LED D1', 'Conector P4']

    # --- Geração de Dados Fictícios com Relações Lógicas ---
    data = []
    np.random.seed(42)
    
    # 500 registros simulando as correlações mais prováveis
    for _ in range(500):
        prod = np.random.choice(produtos_origem)
        setor = np.random.choice(setores)
        lugar = np.random.choice(localizacoes)
        lado = np.random.choice(lado_placa)
        
        # Lógica de Correlação: Exemplo de falhas mais comuns em certos setores/produtos
        if setor == 'SMT' or setor == 'Revisão - Sylmara':
            # Solda e Curto são mais comuns
            falha = np.random.choice(['Curto de solda', 'Falha de solda', 'Solda fria', 'Componente faltando'], p=[0.3, 0.4, 0.2, 0.1])
        elif setor == 'Assistência':
            # Defeito componente, Trilha Rompida e Sem Defeito são mais comuns
            falha = np.random.choice(['Defeito no componente', 'Trilha rompida', 'Sem defeito'], p=[0.4, 0.2, 0.4])
        elif setor in ['Tempo', 'Nível']:
             # Falha de gravação ou Solda
            falha = np.random.choice(['Falha de gravação', 'Falha de solda', 'Defeito no componente'], p=[0.5, 0.3, 0.2])
        else:
            falha = np.random.choice(falhas)
            
        data.append({
            'produto': prod,
            'quantidade': np.random.randint(20, 300),
            'setor': setor,
            'localizacao_componente': lugar,
            'lado_placa': lado,
            'falha': falha 
        })
        
    return pd.DataFrame(data)

def train_and_save_checklist_model():
    """Treina o modelo Scikit-learn (LogisticRegression) e o salva no disco."""
    df = create_initial_training_data()
    
    X = df.drop('falha', axis=1)
    y = df['falha']
    
    numerical_features = ['quantidade']
    categorical_features = ['produto', 'setor', 'localizacao_componente', 'lado_placa']
    
    # Pipelines de Pré-processamento
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')) 
    ])
    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ])
    
    # Pipeline Final
    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1500, random_state=42)) 
    ])
    
    # Treinamento
    model_pipeline.fit(X, y)
    
    # 💾 Salva o modelo e as classes
    joblib.dump(model_pipeline, MODEL_FILE)
    
    with open(CLASSES_FILE, 'w') as f:
        # Garante que as classes salvas correspondam às classes do modelo treinado
        json.dump(list(model_pipeline.named_steps['classifier'].classes_), f)
        
    print(f"Modelo salvo como '{MODEL_FILE}'. Classes: {model_pipeline.named_steps['classifier'].classes_}")

if __name__ == '__main__':
    train_and_save_checklist_model()
    # Lembre-se: Execute este arquivo com: python train_models.py