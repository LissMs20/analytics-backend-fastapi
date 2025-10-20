# train_intent_model.py (Versão Robusta com Automação de Setores)
import joblib
import sys
import os
import numpy as np

# Ajusta o caminho para importar o IntentClassifier do diretório services
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'services')))

try:
    # Importa a classe otimizada
    from intent_classifier import IntentClassifier 
except ImportError as e:
    print(f"Erro ao importar IntentClassifier: {e}")
    print("Certifique-se de que o script 'intent_classifier.py' está no diretório 'services'.")
    sys.exit(1)


def train_and_save_intent_model():
    """
    Treina o IntentClassifier com um conjunto de dados expandido e robusto e salva.
    """
    print("--- 🧠 INICIANDO TREINAMENTO DO MODELO LOCAL DE INTENÇÃO (NLP) ---")

    # 1. Conjunto de Dados de Treinamento BASE (manualmente curado)
    texts = [
        # QUALITY
        "qual a taxa de rejeição semanal?", "mostre o dppm do último mês", 
        "tendência de falhas no último trimestre", "o dppm está bom?", 
        "taxa de refugo por dia", "aumentou o refugo?", "qualidade geral",
        
        # CAUSAS
        "quais são as 5 principais causas raiz de falha?", "pareto das falhas", 
        "qual o produto com maior incidência de falhas?", "qual o principal defeito", 
        "analisar as causas do produto P7370", "causa raiz principal",
        
        # INDIVIDUAL
        "qual o operador com maior volume de defeitos?", "desempenho da máquina 420", 
        "ranking de performance da linha de produção", "performance individual do ID 999", 
        "falha por máquina", "desempenho operadores",
        
        # NLP
        "análise de tópicos das observações do processo", 
        "o que as pessoas estão escrevendo nos comentários?", 
        "resumir o texto livre das anotações", 
        "qual o assunto predominante nos registros de falha?", 
        
        # SMT_FOCO (Mantido separado de 'sector' pois é um tópico específico da Indústria)
        "análise de tendência de falhas de solda", "problemas com a máquina de SMT",
        "auditoria na pasta de solda", "falhas associadas ao estêncil",
        "falhas smt", "tendência smt", "problema smt",
        
        # GENERAL
        "olá, me ajude", "o que significa dppm?", "resumo do dashboard", "tudo bem?", 
        "agora me mostre", "o que é SMT?", "fale sobre qualidade", "me dê um resumo geral",
    ]

    labels = [
        # QUALITY
        "quality", "quality", "quality", "quality", "quality", "quality", "quality",
        # CAUSAS
        "causas", "causas", "causas", "causas", "causas", "causas", 
        # INDIVIDUAL
        "individual", "individual", "individual", "individual", "individual", "individual",
        # NLP
        "nlp", "nlp", "nlp", "nlp",
        # SMT_FOCO
        "smt_foco", "smt_foco", "smt_foco", "smt_foco", "smt_foco", "smt_foco", "smt_foco",
        # GENERAL
        "general", "general", "general", "general", "general", "general", "general", "general",
    ]

    texts += [
        "qual o desempenho da linha 3?", 
        "quem está gerando mais falhas na PTH?",
        "comparar a performance da máquina 10 com a 20",
        "top 5 de máquinas com defeito",
        "desempenho da célula de trabalho A",
    ]
    labels += [
        "individual", 
        "individual",
        "individual",
        "individual",
        "individual",
    ]

    # No bloco NLP:
    texts += [
        "analisar os comentários de observação",
        "agrupar o texto livre por assunto",
        "quais os temas mais falados nas anotações?",
        "resumo das observações",
        "tópicos das anotações de falha",
    ]
    labels += [
        "nlp",
        "nlp",
        "nlp",
        "nlp",
        "nlp",
    ]
    
    # --- 2. Expansão Automática de Setores (Melhoria 3) ---
    # Usando os nomes dos setores reais que você forneceu.
    # Nomes simplificados para capturar a intenção principal (o filtro fino será feito no analyst.py)
    SETORES_CHAVE = [
        'SMT', 'PTH', 'Revisão', 'Proteção 1', 'Proteção 2', 
        'Tempo', 'Nível', 'Assistência'
    ]
    
    # Adiciona variações para cada setor
    for s in SETORES_CHAVE:
        s_lower = s.lower().replace(' ', '_')
        texts += [
            f"análise do setor de {s}", f"como está o setor {s}?",
            f"tendência de falhas do {s}", f"problemas na {s}", 
            f"falhas da {s}", f"desempenho do {s} por operador",
            f"qualidade do setor de {s}", f"rejeição da {s}",
        ]
        labels += ["sector"] * 8
        
    if len(texts) != len(labels):
        print("ERRO: O número de textos de treinamento e rótulos não coincide após a expansão.")
        return

    # 3. Inicialização e Treinamento
    classifier = IntentClassifier()
    print(f"Treinando com {len(texts)} amostras e {len(set(labels))} classes...")
    # A função .train() roda a normalização, o fit do TF-IDF, o fit do LR com balanceamento, e o cross-validation.
    classifier.train(texts, labels) 

    # 4. Salvar o Modelo
    output_filename = "intent_model.joblib"
    if classifier.is_trained:
        try:
            joblib.dump(classifier, output_filename)
            print(f"\n✅ MODELO DE INTENÇÃO TREINADO E SALVO com sucesso como '{output_filename}'")
            print(f"Classes: {', '.join(sorted(list(set(labels))))}")
        except Exception as e:
            print(f"\nERRO: Não foi possível salvar o arquivo '{output_filename}'. Detalhe: {e}")
    else:
        print("\n❌ TREINAMENTO FALHOU: O modelo não foi salvo.")


if __name__ == "__main__":
    train_and_save_intent_model()