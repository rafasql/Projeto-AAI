import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

dados_treino = [
    ("Quero fazer um pix de 100 reais", "realizar_pix"),
    ("Preciso transferir 50 reais", "realizar_pix"),
    ("Qual é o saldo da minha conta", "consultar_saldo"),
    ("Quero consultar meu saldo", "consultar_saldo"),
    ("Qual o horario de funcionamento", "fora_escopo")
]

stopwords = ["o", "a", "e", "do", "da", "para", "de"]

# 1. PRÉ-PROCESSAMENTO
def pré_processar(texto):
    texto_limpo = re.sub(r'[^\w\s]', '', texto.lower()).strip()
    tokens = texto_limpo.split()

    tokens_filtrados = [token for token in tokens if token not in stopwords]

    return " ".join(tokens_filtrados)

X_treino_limpo = [pré_processar(item[0]) for item in dados_treino]
y_treino = [item[1] for item in dados_treino]

# 2. VETORIZAÇÃO E TREINAMENTO
vectorizer = TfidfVectorizer()
X_vetorizado = vectorizer.fit_transform(X_treino_limpo)

modelo = LogisticRegression()
modelo.fit(X_vetorizado, y_treino)

# 3. EXTRAÇÃO DE ENTIDADE
def extrair_valor_reais(texto_original):
    match = re.search(r'(\d+)\s*reais', texto_original, re.IGNORECASE)
    return float(match.group(1)) if match else None

# 4. PIPELINE NLU
def processar_nlu(mensagem_usuario, threshold=0.55):
    texto_p = pré_processar(mensagem_usuario)
    vetor_input = vectorizer.transform([texto_p])

    probas = modelo.predict_proba(vetor_input)[0]

    maior_confianca = max(probas)
    intencao_prevista = modelo.classes_[probas.argmax()]

    if maior_confianca < threshold or intencao_prevista == "fora_escopo":
        return {
            "status": "FALLBACK",
            "confianca": maior_confianca
        }
    else:
        return {
            "status": "SUCESSO",
            "intencao": intencao_prevista,
            "confianca": maior_confianca,
            "valor": extrair_valor_reais(mensagem_usuario)
        }

# --- TESTE 1 ---
if __name__ == "__main__":
    print(processar_nlu("Quero fazer um pix de 100 reais"))