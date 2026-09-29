import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

dados_treino = [
    ("Urgente: o servidor caiu e a aplicacao esta fora", "incidente_critico"),
    ("Erro grave no banco de dados e sistema parou", "incidente_critico"),
    ("Como faco para alterar minha senha do e-mail", "duvida_suporte"),
    ("Qual o procedimento para solicitar novo mouse", "duvida_suporte"),
    ("Gostaria de saber o horario de funcionamento", "fora_escopo")
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

modelo_nb = MultinomialNB()
modelo_nb.fit(X_vetorizado, y_treino)

# 3. EXTRAÇÃO DE ENTIDADE
def extrair_protocolo(texto):
    match = re.search(r'INC-?\d{4}', texto, re.IGNORECASE)
    return match.group(0).upper() if match else None

# 4. PIPELINE NLU
def processar_triagem(mensagem_usuario, threshold=0.60):
    texto_p = pré_processar(mensagem_usuario)
    vetor_input = vectorizer.transform([texto_p])

    probas = modelo_nb.predict_proba(vetor_input)[0]

    maior_confianca = max(probas)
    intencao_prevista = modelo_nb.classes_[probas.argmax()]

    if maior_confianca < threshold:
        return {
            "status": "FALLBACK",
            "confianca": maior_confianca
        }
    else:
        return {
            "status": "SUCESSO",
            "intencao": intencao_prevista,
            "confianca": maior_confianca,
            "protocolo": extrair_protocolo(mensagem_usuario)
        }

# --- TESTE 2 ---
if __name__ == "__main__":
    print(processar_triagem("Urgente: servidor caiu no protocolo INC-9982"))