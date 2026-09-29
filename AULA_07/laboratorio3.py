import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier

dados_treino = [
    ("Onde esta meu pedido e qual o rastreio", "rastrear_pedido"),
    ("Quero saber a localizacao da minha encomenda", "rastrear_pedido"),
    ("Quero cancelar minha compra e pedir reembolso", "cancelar_compra"),
    ("Preciso cancelar o pedido feito ontem", "cancelar_compra"),
    ("Qual o horario de atendimento da loja fisica", "fora_escopo")
]

stopwords = ["o", "a", "meu", "minha", "e", "da", "do"]

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

modelo_tree = DecisionTreeClassifier()
modelo_tree.fit(X_vetorizado, y_treino)

# 3. EXTRAÇÃO DE ENTIDADE
def extrair_codigo_rastreio(texto):
    match = re.search(r'BR\d{9}', texto, re.IGNORECASE)
    return match.group(0).upper() if match else None

# 4. PIPELINE NLU
def processar_ecommerce(mensagem_usuario, threshold=0.50):
    texto_p = pré_processar(mensagem_usuario)
    vetor_input = vectorizer.transform([texto_p])

    probas = modelo_tree.predict_proba(vetor_input)[0]

    maior_confianca = max(probas)
    intencao_prevista = modelo_tree.classes_[probas.argmax()]

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
            "codigo_rastreio": extrair_codigo_rastreio(mensagem_usuario)
        }

# --- TESTE 3 ---
if __name__ == "__main__":
    print(processar_ecommerce("Quero rastrear o pedido BR987654321"))