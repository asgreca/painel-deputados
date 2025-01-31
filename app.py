import os
import requests
import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import asyncio

# 📌 Título do aplicativo no Streamlit
st.title("🗣️ Análise Política - Discursos dos Deputados Federais")

# ================================================================
# 🔹 1) INICIALIZAR SESSION STATE (API KEY, Modelo, etc.)
# ================================================================
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = None
if "selected_model" not in st.session_state:
    st.session_state.selected_model = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# 🔑 Entrada para a API Key da OpenAI
if not st.session_state.openai_api_key:
    st.session_state.openai_api_key = st.text_input(
        "Insira sua API Key da OpenAI:",
        type="password",
        placeholder="Cole sua API Key aqui",
    )

if not st.session_state.openai_api_key:
    st.warning("Por favor, insira sua API Key para continuar.")
    st.stop()

# 🔹 Lista de modelos disponíveis
model_options = {
    "gpt-4o": "Modelo versátil e de alta inteligência.",
    "gpt-4o-mini": "Modelo menor, rápido e acessível, ideal para tarefas específicas.",
}

if not st.session_state.selected_model:
    st.session_state.selected_model = st.selectbox(
        "Escolha o modelo GPT:",
        options=[""] + list(model_options.keys()),
        format_func=lambda x: f"{x} - {model_options.get(x, 'Selecione um modelo')}" if x else "Selecione um modelo",
    )

if not st.session_state.selected_model:
    st.warning("Por favor, escolha um modelo GPT para continuar.")
    st.stop()

# ================================================================
# 📌 CONFIGURAÇÃO DO ENDPOINT ChromaDB
# ================================================================
CHROMA_API_URL = "https://chroma-production-6065.up.railway.app/api/v1/collections/40535baa-0a68-4862-9e4c-1963f4981795/query"

# ================================================================
# 🔹 2) CRIAR EMBEDDINGS E MODELO LLM
# ================================================================
if "embedder" not in st.session_state:
    try:
        st.session_state.embedder = OpenAIEmbeddings(
            openai_api_key=st.session_state.openai_api_key,
            model="text-embedding-ada-002"
        )
    except Exception as e:
        st.error(f"Erro ao criar embeddings: {str(e)}")
        st.stop()

if "llm" not in st.session_state:
    try:
        st.session_state.llm = ChatOpenAI(
            model=st.session_state.selected_model,
            temperature=0.5,
            openai_api_key=st.session_state.openai_api_key,
        )
    except Exception as e:
        st.error(f"Erro ao inicializar o modelo de linguagem: {str(e)}")
        st.stop()

# ================================================================
# 🔎 Função para consultar o ChromaDB (via embeddings)
# ================================================================
async def buscar_contexto(query_text):
    """
    Gera embedding do texto e faz POST no endpoint /query do ChromaDB,
    retornando documentos relevantes.
    """
    try:
        # Gerar embedding usando OpenAI
        st.write("🔍 Gerando embedding para a consulta...")
        embedding = await st.session_state.embedder.aembed_query(query_text)

        # Preparar o payload
        payload = {
            "query_embeddings": [embedding],
            "n_results": 5,
            "include": ["documents", "metadatas", "distances"]
        }
        st.write("📤 Payload enviado ao ChromaDB:", payload)

        # Fazer a requisição
        response = requests.post(
            CHROMA_API_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10
        )

        # Log da resposta
        st.write("🔍 Status da resposta:", response.status_code)
        st.write("📥 Conteúdo da resposta:", response.text)

        if response.status_code == 200:
            data = response.json()
            docs = data.get("documents", [[]])
            if not docs or not docs[0]:
                return "Nenhum contexto relevante foi encontrado."
            return "\n\n".join(doc[:5000] for doc in docs[0])
        else:
            return f"Erro ao buscar contexto: {response.status_code} - {response.text}"

    except Exception as e:
        st.error(f"❌ Erro ao se conectar com ChromaDB: {str(e)}")
        return f"Erro ao se conectar com ChromaDB: {str(e)}"

# ================================================================
# 🔹 3) TESTAR CONEXÃO A PARTIR DE UMA PERGUNTA DE EXEMPLO
# ================================================================
def testar_conexao_chromadb():
    """
    Faz uma consulta de teste ao ChromaDB para verificar conectividade.
    """
    st.info("Testando conexão com ChromaDB usando a query 'teste de conexão'...")

    try:
        contexto_teste = asyncio.run(buscar_contexto("teste de conexão"))

        if "Erro ao buscar contexto" in contexto_teste or "Erro ao se conectar" in contexto_teste:
            st.error(f"❌ Falha ao conectar ou buscar contexto: {contexto_teste}")
        else:
            st.success("✅ Conexão OK! Resposta de teste obtida.")
            st.write("**Resposta do ChromaDB:**")
            st.write(contexto_teste)

    except Exception as e:
        st.error(f"❌ Erro inesperado ao testar conexão com ChromaDB: {str(e)}")

if "test_conexao_feito" not in st.session_state:
    st.session_state.test_conexao_feito = True
    testar_conexao_chromadb()

# ================================================================
# 🔹 4) SETUP DO PROMPT E DA CADEIA LLM
# ================================================================
prompt_text = """
Você é um analista político. Sua função é analisar os discursos exclusivamente dos deputados federais 
e fornecer respostas detalhadas e objetivas com base no histórico e no contexto abaixo. Não fale frases genéricas e evasivas, sempre indique a origem da ideia, onde estava registrada como em qual comissão, data e número da reunião. A resposta a ser dada deve ter sempre pelo menos 3 parágrafos.

Histórico:
{history}

Contexto:
{context}

Pergunta:
{question}

Resposta:
"""
prompt = PromptTemplate(
    input_variables=["history", "context", "question"],
    template=prompt_text
)
chain = LLMChain(llm=st.session_state.llm, prompt=prompt)

# ================================================================
# 💬 5) EXIBIR HISTÓRICO & INTERAÇÃO NO CHAT
# ================================================================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Digite sua pergunta:"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 🔍 Buscar contexto no ChromaDB
    contexto = asyncio.run(buscar_contexto(user_input))

    try:
        response = asyncio.run(chain.arun(history="", context=contexto, question=user_input))
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant").markdown(response)
    except Exception as e:
        st.error(f"Erro ao gerar resposta: {e}")
