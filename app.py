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
# 🔹 1) ENTRADA DA API KEY COM BOTÃO DE CONFIRMAÇÃO
# ================================================================
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = ""

api_key = st.text_input(
    "🔑 Insira sua API Key da OpenAI:",
    type="password",
    placeholder="Cole sua API Key aqui",
)

if st.button("✅ Confirmar API Key"):
    if api_key:
        st.session_state.openai_api_key = api_key
        st.success("API Key salva com sucesso!")
    else:
        st.error("Por favor, insira uma API Key válida.")

if not st.session_state.openai_api_key:
    st.warning("⚠️ API Key não definida. Por favor, insira sua chave para continuar.")
    st.stop()

# ================================================================
# 🔹 2) SELEÇÃO DO MODELO GPT COM BOTÃO DE CONFIRMAÇÃO
# ================================================================
model_options = {
    "gpt-4o": "Modelo versátil e de alta inteligência.",
    "gpt-4o-mini": "Modelo menor, rápido e acessível, ideal para tarefas específicas.",
}

if "selected_model" not in st.session_state:
    st.session_state.selected_model = ""

selected_model = st.selectbox(
    "🧠 Escolha o modelo GPT:",
    options=[""] + list(model_options.keys()),
    format_func=lambda x: f"{x} - {model_options.get(x, '')}" if x else "Selecione um modelo",
)

if st.button("✅ Confirmar Modelo"):
    if selected_model:
        st.session_state.selected_model = selected_model
        st.success(f"Modelo **{selected_model}** selecionado com sucesso!")
    else:
        st.error("Por favor, selecione um modelo GPT válido.")

if not st.session_state.selected_model:
    st.warning("⚠️ Nenhum modelo selecionado. Escolha um para continuar.")
    st.stop()

# ================================================================
# 📌 CONFIGURAÇÃO DO ENDPOINT ChromaDB
# ================================================================
CHROMA_API_URL = "https://chroma-production-6065.up.railway.app/api/v1/collections/40535baa-0a68-4862-9e4c-1963f4981795/query"

# ================================================================
# 🔹 3) CRIAR EMBEDDINGS E MODELO LLM
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
        embedding = await st.session_state.embedder.aembed_query(query_text)
        payload = {
            "query_embeddings": [embedding],
            "n_results": 5,
            "include": ["documents"]
        }

        response = requests.post(
            CHROMA_API_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10
        )

        if response.status_code == 200:
            data = response.json()
            docs = data.get("documents", [[]])
            if not docs or not docs[0]:
                return "Nenhum contexto relevante foi encontrado."
            return "\n\n".join(doc[:5000] for doc in docs[0])
        else:
            return f"Erro ao buscar contexto: {response.status_code} - {response.text}"

    except Exception as e:
        return f"Erro ao se conectar com ChromaDB: {str(e)}"

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
MAX_HISTORICO = 5  # Número máximo de interações a serem armazenadas

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("💬 Digite sua pergunta:"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 🔍 Buscar contexto no ChromaDB
    contexto = asyncio.run(buscar_contexto(user_input))

    # Criar histórico formatado com até MAX_HISTORICO mensagens anteriores
    historico_formatado = "\n\n".join(
        [f"Usuário: {msg['content']}" if msg["role"] == "user" else f"Assistente: {msg['content']}" 
        for msg in st.session_state.messages[-MAX_HISTORICO:]]
    )

    try:
        # Passando o histórico real para o modelo
        response = asyncio.run(chain.arun(history=historico_formatado, context=contexto, question=user_input))

        # Armazena resposta no histórico
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant").markdown(response)
    except Exception as e:
        st.error(f"Erro ao gerar resposta: {e}")
