import streamlit as st
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

embedding_fn = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
chroma_client = Chroma(
    embedding_function=embedding_fn,
    persist_directory="chroma_db"
    )

# Initialize Ollama LLM for LangChain
llm = Ollama(model="llama3.1")

# Define the prompt template string inside a docstring for clarity
template = """
You are Co-Pilot, an expert AI assistant designed to help users with their questions and tasks.
Use the provided context from retrieved documents to inform your answers.
Always be concise, accurate, and helpful.

Chat History:
{history}

Relevant Context:
{context}

User Question:
{user_input}

Your Response:
"""

# Create a prompt template for LangChain
prompt = PromptTemplate(
    input_variables=["history", "user_input", "context"],
    template=template
)

# Create the LLMChain
chain = LLMChain(llm=llm, prompt=prompt)

st.title("Co-Pilot Chatbot with ChromaDB & LangChain")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("You:", key="user_input")

def retrieve_context(query):
    # Query ChromaDB for similar documents (no collection)
    results = chroma_client.similarity_search(
        query=query,
        k=10
    )
    print("Retrieved documents:", results)
    # Concatenate retrieved documents as context
    docs = [doc for doc in results['documents'][0]]
    return "\n".join(docs)

def format_history(chat_history):
    # Format chat history for the prompt
    return "\n".join(
        [f"User: {entry['user']}\nAssistant: {entry['bot']}" for entry in chat_history]
    )

if st.button("Send") and user_input:
    # Retrieve context from ChromaDB
    context = retrieve_context(user_input)
    # Format chat history
    history = format_history(st.session_state.chat_history)
    # Run the LLMChain
    bot_reply = chain.run(
        history=history,
        user_input=user_input,
        context=context
    )
    # Update chat history
    st.session_state.chat_history.append({"user": user_input, "bot": bot_reply})

# Display chat history
for entry in st.session_state.chat_history:
    st.markdown(f"**You:** {entry['user']}")
    st.markdown(f"**Bot:** {entry['bot']}")

# Optional: Add document upload for RAG
st.sidebar.header("Add Documents to ChromaDB")
uploaded_file = st.sidebar.file_uploader("Upload a text file", type=["txt"])
if uploaded_file:
    text = uploaded_file.read().decode("utf-8")
    # Embed and add to ChromaDB (no collection)
    embedding = embedding_fn([text])[0]
    chroma_client.add(
        documents=[text],
        embeddings=[embedding],
        ids=[str(hash(text))]
    )
    st.sidebar.success("Document added to ChromaDB!")