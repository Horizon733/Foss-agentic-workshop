import os
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
import time

# --- RAG/ChromaDB imports ---
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

import streamlit as st

# === AGENT SETUP ===
PROMPT = """
You are Co-Pilot, an expert AI assistant designed to help users with their questions and tasks.
Use the provided context from retrieved documents to inform your answers.
Always be concise, accurate, and helpful.
Use rag_search tool whenever necessary.

"""


# --- ChromaDB RAG setup ---
embedding_fn = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
chroma_client = Chroma(
    embedding_function=embedding_fn,
    persist_directory="chroma_db"
)

def retrieve_rag_context(query: str, k: int = 5) -> List[str]:
    results = chroma_client.similarity_search(query=query, k=k)
    docs = [doc.page_content for doc in results]
    return docs

ollama_model = OpenAIModel(
    model_name='qwen3',
    provider=OpenAIProvider(base_url='http://localhost:11434/v1'),
)

class RAGquery(BaseModel):
    query: str

class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the message sender, either 'user' or 'bot'. 'bot' indicates an AI response.")
    content: str

agent = Agent(
    model=ollama_model,
    deps_type=RAGquery,
    output_type=ChatMessage,
    system_prompt=PROMPT
)

@agent.tool
def rag_search(ctx: RunContext[RAGquery], query: str, k: int = 5) -> str:
    """Retrieve relevant context from ChromaDB for a given query."""
    return "\n".join(retrieve_rag_context(query, k))

# --- Streamlit Chat App ---
st.set_page_config(page_title="Co-Pilot Agentic Chat", page_icon="🤖")
st.markdown(
    """
    <style>
    .block-container { padding-top: 1rem; }
    .stChatMessage { margin-bottom: 1.5rem; }
    .stTextInput>div>div>input { font-size: 1.1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("🤖 Co-Pilot Agentic Chat")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history using ChatGPT-like bubbles
for entry in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(entry["user"])
    with st.chat_message("assistant"):
        st.markdown(entry["bot"])

# Chat input at the bottom
def chat_input():
    return st.text_input(
        "Type your message...",
        key="user_input",
        placeholder="Ask me anything...",
        label_visibility="collapsed"
    )

user_input = chat_input()

if user_input and (st.session_state.get("last_input") != user_input):
    st.session_state["last_input"] = user_input
    
    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Show spinner animation
    with st.chat_message("assistant"):
        with st.spinner("Co-Pilot is thinking..."):
            # Run the agent
            result = agent.run_sync(
                f"User: {user_input}",
                deps=RAGquery(query=user_input)
            )
            bot_reply = result.output.content if hasattr(result.output, "content") else str(result.output)
    
    # Add to chat history and rerun to show final result
    st.session_state.chat_history.append({"user": user_input, "bot": bot_reply})
    st.rerun()
    bot_reply = result.output.content if hasattr(result.output, "content") else str(result.output)
    
    # Add to chat history and rerun to show final result
    st.session_state.chat_history.append({"user": user_input, "bot": bot_reply})
    st.rerun()

