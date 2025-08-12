import streamlit as st
from langchain.llms import Ollama

st.set_page_config(page_title="Prompt Engineering Copilot", layout="centered")
st.title("🦾 Prompt Engineering Copilot")

# Sidebar for model selection and settings
with st.sidebar:
    st.header("Settings")
    ollama_url = st.text_input("Ollama API URL", value="http://localhost:11434")
    model = st.selectbox("Model", options=["llama3.1", "qwen3", "gemma3"], index=0)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.05)

# Prompt input
prompt = st.text_area("Enter your prompt:", height=150)

if st.button("Generate"):
    if not prompt.strip():
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Generating response..."):
            llm = Ollama(
                base_url=ollama_url,
                model=model,
                temperature=temperature,
            )
            try:
                response = llm(prompt)
                st.success("Response:")
                st.code(response, language="markdown")
            except Exception as e:
                st.error(f"Error: {e}")

st.markdown("---")