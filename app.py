import streamlit as st
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain_chroma import Chroma

# --- App Configuration ---
st.set_page_config(page_title="Chat with Your Docs", page_icon="📄", layout="wide")
st.title("📄 Chat with Your Local Documents")
st.markdown("Powered by Llama.cpp, LangChain, and Streamlit")

# --- Model and Retriever Loading ---
#MODEL_PATH = "/home/sam/sandbox/rag/models/capybarahermes-2.5-mistral-7b.Q6_K.gguf" # <-- IMPORTANT: Set your GGUF model path here
#MODEL_PATH = "/home/sam/sandbox/rag/models/jan-nano-128k-Q8_0.gguf"
MODEL_PATH = "/home/sam/sandbox/rag/models/DeepSeek-R1-0528-Qwen3-8B-BF16.gguf"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
PERSIST_DIRECTORY = "./chroma_db" # Directory where your vector store is saved

@st.cache_resource
def load_llm(model_path):
    """Loads the LlamaCpp model."""
    llm = LlamaCpp(
        model_path=model_path,
        temperature=0.7,
        max_tokens=2000,
        top_p=1,
        n_ctx=4096,
        # --- GPU Acceleration Parameters ---
        n_gpu_layers=-1, # Offload all layers to GPU. -1 = all, 0 = none.
        n_batch=512,     # Should be between 1 and n_ctx.
        # --- CPU Threading (if not using GPU fully) ---
        # n_threads=8,     # Set to the number of physical CPU cores.
        verbose=True
    )
    return llm

@st.cache_resource
def load_retriever(embedding_model, persist_directory):
    """Loads the Chroma vector store and retriever."""
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    return retriever

# --- Load resources ---
try:
    llm = load_llm(MODEL_PATH)
    retriever = load_retriever(EMBEDDING_MODEL_NAME, PERSIST_DIRECTORY)
    
    # Create the RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
except Exception as e:
    st.error(f"Failed to load resources. Please check your model path and ChromaDB directory. Error: {e}")
    st.stop()


# --- Chat History Management ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Ask me anything about your documents."}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- User Input and Response ---
if prompt := st.chat_input("What is your question?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = qa_chain.invoke(prompt)
            answer = response['result']
            sources = "\n\n*Sources:*\n" + "\n".join([f"- {doc.metadata.get('source', 'Unknown')}" for doc in response['source_documents']])
            full_response = answer + sources
            st.markdown(full_response)
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})