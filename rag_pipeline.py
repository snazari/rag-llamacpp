# This is for Local RAG with a Local LLM - Does not currently use GPU - see the LlamaCpp lines below
# Sam Nazari
# 2025-07-08
import os
from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

def load_markdown_with_fallback(path):
    """
    Loads markdown files from a directory with multiple encoding fallbacks.
    """
    docs = []
    md_files = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(".md"):
                md_files.append(os.path.join(root, file))

    for file_path in md_files:
        loaded = False
        # Try multiple encodings in order
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1', 'utf-16']
        
        for encoding in encodings:
            try:
                loader = UnstructuredFileLoader(file_path, unstructured_kwargs={'encoding': encoding})
                docs.extend(loader.load())
                loaded = True
                break
            except (UnicodeDecodeError, Exception) as e:
                if encoding == encodings[-1]:  # Last encoding failed
                    print(f"Failed to load {file_path} with all encodings: {e}")
                continue
        
        if not loaded:
            print(f"Skipping {file_path} - could not decode with any encoding")
    
    return docs

# --- 1. Document Loading and Splitting ---
txt_loader = DirectoryLoader('docs/final_for_rag', glob="**/*.txt", loader_cls=TextLoader, recursive=True)
md_loader = DirectoryLoader('docs/md', glob="**/*.md", loader_cls=UnstructuredMarkdownLoader, recursive=True)

# Load the documents from each loader and combine them
documents = txt_loader.load() + md_loader.load()

print(f"Loaded {len(documents)} documents.")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# --- 2. Embedding and Vector Store ---
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Process documents in batches to avoid ChromaDB batch size limit
BATCH_SIZE = 5000  # Set batch size below ChromaDB's limit
print(f"Processing {len(texts)} text chunks in batches of {BATCH_SIZE}...")

# Initialize the vectorstore with the first batch
first_batch = texts[:BATCH_SIZE]
vectorstore = Chroma.from_documents(documents=first_batch, embedding=embedding_model, persist_directory="./chroma_db_2")
print(f"Created vectorstore with first batch of {len(first_batch)} documents.")

# Add remaining documents in batches
for i in range(BATCH_SIZE, len(texts), BATCH_SIZE):
    batch = texts[i:i+BATCH_SIZE]
    print(f"Adding batch {i//BATCH_SIZE + 1}: {len(batch)} documents...")
    vectorstore.add_documents(batch)
    print(f"Added batch {i//BATCH_SIZE + 1} successfully.")

print(f"Vectorstore creation complete with {len(texts)} total documents.")

# --- 3. Loading the Local LLM ---
# MODEL_PATH = "/home/sam/sandbox/rag/models/capybarahermes-2.5-mistral-7b.Q6_K.gguf" # <-- IMPORTANT: Set your model path here
# MODEL_PATH = "/home/sam/sandbox/rag/models/DeepSeek-R1-0528-Qwen3-8B-BF16.gguf" # <-- IMPORTANT: Set your model path here
# MODEL_PATH = "/home/sam/sandbox/rag/models/jan-nano-128k-Q8_0.gguf" # <-- IMPORTANT: Set your model path here
MODEL_PATH = "/home/sam/sandbox/rag/models/L3.2-8X3B-MOE-Dark-Champion-Inst-18.4B-uncen-ablit_D_AU-q5_k_m.gguf"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
llm = LlamaCpp(
    model_path=MODEL_PATH,
    temperature=0.75,
    max_tokens=2000,
    top_p=1,
    callbacks=callback_manager.handlers,
    verbose=True,
    n_ctx=4096,
    n_gpu_layers=32 # Re-enable GPU offloading
)

# --- 4. Creating the RAG Chain ---
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
    return_source_documents=True
)

# --- 5. Interactive Question Answering ---
def process_llm_response(llm_response):
    print("\n--- Answer ---")
    print(llm_response['result'])
    print('\n--- Sources ---')
    for source in llm_response["source_documents"]:
        print(f"- {source.metadata['source']}")

print("Local RAG system is ready. Type 'exit' or 'quit' to stop.")
while True:
    query = input("\nAsk a question about your documents: ")
    if query.lower() in ["exit", "quit"]:
        break
    llm_response = qa_chain.invoke(query)
    process_llm_response(llm_response)
