import os
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
)
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# --- Configuration ---
# Define paths for easy management
DOCS_DIR = "./docs"
STORAGE_DIR = "./storage" # Directory where LlamaIndex will save its index
MODEL_PATH = "../models/Qwen3-30B-A3B-Instruct-2507-Q4_0.gguf"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LOCAL_EMBEDDING_DIR = "./embedding_models"  # Local directory for embedding models
EMBEDDING_CACHE_DIR = "./embedding_models_cache"  # Cache directory for downloaded models

def setup_embedding_model():
    """
    Setup embedding model with local directory preference and HuggingFace fallback.
    """
    # Create directories if they don't exist
    os.makedirs(LOCAL_EMBEDDING_DIR, exist_ok=True)
    os.makedirs(EMBEDDING_CACHE_DIR, exist_ok=True)
    
    # Check for local embedding model first
    local_model_path = os.path.join(LOCAL_EMBEDDING_DIR, EMBEDDING_MODEL_NAME)
    
    if os.path.exists(local_model_path):
        print(f"✅ Found local embedding model at: {local_model_path}")
        try:
            # Try to load from local directory
            embedding_model = HuggingFaceEmbedding(
                model_name=local_model_path,
                cache_folder=EMBEDDING_CACHE_DIR,
                device="cuda"  # Use GPU if available
            )
            print(f"✅ Successfully loaded local embedding model")
            return embedding_model
        except Exception as e:
            print(f"⚠️ Failed to load local model: {e}")
            print(f"Falling back to HuggingFace download...")
    else:
        print(f"📥 Local embedding model not found at {local_model_path}")
        print(f"Will download from HuggingFace and cache locally...")
    
    # Fallback to HuggingFace download with local caching
    try:
        embedding_model = HuggingFaceEmbedding(
            model_name=EMBEDDING_MODEL_NAME,
            cache_folder=EMBEDDING_CACHE_DIR,
            device="cuda"  # Use GPU if available
        )
        print(f"✅ Successfully loaded embedding model from HuggingFace")
        
        # Provide tip for future local usage
        print(f"💡 Tip: You can copy the downloaded model from {EMBEDDING_CACHE_DIR} to {LOCAL_EMBEDDING_DIR} to avoid future downloads.")
        
        return embedding_model
    except Exception as e:
        print(f"❌ Failed to load embedding model: {e}")
        raise

def main():
    """
    This script loads documents from the 'docs' folder, creates a vector index
    using LlamaIndex, and saves it to the 'storage' directory.
    """
    print("🚀 Starting data preparation with LlamaIndex...")

    # --- Configure Global Settings ---
    # LlamaIndex uses a global Settings object to configure components.
    # This makes it easy to use the same models for both indexing and querying.

    print(f"Loading LLM from: {MODEL_PATH}")
    Settings.llm = LlamaCPP(
        model_path=MODEL_PATH,
        temperature=0.1,
        max_new_tokens=512,
        context_window=3900,
        # Offload all layers to GPU
        model_kwargs={"n_gpu_layers": -1},
        verbose=True,
    )

    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    Settings.embed_model = setup_embedding_model()

    # --- Load Documents ---
    if not os.path.exists(DOCS_DIR):
        print(f"❌ Error: Document directory not found at '{DOCS_DIR}'")
        return

    print(f"📚 Loading documents from '{DOCS_DIR}'...")
    # SimpleDirectoryReader is a powerful LlamaIndex tool that can read
    # various file types (PDF, MD, DOCX, etc.) from a folder.
    documents = SimpleDirectoryReader(DOCS_DIR).load_data()
    print(f"Loaded {len(documents)} document(s).")

    # --- Create and Persist the Index ---
    # This is the core of LlamaIndex. It automatically handles:
    # 1. Text Splitting (chunking the documents)
    # 2. Embedding (converting text chunks to vectors)
    # 3. Indexing (storing the vectors for fast retrieval)
    print("⚙️ Creating the vector index...")
    index = VectorStoreIndex.from_documents(documents)

    # Save the index to disk for later use in your application
    print(f"💾 Persisting index to '{STORAGE_DIR}'...")
    index.storage_context.persist(persist_dir=STORAGE_DIR)

    print("\n✅ Data preparation complete!")
    print(f"Your index has been successfully created in the '{STORAGE_DIR}' directory.")

if __name__ == "__main__":
    main()