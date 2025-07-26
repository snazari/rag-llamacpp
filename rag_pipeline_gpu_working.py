import os
import argparse
import json
import hashlib
import glob
import logging
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader, UnstructuredMarkdownLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA, HypotheticalDocumentEmbedder
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors.cross_encoder_rerank import CrossEncoderReranker
from langchain_core.embeddings import Embeddings
from pydantic.v1 import Field
from typing import Any, List
from langchain_core.documents import Document

# --- Custom HyDE Retriever ---
class HydeRetriever(VectorStoreRetriever):
    """Retriever that uses HyDE to embed the query and then searches the vector store."""

    embeddings: Embeddings = Field(..., description="The embeddings model to use for the query.")

    def _get_relevant_documents(self, query: str, *, run_manager: Any) -> List[Document]:
        """Get documents relevant to a query.

        Args:
            query: String to find relevant documents for.
            run_manager: The callbacks handler to use for this call.

        Returns:
            List of relevant documents.
        """
        embedded_query = self.embeddings.embed_query(query)
        return self.vectorstore.similarity_search_by_vector(
            embedding=embedded_query, **self.search_kwargs
        )

# --- Configuration ---
PERSIST_DIRECTORY = "./chroma_db"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
#MODEL_PATH = "/home/sam/sandbox/rag/models/capybarahermes-2.5-mistral-7b.Q6_K.gguf"
#MODEL_PATH = "/home/sam/sandbox/rag/models/jan-nano-128k-Q8_0.gguf"
#MODEL_PATH = "/home/sam/sandbox/rag/models/DeepSeek-R1-0528-Qwen3-8B-BF16.gguf"
#MODEL_PATH = "/home/sam/sandbox/rag/models/LFM2-1.2B-F16.gguf"
#MODEL_PATH = "/home/sam/sandbox/rag/models/UNCENSORED-Fusetrix-Dolphin-3.2-1B-GRPO_Creative_RP.Q8_0.gguf"
#MODEL_PATH = "/home/sam/sandbox/rag/models/L3.2-8X3B-MOE-Dark-Champion-Inst-18.4B-uncen-ablit_D_AU-q5_k_m.gguf"
#MODEL_PATH = "/home/sam/sandbox/rag/models/Qwen2.5-7B-Instruct.fp16.gguf"
#MODEL_PATH = "/home/sam/sandbox/rag/models/Qwen3-8B.fp16.gguf"
MODEL_PATH = "/home/sam/sandbox/rag/models/Meta-Llama-3-8B-Instruct.fp16.gguf"
TXT_DOCUMENT_DIRECTORY = '/home/sam/sandbox/rag/docs/final_for_rag'
MD_DOCUMENT_DIRECTORY = '/home/sam/sandbox/rag/docs/md/'

# --- Helper Functions for Incremental Updates ---

def load_manifest():
    MANIFEST_FILE = os.path.join(PERSIST_DIRECTORY, "processed_files.json")
    if os.path.exists(MANIFEST_FILE):
        with open(MANIFEST_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_manifest(manifest):
    if not os.path.exists(PERSIST_DIRECTORY):
        os.makedirs(PERSIST_DIRECTORY)
    MANIFEST_FILE = os.path.join(PERSIST_DIRECTORY, "processed_files.json")
    with open(MANIFEST_FILE, 'w') as f:
        json.dump(manifest, f, indent=4)

def get_file_hash(filepath):
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("rag_pipeline.log"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)
console = Console()

def process_llm_response(llm_response):
    answer_panel = Panel(
        Markdown(llm_response['result']),
        title="[bold green]Answer[/bold green]",
        border_style="green"
    )
    console.print(answer_panel)

    sources_content = ""
    for source in llm_response["source_documents"]:
        sources_content += f"- {source.metadata.get('source', 'Unknown source')}\n"

    sources_panel = Panel(
        Markdown(sources_content),
        title="[bold blue]Sources[/bold blue]",
        border_style="blue"
    )
    console.print(sources_panel)

def main():
    """Main function to run the RAG pipeline."""
    parser = argparse.ArgumentParser(description="RAG pipeline for document Q&A with incremental updates.")
    parser.add_argument("--ingest-only", action="store_true", help="Run the ingestion process only and then exit.")
    args = parser.parse_args()

    # --- 1. Load Embedding Model ---
    logger.info("Loading embedding model...")
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )
    logger.info("Embedding model loaded.")

    # --- 2. Handle Document Ingestion (Incremental Updates) ---
    if args.ingest_only:
        logger.info("--- Starting document ingestion with incremental updates... ---")
        manifest = load_manifest()

        # Scan for all current documents
        all_current_files = {}
        for dir_path in [TXT_DOCUMENT_DIRECTORY, MD_DOCUMENT_DIRECTORY]:
            for filepath in glob.glob(os.path.join(dir_path, "**/*"), recursive=True):
                if os.path.isfile(filepath) and filepath.endswith(('.txt', '.md')):
                    all_current_files[filepath] = get_file_hash(filepath)

        # Identify new, modified, and deleted files
        files_to_add = {fp for fp, h in all_current_files.items() if manifest.get(fp) != h}
        files_to_delete = {fp for fp in manifest if fp not in all_current_files}
        files_to_update = {fp for fp in files_to_add if fp in manifest}
        files_to_delete.update(files_to_update) # Treat updates as delete then add

        vectorstore = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedding_model)

        # Delete documents
        if files_to_delete:
            logger.info(f"Deleting {len(files_to_delete)} old/modified documents...")
            ids_to_delete = []
            for filepath in files_to_delete:
                logger.debug(f"Getting chunks for deleted file: {filepath}")
                results = vectorstore.get(where={"source": filepath})
                if results['ids']:
                    ids_to_delete.extend(results['ids'])
            if ids_to_delete:
                vectorstore.delete(ids=ids_to_delete)
                logger.info(f"Deleted {len(ids_to_delete)} chunks for {len(files_to_delete)} files.")

        # Add new/modified documents
        if files_to_add:
            logger.info(f"Adding {len(files_to_add)} new/modified documents...")
            documents_to_load = []
            for filepath in files_to_add:
                logger.debug(f"Loading new/modified file: {filepath}")
                loader = UnstructuredFileLoader(filepath) if filepath.endswith('.txt') else UnstructuredMarkdownLoader(filepath)
                documents_to_load.extend(loader.load())

            if documents_to_load:
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
                splits = text_splitter.split_documents(documents_to_load)
                doc_ids = [f"{doc.metadata['source']}_{i}" for i, doc in enumerate(splits)]
                logger.info(f"Adding {len(splits)} document chunks to the vector store.")
                batch_size = 5000
                for i in range(0, len(splits), batch_size):
                    batch_splits = splits[i:i + batch_size]
                    batch_ids = doc_ids[i:i + batch_size]
                    logger.info(f"Adding batch {i // batch_size + 1}/{(len(splits) - 1) // batch_size + 1} with {len(batch_splits)} chunks.")
                    vectorstore.add_documents(documents=batch_splits, ids=batch_ids)
                logger.info(f"Added {len(splits)} chunks.")

        save_manifest(all_current_files)
        logger.info("Ingestion process complete. Exiting.")
        return

    # --- 3. Setup for Q&A (if not ingest-only) ---
    if not os.path.exists(PERSIST_DIRECTORY):
        logger.error("Chroma database not found. Please run with --ingest-only first.")
        return

    logger.info("Loading vector store for Q&A...")
    vectorstore = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedding_model)

    logger.info("Loading LLM for Q&A...")
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model file not found at {MODEL_PATH}")
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

    # --- 2. Initialize LLM, Callbacks, and Embeddings ---
    logger.info("Initializing models and embeddings...")
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    # LLM for generating the final answer (with streaming)
    llm = LlamaCpp(
        model_path=MODEL_PATH,
        n_gpu_layers=-1,  # Offload all layers to GPU
        n_batch=2048,     # Should be between 1 and n_ctx, consider memory limits
        n_ctx=8192,       # Context window
        f16_kv=True,      # Use half-precision for KV cache, saves memory
        callback_manager=callback_manager,
        verbose=False,     # Disable detailed llama.cpp logging
        temperature=0.1,
    )

    # A separate, non-verbose LLM for generating hypothetical documents
    hyde_llm = LlamaCpp(
        model_path=MODEL_PATH,
        n_gpu_layers=-1,
        n_batch=1024,
        n_ctx=8192,
        f16_kv=True,
        callback_manager=None, # No streaming
        verbose=False, # No llama.cpp logs
        temperature=0.5,
    )

    # --- 4. Creating the RAG Chain with HyDE ---
    logger.info("Creating RAG chain with HyDE...")

    # Hypothetical Document Embedder (HyDE)
    hyde_prompt_template = """Please write a short, hypothetical document that answers the user's question.
Question: {question}
Hypothetical Document:"""
    HYDE_PROMPT = PromptTemplate.from_template(hyde_prompt_template)

    # 1. Create the HyDE embedder using the non-verbose LLM
    hyde_embeddings = HypotheticalDocumentEmbedder.from_llm(hyde_llm, embedding_model, custom_prompt=HYDE_PROMPT)

    # 2. Create an instance of our custom HydeRetriever
    hyde_retriever = HydeRetriever(
        vectorstore=vectorstore,
        embeddings=hyde_embeddings,
        search_kwargs={"k": 10},
    )

    # 3. Set up the re-ranker
    reranker_model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", model_kwargs={'device': 'cpu'})
    compressor = CrossEncoderReranker(model=reranker_model, top_n=3)

    # 4. Create the compression retriever
    retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=hyde_retriever)

    # Final QA Chain Prompt
    question_prompt_template = """Use the following pieces of context to answer the question. If you don't know the answer, say "I don't know".
Context: {context}
Question: {question}
Answer:"""
    QUESTION_PROMPT = PromptTemplate.from_template(question_prompt_template)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever,
        return_source_documents=True, chain_type_kwargs={"prompt": QUESTION_PROMPT}
    )

    # --- 5. Interactive Question Answering ---
    console.print(Panel("[bold cyan]Local RAG system is ready. Type 'exit' or 'quit' to stop.[/bold cyan]"))
    try:
        while True:
            query = console.input("\n[bold yellow]Ask a question about your documents: [/bold yellow]")
            if query.lower() in ["exit", "quit"]:
                logger.info("Exiting interactive Q&A.")
                break
            logger.info(f"Received query: {query}")
            with console.status("[bold green]Searching for answers...[/bold green]"):
                llm_response = qa_chain.invoke(query)
            process_llm_response(llm_response)
    except (EOFError, KeyboardInterrupt):
        print("\nExiting...")

if __name__ == "__main__":
    main()
