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
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA, HypotheticalDocumentEmbedder
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors.cross_encoder_rerank import CrossEncoderReranker
from langchain_core.embeddings import Embeddings
from pydantic.v1 import Field
from typing import Any, List
from langchain_core.documents import Document

# --- Configuration ---
PERSIST_DIRECTORY = "./chroma_db_cloud"
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"
TXT_DOCUMENT_DIRECTORY = './docs/final_for_rag'
MD_DOCUMENT_DIRECTORY = './docs/md/'

# OpenAI Configuration
OPENAI_MODEL = "gpt-4o-mini"  # You can change this to gpt-4, gpt-3.5-turbo, etc.

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

def create_rag_chain(embedding_model, vectorstore, streaming=True):
    """Initializes and returns the RAG chain using OpenAI API.

    Args:
        embedding_model: The embedding model to use.
        vectorstore: The vector store to retrieve from.
        streaming (bool): Whether to use streaming responses from OpenAI.

    Returns:
        A RetrievalQA chain.
    """
    logger.info("Setting up OpenAI LLM for Q&A...")

    # Check if OpenAI API key is set
    OPENAI_API_KEY = ""
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY environment variable not set. Please set it before running.")
        raise ValueError("OPENAI_API_KEY environment variable not set")
    # Initialize OpenAI LLM
    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=0.2,
        max_tokens=4096,
        streaming=streaming,
        verbose=True,
        openai_api_key=OPENAI_API_KEY
    )

    # --- Creating Hybrid RAG Chain with Direct + Multi-Query Retrieval ---
    logger.info("Creating hybrid RAG chain with direct similarity and multi-query retrieval...")

    # 1. Direct similarity retriever (using original embeddings)
    direct_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 10, "lambda_mult": 0.7}
    )

    # 2. Multi-Query Retrieval with direct retriever
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=direct_retriever,
        llm=llm,
        prompt=PromptTemplate(
            input_variables=["question"],
            template="""Rewrite this question in 3 different ways:

{question}

1.
2.
3."""
        )
    )

    # 3. Set up the re-ranker
    reranker_model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", model_kwargs={'device': 'cuda'})
    compressor = CrossEncoderReranker(model=reranker_model, top_n=5)

    # 4. Create final retriever with compression
    retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=multi_query_retriever
    )

    # 5. Simplified QA Chain Prompt
    question_prompt_template = """Answer the question based on the context below. Be concise and accurate.

Context: {context}

Question: {question}

Answer:"""
    QUESTION_PROMPT = PromptTemplate.from_template(question_prompt_template)

    # 6. Create the final chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever,
        return_source_documents=True, chain_type_kwargs={"prompt": QUESTION_PROMPT}
    )

    return qa_chain

def main():
    """Main function to run the RAG pipeline."""
    parser = argparse.ArgumentParser(description="Local RAG system with OpenAI API")
    parser.add_argument("--ingest-only", action="store_true", help="Only ingest documents, don't start Q&A")
    args = parser.parse_args()

    # --- 1. Initialize Embedding Model ---
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # --- 2. Document Ingestion (Incremental) ---
    if args.ingest_only or not os.path.exists(PERSIST_DIRECTORY):
        logger.info("Starting document ingestion process...")

        # Load existing manifest
        manifest = load_manifest()

        # Get all current files and their hashes
        all_current_files = {}
        for directory in [TXT_DOCUMENT_DIRECTORY, MD_DOCUMENT_DIRECTORY]:
            if os.path.exists(directory):
                for pattern in ['*.txt', '*.md']:
                    for filepath in glob.glob(os.path.join(directory, '**', pattern), recursive=True):
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
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1024,
                    chunk_overlap=256,
                    separators=["\n\n", "\n", ". ", " ", ""]  # Better splitting
                )
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

    # --- Create the RAG Chain ---
    qa_chain = create_rag_chain(embedding_model, vectorstore, streaming=True)

    # --- 4. Interactive Question Answering ---
    console.print(Panel("[bold cyan]Cloud RAG system with OpenAI is ready. Type 'exit' or 'quit' to stop.[/bold cyan]"))
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
