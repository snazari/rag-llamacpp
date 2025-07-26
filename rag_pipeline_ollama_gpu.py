import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain.chains import RetrievalQA
from langchain_community.vectorstores.utils import filter_complex_metadata

# --- Configuration ---
DOCS_PATH = "docs/obsidianAmentum"
DB_PATH = "./chroma_db_ollama"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# This should be a model you have pulled with `ollama pull <model_name>`
#OLLAMA_MODEL = "deepseek-r1:14b"
OLLAMA_MODEL = "deepseek-r1:32b"

def load_markdown_with_fallback(path):
    """
    Loads markdown files from a directory with a fallback to 'latin-1' encoding.
    This function handles potential encoding errors in your documents.
    """
    docs = []
    md_files = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(".md"):
                md_files.append(os.path.join(root, file))

    for file_path in md_files:
        try:
            loader = UnstructuredFileLoader(file_path, mode="elements")
            docs.extend(loader.load())
        except UnicodeDecodeError:
            print(f"UTF-8 decoding failed for {file_path}. Trying latin-1.")
            try:
                loader = UnstructuredFileLoader(file_path, mode="elements", unstructured_kwargs={'encoding': 'latin-1'})
                docs.extend(loader.load())
            except Exception as e:
                print(f"Failed to load {file_path} with latin-1: {e}")
        except Exception as e:
            print(f"An unexpected error occurred with file {file_path}: {e}")
    return docs

def create_or_load_vector_db(db_path, docs_path):
    """
    Creates a new vector database if one doesn't exist, otherwise loads the existing one.
    """
    # Initialize the embedding model, configured to use the GPU
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cuda'}
    )

    if not os.path.exists(db_path):
        print(f"Database not found. Creating a new one at: {db_path}")

        # --- 1. Document Loading and Splitting ---
        print("Loading documents...")
        pdf_loader = DirectoryLoader(docs_path, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True)
        pdf_documents = pdf_loader.load()
        markdown_documents = load_markdown_with_fallback(docs_path)
        documents = pdf_documents + markdown_documents
        print(f"Loaded {len(documents)} total documents.")

        filtered_documents = filter_complex_metadata(documents)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(filtered_documents)
        print(f"Split documents into {len(texts)} chunks.")

        # --- 2. Embedding and Vector Store Creation ---
        print("Creating vector store with embeddings... (This may take a while)")
        vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=embedding_model,
            persist_directory=db_path
        )
        print("Vector store created and saved successfully.")
    else:
        print(f"Loading existing database from: {db_path}")
        vectorstore = Chroma(
            persist_directory=db_path,
            embedding_function=embedding_model
        )
        print("Database loaded successfully.")

    return vectorstore

def main():
    """
    Main function to run the RAG pipeline.
    """
    # Create or load the vector database
    vectorstore = create_or_load_vector_db(DB_PATH, DOCS_PATH)

    # --- 3. Connect to the Local LLM via Ollama ---
    print(f"Connecting to local LLM '{OLLAMA_MODEL}' via Ollama...")
    # This is much simpler than loading the model directly.
    # Assumes the Ollama application is running.
    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.7)

    # --- 4. Creating the RAG Chain ---
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 documents
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    # --- 5. Interactive Question Answering Loop ---
    print("\nLocal RAG system is ready. Type 'exit' or 'quit' to stop.")
    while True:
        query = input("\nAsk a question about your documents: ")
        if query.lower() in ["exit", "quit"]:
            break
        if not query.strip():
            continue

        # Get the response from the chain
        response = qa_chain.invoke(query)

        # Print the results
        print("\n--- Answer ---")
        print(response['result'])
        print('\n--- Sources ---')
        # Check if source documents were returned
        if response.get("source_documents"):
            for source in response["source_documents"]:
                print(f"- {source.metadata.get('source', 'Unknown')}")
        else:
            print("No source documents found for this response.")


if __name__ == "__main__":
    main()
