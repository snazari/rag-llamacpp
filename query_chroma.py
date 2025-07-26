import argparse
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# --- Configuration ---
PERSIST_DIRECTORY = "./chroma_db"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

def main():
    """
    Loads an existing Chroma vector store and performs a similarity search or inspection.
    """
    # --- 1. Set up command-line argument parsing with subparsers ---
    parser = argparse.ArgumentParser(description="Query and inspect a local ChromaDB vector store.")
    subparsers = parser.add_subparsers(dest="command", help="Available commands", required=True)

    # 'info' command
    subparsers.add_parser("info", help="Display diagnostic information about the database.")

    # 'peek' command
    peek_parser = subparsers.add_parser("peek", help="Display the first N documents from the database.")
    peek_parser.add_argument("n", type=int, nargs='?', default=5, help="Number of documents to display (default: 5).")

    # 'list-metadata' command
    subparsers.add_parser("list-metadata", help="List unique metadata keys from a sample of documents.")

    # 'search' command
    search_parser = subparsers.add_parser("search", help="Perform a similarity search.")
    search_parser.add_argument("query_text", type=str, help="The text to search for.")
    search_parser.add_argument("-k", type=int, default=4, help="The number of documents to return.")
    search_parser.add_argument("--where", type=str, help="A JSON string for a metadata filter (e.g., '{\"source\": \"/path/to/doc.pdf\"}').")

    args = parser.parse_args()

    # Check if the database directory exists
    if not os.path.exists(PERSIST_DIRECTORY):
        print(f"Error: Chroma database directory not found at '{PERSIST_DIRECTORY}'")
        print("Please run your main RAG pipeline script first to create the database.")
        return

    # --- 2. Load the Embedding Model ---
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )
    print("Embedding model loaded.")

    # --- 3. Load the Existing Vector Store ---
    print(f"Loading vector store from: {PERSIST_DIRECTORY}...")
    vectorstore = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embedding_model
    )
    print("Vector store loaded successfully.")

    # --- 4. Perform Actions based on the command ---
    import json

    if args.command == "info":
        print("\n--- Database Information ---")
        try:
            collection_count = vectorstore._collection.count()
            print(f"Total documents in the database: {collection_count}")
        except Exception as e:
            print(f"Could not retrieve document count: {e}")
        print("-" * 28)

    elif args.command == "peek":
        print(f"\n--- Peeking at first {args.n} documents ---")
        try:
            docs = vectorstore.get(limit=args.n)
            for i, doc_content in enumerate(docs['documents']):
                print(f"--- Document {i+1} ---")
                metadata = docs['metadatas'][i]
                print(f"Metadata: {metadata}")
                content = doc_content.replace('\n', ' ').strip()
                print(f"Content: {content[:500]}...")
                print("-" * 20)
        except Exception as e:
            print(f"Could not retrieve documents: {e}")
        print("-" * 35)

    elif args.command == "list-metadata":
        print("\n--- Unique Metadata Keys (from a sample of 100 documents) ---")
        try:
            docs = vectorstore.get(limit=100)  # Sample 100 docs
            unique_keys = set()
            for metadata in docs['metadatas']:
                unique_keys.update(metadata.keys())
            if unique_keys:
                for key in sorted(list(unique_keys)):
                    print(f"- {key}")
            else:
                print("No metadata keys found in the sample.")
        except Exception as e:
            print(f"Could not retrieve metadata: {e}")
        print("-" * 60)

    elif args.command == "search":
        where_filter = None
        if args.where:
            try:
                where_filter = json.loads(args.where)
                print(f"\nUsing search filter: {where_filter}")
            except json.JSONDecodeError:
                print("Error: Invalid JSON format for --where filter. Please use a valid JSON string.")
                return

        print(f"\nSearching for: '{args.query_text}'...")
        results = vectorstore.similarity_search(args.query_text, k=args.k, filter=where_filter)

        if not results:
            print("No results found.")
        else:
            print(f"\nFound {len(results)} relevant documents (top {args.k}):")
            print("-" * 50)
            for i, doc in enumerate(results):
                print(f"--- Result {i+1} ---")
                source = doc.metadata.get('source', 'Unknown')
                print(f"Source: {source}")
                content = doc.page_content.replace('\n', ' ').strip()
                print(f"Content: {content[:500]}...")
                print("-" * 50)

if __name__ == "__main__":
    main()
