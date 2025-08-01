"""
This file is to generate a semantic search index for a directory of text files.

The parameters the user can set are:
    - ROOT_DIR: the path to the directory to index
    - DATA_DIR: the path to the data directory for the index
    - INDEX_NAME: the name of the index
    - NUM_WORKERS: the number of workers to use when building the index
    - BATCH_SIZE: the batch size to use when building the index
    - MAX_TOKENS: the maximum number of tokens to use when building the index

(c) Sam Nazari 2025
"""
import os
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.markdown import Markdown
from rich.progress import track
from rich import box
from llama_index.core import (
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Initialize rich console
console = Console()

# --- Configuration ---
STORAGE_DIR = "./storage"  # Directory where your index is stored
MODEL_PATH = "../models/Qwen3-30B-A3B-Instruct-2507-Q4_0.gguf"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

def main():
    """
    This script loads a pre-built LlamaIndex index and allows you to
    query it interactively from your terminal.
    """
    # Display welcome banner
    welcome_text = Text("🚀 LlamaIndex RAG Pipeline", style="bold magenta")
    welcome_panel = Panel(
        welcome_text,
        title="[bold blue]Welcome[/bold blue]",
        border_style="blue",
        box=box.ROUNDED
    )
    console.print(welcome_panel)
    
    # --- Check for Existing Index ---
    if not os.path.exists(STORAGE_DIR):
        error_panel = Panel(
            f"❌ Storage directory not found at '{STORAGE_DIR}'\n"
            "Please run 'prepare_docs.py' first to create the index.",
            title="[bold red]Error[/bold red]",
            border_style="red",
            box=box.ROUNDED
        )
        console.print(error_panel)
        return

    # --- Configure Global Settings ---
    with console.status("[bold green]Loading models...[/bold green]", spinner="dots"):
        console.print(f"[cyan]Loading LLM from:[/cyan] {MODEL_PATH}")
        Settings.llm = LlamaCPP(
            model_path=MODEL_PATH,
            temperature=0.1,
            max_new_tokens=512,
            context_window=3900,
            model_kwargs={"n_gpu_layers": -1}, # Use GPU
            verbose=True, # Set to True for more detailed LLM output
        )

        console.print(f"[cyan]Loading embedding model:[/cyan] {EMBEDDING_MODEL_NAME}")
        Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME)

    # --- Load the Index ---
    with console.status("[bold green]Loading index...[/bold green]", spinner="dots"):
        console.print(f"[cyan]💾 Loading index from '{STORAGE_DIR}'...[/cyan]")
        storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
        index = load_index_from_storage(storage_context)
        
    success_panel = Panel(
        "✅ Index loaded successfully!",
        title="[bold green]Success[/bold green]",
        border_style="green",
        box=box.ROUNDED
    )
    console.print(success_panel)

    # --- Create a Query Engine ---
    with console.status("[bold green]Creating query engine...[/bold green]", spinner="dots"):
        query_engine = index.as_query_engine(
            streaming=True,
            similarity_top_k=5,  # Retrieve top 5 most relevant chunks
            response_mode="tree_summarize"  # Better for handling multiple sources
        )
    
    ready_panel = Panel(
        "🎯 Query engine ready! You can now ask questions about your documents.",
        title="[bold green]Ready[/bold green]",
        border_style="green",
        box=box.ROUNDED
    )
    console.print(ready_panel)

    def display_sources(response):
        """Display the source documents used to generate the answer with rich formatting."""
        if hasattr(response, 'source_nodes') and response.source_nodes:
            # Create a table for sources
            sources_table = Table(
                title="📚 Source Documents",
                box=box.ROUNDED,
                title_style="bold blue",
                header_style="bold cyan"
            )
            sources_table.add_column("#", style="dim", width=3)
            sources_table.add_column("Document Path", style="yellow", min_width=30)
            sources_table.add_column("Preview", style="white", min_width=40)
            sources_table.add_column("Relevance", style="green", width=10)
            
            seen_sources = set()  # To avoid duplicate file paths
            row_count = 0
            
            for node in response.source_nodes:
                # Get the source file path
                source_path = node.node.metadata.get('file_path', 'Unknown source')
                
                # Only show unique sources
                if source_path not in seen_sources:
                    seen_sources.add(source_path)
                    row_count += 1
                    
                    # Show a snippet of the relevant content
                    content_preview = node.node.text[:100].replace('\n', ' ')
                    if len(node.node.text) > 100:
                        content_preview += "..."
                    
                    # Show relevance score if available
                    relevance_score = f"{node.score:.3f}" if hasattr(node, 'score') and node.score is not None else "N/A"
                    
                    sources_table.add_row(
                        str(row_count),
                        source_path,
                        content_preview,
                        relevance_score
                    )
            
            if row_count > 0:
                console.print(sources_table)
            else:
                no_sources_panel = Panel(
                    "No source documents found.",
                    title="[bold yellow]Sources[/bold yellow]",
                    border_style="yellow",
                    box=box.ROUNDED
                )
                console.print(no_sources_panel)
        else:
            no_info_panel = Panel(
                "No source information available",
                title="[bold yellow]Sources[/bold yellow]",
                border_style="yellow",
                box=box.ROUNDED
            )
            console.print(no_info_panel)

    # --- Interactive Query Loop ---
    while True:
        try:
            # Create an input prompt with style
            console.print()
            query_panel = Panel(
                "Type your question below (or 'exit' to quit)",
                title="[bold magenta]🤔 Ask a Question[/bold magenta]",
                border_style="magenta",
                box=box.ROUNDED
            )
            console.print(query_panel)
            
            query = console.input("[bold yellow]❓ Your question: [/bold yellow]")
            
            if query.lower() == 'exit':
                goodbye_panel = Panel(
                    "👋 Thank you for using the RAG pipeline!",
                    title="[bold green]Goodbye[/bold green]",
                    border_style="green",
                    box=box.ROUNDED
                )
                console.print(goodbye_panel)
                break
                
            if not query.strip():
                continue

            # Show thinking status
            with console.status("[bold green]🤖 Analyzing your question...[/bold green]", spinner="dots"):
                response = query_engine.query(query)

            # Display answer in a styled panel
            answer_panel = Panel(
                "",  # We'll stream the content
                title="[bold green]💡 Answer[/bold green]",
                border_style="green",
                box=box.ROUNDED
            )
            console.print(answer_panel)
            
            # Stream the response (no need for manual color formatting)
            response.print_response_stream()
            
            # Display source documents
            display_sources(response)
            
            # Add a separator
            console.print("\n" + "="*80, style="dim")

        except KeyboardInterrupt:
            console.print("\n[bold red]👋 Exiting...[/bold red]")
            break

if __name__ == "__main__":
    main()