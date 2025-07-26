import os
import json
from datetime import datetime
from collections import defaultdict, Counter
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from langchain_community.llms import LlamaCpp
from langchain_community.chat_models import ChatLlamaCpp

# Configuration: Choose between local GPU pipeline or cloud-based pipeline
USE_CLOUD_PIPELINE = True  # Set to False to use local GPU pipeline

if USE_CLOUD_PIPELINE:
    # Import cloud-based RAG pipeline components
    from rag_pipeline_gpu_cloud import (
        create_rag_chain,
        HuggingFaceEmbeddings,
        Chroma,
        EMBEDDING_MODEL_NAME,
        PERSIST_DIRECTORY,
        logger
    )
    print("🌐 Using Cloud-based RAG Pipeline (OpenAI API)")
else:
    # Import local GPU RAG pipeline components
    from rag_pipeline_gpu import (
        create_rag_chain,
        HuggingFaceEmbeddings,
        Chroma,
        EMBEDDING_MODEL_NAME,
        PERSIST_DIRECTORY,
        logger
    )
    print("🖥️ Using Local GPU RAG Pipeline (LlamaCpp)")

# --- 1. Expanded Ground Truth Test Set ---
# Comprehensive dataset with various question types to test different aspects of the RAG pipeline
ground_truth_data = [
    {
        "question": "What are the key requirements for the ARPA-H TO-02 RSOPHO Support Request?",
        "ground_truth": "The key requirements include providing subject matter expertise in AI/ML, supporting the development of advanced algorithms for physiological data analysis, and ensuring robust data security measures for sensitive health information.",
        "category": "technical_requirements"
    },
    {
        "question": "What is the role of Amentum in the project?",
        "ground_truth": "Amentum's role involves program management, systems engineering, and providing logistical support to ensure the successful integration and deployment of the developed technologies.",
        "category": "organizational_roles"
    },
    {
        "question": "Who is Harley McKinley and what is his role in the ARPA-H project?",
        "ground_truth": "Harley McKinley is the ARPA-H Lead at Amentum, responsible for coordinating the ARPA-H team activities and staffing requirements for various technical objectives including TO3 support.",
        "category": "personnel"
    },
    {
        "question": "What specific expertise is needed for the ARPA-H TO3 support?",
        "ground_truth": "The ARPA-H TO3 support requires a distinct set of AI/ML expertise, including candidates who can be interviewed and assessed for their technical capabilities in artificial intelligence and machine learning applications.",
        "category": "technical_requirements"
    },
    {
        "question": "What are Amentum's key capabilities in digital engineering?",
        "ground_truth": "Amentum is a market leader in digital engineering with extensive experience, advanced tools, and strategic partnerships. They integrate system engineering models and utilize intelligent asset management with sensor-rich infrastructure.",
        "category": "capabilities"
    },
    {
        "question": "What is the staffing structure for the ARPA-H project?",
        "ground_truth": "The ARPA-H project has a staffing table with requirements for 9 slots, with specific AI/ML expertise requirements and coordination through Harley McKinley as the project lead.",
        "category": "organizational_structure"
    },
]


def run_evaluation():
    """Runs the RAG pipeline evaluation using Ragas."""
    pipeline_type = "Cloud-based (OpenAI)" if USE_CLOUD_PIPELINE else "Local GPU (LlamaCpp)"
    logger.info(f"Starting RAG pipeline evaluation using {pipeline_type}...")
    
    # Validate environment for cloud pipeline
    if USE_CLOUD_PIPELINE:
        # Check if we're in the right conda environment
        conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'unknown')
        logger.info(f"Current conda environment: {conda_env}")
        if conda_env != 'rag-llamacpp':
            logger.warning(f"Expected conda environment 'rag-llamacpp', but found '{conda_env}'")
            logger.warning("Please activate the correct environment: conda activate rag-llamacpp")

    # --- 2. Load the RAG Pipeline Components ---
    logger.info("Loading embedding model and vector store...")
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )

    if not os.path.exists(PERSIST_DIRECTORY):
        logger.error(f"Chroma database not found at {PERSIST_DIRECTORY}. Please run ingestion first.")
        return

    vectorstore = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedding_model)

    # Create the RAG chain, ensuring streaming is off for evaluation
    try:
        qa_chain = create_rag_chain(embedding_model, vectorstore, streaming=False)
        logger.info(f"Successfully created {pipeline_type} RAG chain")
    except Exception as e:
        logger.error(f"Failed to create RAG chain: {str(e)}")
        if USE_CLOUD_PIPELINE:
            logger.error("If using cloud pipeline, ensure OPENAI_API_KEY is properly set")
        raise

    # --- 3. Generate Answers and Contexts for the Test Set ---
    logger.info("Generating answers for the test set...")
    results = []
    for item in ground_truth_data:
        question = item["question"]
        logger.info(f"Processing question: {question}")
        response = qa_chain.invoke(question)
        results.append({
            "question": question,
            "answer": response['result'],
            "contexts": [doc.page_content for doc in response['source_documents']],
            "ground_truth": item["ground_truth"]
        })

    # --- 4. Prepare Dataset for Ragas ---
    dataset = Dataset.from_list(results)

    # --- 5. Enhanced Evaluation with Custom Metrics ---
    logger.info("Running enhanced RAG pipeline evaluation...")
    
    # Calculate enhanced metrics
    evaluation_metrics = calculate_enhanced_metrics(results)
    
    # Display comprehensive results
    display_comprehensive_results(results, evaluation_metrics)
    
    # Save evaluation results
    save_evaluation_results(results, evaluation_metrics)
    
    return results, evaluation_metrics

    # --- 6. Evaluation Complete ---
    logger.info("Enhanced evaluation complete.")
    print("\n=== RAG Pipeline Successfully Evaluated ===")
    print("The RAG pipeline is working correctly with:")
    print("✓ Document retrieval from ChromaDB")
    print("✓ HyDE (Hypothetical Document Embeddings)")
    print("✓ Cross-encoder re-ranking")
    print("✓ Local LLM answer generation")
    print("✓ Enhanced evaluation metrics and analysis")
    print("\nEvaluation results have been saved to 'evaluation_results.json'")
    print("Consider reviewing the detailed metrics for insights into pipeline performance.")


def calculate_enhanced_metrics(results):
    """Calculate enhanced evaluation metrics for the RAG pipeline."""
    metrics = {
        'total_questions': len(results),
        'avg_contexts_retrieved': sum(len(r['contexts']) for r in results) / len(results),
        'category_breakdown': defaultdict(int),
        'context_length_stats': [],
        'answer_length_stats': [],
        'retrieval_consistency': [],
        'timestamp': datetime.now().isoformat()
    }
    
    # Analyze by category
    for result in results:
        category = result.get('category', 'unknown')
        metrics['category_breakdown'][category] += 1
        
        # Context analysis
        context_lengths = [len(ctx) for ctx in result['contexts']]
        metrics['context_length_stats'].extend(context_lengths)
        
        # Answer analysis
        metrics['answer_length_stats'].append(len(result['answer']))
        
        # Check if contexts contain relevant keywords from the question
        question_words = set(result['question'].lower().split())
        context_relevance = []
        for ctx in result['contexts']:
            ctx_words = set(ctx.lower().split())
            overlap = len(question_words.intersection(ctx_words))
            context_relevance.append(overlap / len(question_words) if question_words else 0)
        metrics['retrieval_consistency'].extend(context_relevance)
    
    # Calculate statistics
    if metrics['context_length_stats']:
        metrics['avg_context_length'] = sum(metrics['context_length_stats']) / len(metrics['context_length_stats'])
        metrics['min_context_length'] = min(metrics['context_length_stats'])
        metrics['max_context_length'] = max(metrics['context_length_stats'])
    
    if metrics['answer_length_stats']:
        metrics['avg_answer_length'] = sum(metrics['answer_length_stats']) / len(metrics['answer_length_stats'])
        metrics['min_answer_length'] = min(metrics['answer_length_stats'])
        metrics['max_answer_length'] = max(metrics['answer_length_stats'])
    
    if metrics['retrieval_consistency']:
        metrics['avg_retrieval_relevance'] = sum(metrics['retrieval_consistency']) / len(metrics['retrieval_consistency'])
    
    return metrics


def display_comprehensive_results(results, metrics):
    """Display comprehensive evaluation results."""
    print("\n" + "=" * 80)
    print("                    COMPREHENSIVE RAG EVALUATION RESULTS")
    print("=" * 80)
    
    # Overall Statistics
    print(f"\n📊 OVERALL STATISTICS")
    print(f"   Total Questions Evaluated: {metrics['total_questions']}")
    print(f"   Average Contexts Retrieved: {metrics['avg_contexts_retrieved']:.2f}")
    print(f"   Average Answer Length: {metrics.get('avg_answer_length', 0):.0f} characters")
    print(f"   Average Context Length: {metrics.get('avg_context_length', 0):.0f} characters")
    print(f"   Average Retrieval Relevance: {metrics.get('avg_retrieval_relevance', 0):.2%}")
    
    # Category Breakdown
    print(f"\n📋 QUESTION CATEGORIES")
    for category, count in metrics['category_breakdown'].items():
        percentage = (count / metrics['total_questions']) * 100
        print(f"   {category.replace('_', ' ').title()}: {count} questions ({percentage:.1f}%)")
    
    # Detailed Results by Category
    print(f"\n📝 DETAILED RESULTS BY CATEGORY")
    print("-" * 80)
    
    category_results = defaultdict(list)
    for result in results:
        category = result.get('category', 'unknown')
        category_results[category].append(result)
    
    for category, cat_results in category_results.items():
        print(f"\n🏷️  {category.replace('_', ' ').title().upper()} ({len(cat_results)} questions)")
        print("-" * 60)
        
        for i, result in enumerate(cat_results, 1):
            print(f"\n   Question {i}: {result['question']}")
            print(f"   Ground Truth: {result['ground_truth'][:150]}{'...' if len(result['ground_truth']) > 150 else ''}")
            print(f"   Generated Answer: {result['answer'][:150]}{'...' if len(result['answer']) > 150 else ''}")
            print(f"   Contexts Retrieved: {len(result['contexts'])}")
            
            # Show top context snippet
            if result['contexts']:
                top_context = result['contexts'][0][:200] + "..." if len(result['contexts'][0]) > 200 else result['contexts'][0]
                print(f"   Top Context: {top_context}")
            print()
    
    print("=" * 80)


def save_evaluation_results(results, metrics):
    """Save evaluation results to a JSON file."""
    pipeline_version = 'cloud_openai_with_hybrid_retrieval' if USE_CLOUD_PIPELINE else 'gpu_enhanced_with_hyde_reranker'
    pipeline_type = 'Cloud-based (OpenAI API)' if USE_CLOUD_PIPELINE else 'Local GPU (LlamaCpp)'
    
    evaluation_data = {
        'metadata': {
            'evaluation_timestamp': metrics['timestamp'],
            'total_questions': metrics['total_questions'],
            'rag_pipeline_version': pipeline_version,
            'pipeline_type': pipeline_type,
            'conda_environment': os.environ.get('CONDA_DEFAULT_ENV', 'unknown')
        },
        'metrics': metrics,
        'detailed_results': results
    }
    
    filename = f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(evaluation_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Evaluation results saved to {filename}")

if __name__ == "__main__":
    run_evaluation()
