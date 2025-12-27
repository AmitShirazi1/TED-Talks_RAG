"""
Example usage of the TED Talks RAG system with parameter testing.
Tests model performance on example queries with different parameter combinations.
"""

from rag_system import TEDTalksRAG
import json
from rag_system import print_result
from utils.consts import DEFAULT_RETRIEVE_TOP_K, CSV_FILE_PATH, EMBEDDING_MODEL, CHAT_MODEL
from typing import List, Dict, Any
import time

# Load API keys
with open('utils/api_keys.json', 'r') as f:
    api_keys = json.load(f)
    PINECONE_API_KEY = api_keys['PINECONE_API_KEY']
    LLMOD_API_KEY = api_keys['LLMOD_API_KEY']

# Example queries to test
EXAMPLE_QUERIES = [
    # Questions about the first few transcripts
    "What did Al Gore say about climate change solutions?",
    "What are the main themes in Hans Rosling's talk about statistics?",
    "What did David Pogue say about simplicity in technology?",
    "What environmental justice issues did Majora Carter discuss?",
    # Questions from assignment PDF
    "Find a TED talk that discusses overcoming fear or anxiety. Provide the title and speaker.",
    "Which TED talk focuses on education or learning? Return a list of exactly 3 talk titles.",
    "Find a TED talk where the speaker talks about technology improving people's lives. Provide the title and a short summary of the key idea.",
    "I'm looking for a TED talk about climate change and what individuals can do in their daily lives. Which talk would you recommend?"
]

# Parameter configurations to test
PARAMETER_CONFIGS = [
    {
        "name": "Default",
        "temperature": 0.1,
        "top_k": DEFAULT_RETRIEVE_TOP_K,
        "chat_model": CHAT_MODEL,
        "embedding_model": EMBEDDING_MODEL,
    },
    {
        "name": "Low Temperature",
        "temperature": 0.0,
        "top_k": DEFAULT_RETRIEVE_TOP_K,
        "chat_model": CHAT_MODEL,
        "embedding_model": EMBEDDING_MODEL,
    },
    {
        "name": "High Temperature",
        "temperature": 0.5,
        "top_k": DEFAULT_RETRIEVE_TOP_K,
        "chat_model": CHAT_MODEL,
        "embedding_model": EMBEDDING_MODEL,
    },
    {
        "name": "More Context (top_k=10)",
        "temperature": 0.1,
        "top_k": 10,
        "chat_model": CHAT_MODEL,
        "embedding_model": EMBEDDING_MODEL,
    },
    {
        "name": "Less Context (top_k=3)",
        "temperature": 0.1,
        "top_k": 3,
        "chat_model": CHAT_MODEL,
        "embedding_model": EMBEDDING_MODEL,
    },
    # Add more configurations as needed
    # {
    #     "name": "Custom Config",
    #     "temperature": 0.2,
    #     "top_k": 7,
    #     "chat_model": CHAT_MODEL,
    #     "embedding_model": EMBEDDING_MODEL,
    # },
]


def test_query_with_config(
    rag: TEDTalksRAG,
    question: str,
    config: Dict[str, Any],
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Test a single query with a specific parameter configuration.
    
    Args:
        rag: Initialized RAG system
        question: Query to test
        config: Parameter configuration dictionary
        verbose: Whether to print results
        
    Returns:
        Dictionary with results and metadata
    """
    start_time = time.time()
    
    # Query with the specified top_k
    result = rag.query(question, top_k=config["top_k"])
    
    elapsed_time = time.time() - start_time
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"Config: {config['name']}")
        print(f"Question: {question}")
        print(f"{'='*80}")
        print(f"Answer:\n{result['answer']}\n")
        print(f"Sources ({result['num_sources']}):")
        for i, source in enumerate(result['sources'], 1):
            chunk_num = source.get('chunk_index', 'N/A')
            print(f"  {i}. '{source['title']}' by {source['speaker']}, "
                  f"chunk {chunk_num} (relevance: {source['relevance_score']:.3f})")
        print(f"\nTime: {elapsed_time:.2f}s")
        print(f"{'='*80}\n")
    
    return {
        "config_name": config["name"],
        "question": question,
        "answer": result["answer"],
        "num_sources": result["num_sources"],
        "sources": result["sources"],
        "elapsed_time": elapsed_time,
        "parameters": {
            "temperature": config["temperature"],
            "top_k": config["top_k"],
        }
    }


def test_all_configs(
    questions: List[str] = None,
    configs: List[Dict[str, Any]] = None,
    compare_mode: bool = True
):
    """
    Test all queries with all parameter configurations.
    
    Args:
        questions: List of questions to test (default: EXAMPLE_QUERIES)
        configs: List of parameter configurations (default: PARAMETER_CONFIGS)
        compare_mode: If True, compare results side-by-side. If False, show full details.
    """
    if questions is None:
        questions = EXAMPLE_QUERIES
    if configs is None:
        configs = PARAMETER_CONFIGS
    
    print("=" * 80)
    print("TED Talks RAG System - Parameter Testing")
    print("=" * 80)
    print(f"Testing {len(questions)} queries with {len(configs)} configurations")
    print("=" * 80)
    
    # Initialize RAG system with first config (will reuse for all queries)
    # Note: We'll need to reinitialize if we want to test different models
    base_config = configs[0]
    rag = TEDTalksRAG(
        pinecone_api_key=PINECONE_API_KEY,
        llmod_api_key=LLMOD_API_KEY,
        temperature=base_config["temperature"],
        chat_model=base_config["chat_model"],
        embedding_model=base_config["embedding_model"]
    )
    
    all_results = []
    
    for question_idx, question in enumerate(questions, 1):
        print(f"\n\n{'#'*80}")
        print(f"QUESTION {question_idx}/{len(questions)}")
        print(f"{'#'*80}")
        print(f"Query: {question}\n")
        
        question_results = []
        
        for config in configs:
            # Reinitialize RAG if temperature or model changed
            if (config["temperature"] != rag.temperature or 
                config["chat_model"] != rag.chat_model or
                config["embedding_model"] != rag.embedding_model):
                rag = TEDTalksRAG(
                    pinecone_api_key=PINECONE_API_KEY,
                    llmod_api_key=LLMOD_API_KEY,
                    temperature=config["temperature"],
                    chat_model=config["chat_model"],
                    embedding_model=config["embedding_model"]
                )
            
            result = test_query_with_config(
                rag, 
                question, 
                config, 
                verbose=not compare_mode
            )
            question_results.append(result)
        
        all_results.append({
            "question": question,
            "results": question_results
        })
        
        # Show comparison if in compare mode
        if compare_mode:
            print_comparison(question, question_results)
    
    return all_results


def print_comparison(question: str, results: List[Dict[str, Any]]):
    """Print a side-by-side comparison of results for different configurations."""
    print(f"\n{'='*80}")
    print(f"COMPARISON: {question}")
    print(f"{'='*80}")
    
    for result in results:
        config_name = result["config_name"]
        answer = result["answer"]
        num_sources = result["num_sources"]
        elapsed_time = result["elapsed_time"]
        params = result["parameters"]
        
        # Truncate answer for comparison view
        answer_preview = answer[:200] + "..." if len(answer) > 200 else answer
        
        print(f"\n{'-'*80}")
        print(f"Config: {config_name}")
        print(f"  Temperature: {params['temperature']}, Top-K: {params['top_k']}")
        print(f"  Time: {elapsed_time:.2f}s, Sources: {num_sources}")
        print(f"  Answer Preview: {answer_preview}")
        print(f"{'-'*80}")


def test_single_config(
    questions: List[str] = None,
    config_name: str = "Default"
):
    """
    Test all queries with a single parameter configuration.
    
    Args:
        questions: List of questions to test (default: EXAMPLE_QUERIES)
        config_name: Name of the configuration to use from PARAMETER_CONFIGS
    """
    if questions is None:
        questions = EXAMPLE_QUERIES
    
    # Find the config
    config = next((c for c in PARAMETER_CONFIGS if c["name"] == config_name), None)
    if config is None:
        print(f"Configuration '{config_name}' not found. Available: {[c['name'] for c in PARAMETER_CONFIGS]}")
        return
    
    print("=" * 80)
    print(f"TED Talks RAG System - Testing with '{config_name}' Configuration")
    print("=" * 80)
    print(f"Parameters: Temperature={config['temperature']}, Top-K={config['top_k']}")
    print("=" * 80)
    
    # Initialize RAG system
    rag = TEDTalksRAG(
        pinecone_api_key=PINECONE_API_KEY,
        llmod_api_key=LLMOD_API_KEY,
        temperature=config["temperature"],
        chat_model=config["chat_model"],
        embedding_model=config["embedding_model"]
    )
    
    for question in questions:
        test_query_with_config(rag, question, config, verbose=True)


def main():
    """
    Main function. Modify this to change testing behavior.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Test TED Talks RAG with different parameters")
    parser.add_argument(
        "--mode",
        choices=["compare", "single", "all"],
        default="compare",
        help="Testing mode: 'compare' (side-by-side), 'single' (one config with full details), 'all' (all configs with full details)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="Default",
        help="Configuration name to use in 'single' mode (default: 'Default')"
    )
    parser.add_argument(
        "--queries",
        type=int,
        default=None,
        help="Number of queries to test (default: all)"
    )
    
    args = parser.parse_args()
    
    # Select queries to test
    questions = EXAMPLE_QUERIES
    if args.queries:
        questions = EXAMPLE_QUERIES[:args.queries]
    
    if args.mode == "compare":
        # Compare mode: show side-by-side results
        test_all_configs(questions=questions, compare_mode=True)
    elif args.mode == "single":
        # Single config mode: show full details for one configuration
        test_single_config(questions=questions, config_name=args.config)
    elif args.mode == "all":
        # All configs mode: show full details for all configurations
        test_all_configs(questions=questions, compare_mode=False)


if __name__ == "__main__":
    main()
