"""
Example usage of the TED Talks RAG system.
"""

from rag_system import TEDTalksRAG
import json
from rag_system import print_result
from utils.consts import DEFAULT_RETRIEVE_TOP_K, CSV_FILE_PATH

# Try to load environment variables from .env file (optional)
with open('utils/api_keys.json', 'r') as f:
    api_keys = json.load(f)
    PINECONE_API_KEY = api_keys['PINECONE_API_KEY']
    LLMOD_API_KEY = api_keys['LLMOD_API_KEY']

def main():
    # Initialize the RAG system
    rag = TEDTalksRAG(
        pinecone_api_key=PINECONE_API_KEY,
        llmod_api_key=LLMOD_API_KEY
    )
    
    # Example 1: Index the talks
    print("Indexing TED Talks...")
    rag.load_and_index_talks(CSV_FILE_PATH)
    print("Indexing complete!")
    
    # Example 2: Query the system
    questions = [
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
    
    print("=" * 80)
    print("TED Talks RAG System - Example Queries")
    print("=" * 80)
    
    for question in questions:
        print_result(rag, question, DEFAULT_RETRIEVE_TOP_K)
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()

