"""
Example usage of the TED Talks RAG system.
"""

from rag_system import TEDTalksRAG
import json

# Try to load environment variables from .env file (optional)
with open('utils/api_keys.json', 'r') as f:
    api_keys = json.load(f)
    PINECONE_API_KEY = api_keys['PINECONE_API_KEY']
    OPENAI_API_KEY = api_keys['OPENAI_API_KEY']

def main():
    # Initialize the RAG system
    rag = TEDTalksRAG(
        pinecone_api_key=PINECONE_API_KEY,
        openai_api_key=OPENAI_API_KEY
    )
    
    # Example 1: Index the talks (uncomment to run)
    # print("Indexing TED Talks...")
    # rag.load_and_index_talks("ted_talks_en.csv")
    # print("Indexing complete!")
    
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
        print(f"\nQuestion: {question}")
        print("-" * 80)
        
        result = rag.query(question, top_k=5)
        
        print(f"\nAnswer:\n{result['answer']}\n")
        print(f"Sources ({result['num_sources']}):")
        for i, source in enumerate(result['sources'], 1):
            print(f"  {i}. '{source['title']}' by {source['speaker']} "
                  f"(relevance: {source['relevance_score']:.3f})")
        
        print("\n" + "=" * 80)

if __name__ == "__main__":
    main()

