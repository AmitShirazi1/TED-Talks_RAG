"""
TED Talks RAG System
A Retrieval-Augmented Generation system for querying TED Talks.
"""

import os
import pandas as pd
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from typing import List, Dict
from utils.consts import CSV_FILE_PATH, INDEX_NAME, EMBEDDING_MODEL, CHAT_MODEL, DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP, DEFAULT_RETRIEVE_TOP_K
import json


class TEDTalksRAG:
    """
    RAG system for TED Talks that:
    1. Stores talk transcripts in Pinecone vector database
    2. Retrieves relevant chunks based on user queries
    3. Answers questions using only retrieved TED Talk data
    """
    
    def __init__(self, 
                 pinecone_api_key: str,
                 openai_api_key: str,
                 index_name: str = INDEX_NAME,
                 embedding_model: str = EMBEDDING_MODEL,
                 chat_model: str = CHAT_MODEL,
                 temperature: float = 0.1):
        """
        Initialize the RAG system.
        
        Args:
            pinecone_api_key: Pinecone API key
            openai_api_key: OpenAI API key
            index_name: Name of the Pinecone index
            embedding_model: OpenAI embedding model to use
            chat_model: OpenAI chat model to use
        """
        self.pinecone_api_key = pinecone_api_key
        self.openai_api_key = openai_api_key
        self.index_name = index_name
        self.embedding_model = embedding_model
        self.chat_model = chat_model
        self.temperature = temperature
        
        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=openai_api_key)
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=pinecone_api_key)
        
        # Connect to or create index
        self._setup_index()
    

    def _setup_index(self):
        """ Create or connect to Pinecone index. """
        # Check if index exists
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]
        print(f"Existing indexes: {existing_indexes}")
        if self.index_name not in existing_indexes:
            # Create new index
            self.pc.create_index(
                name=self.index_name,
                dimension=1536,  # RPRTHPB-text-embedding-3-small's dimension
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            print(f"Created new index: {self.index_name}")
        else:
            print(f"Connected to existing index: {self.index_name}")
        
        self.index = self.pc.Index(self.index_name)
    

    def _chunk_text(self, text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
            chunk_size: Maximum size of each chunk (in characters)
            overlap: Number of characters to overlap between chunks
        
        Returns:
            List of text chunks
        """
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings
                last_period = chunk.rfind('. ')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > chunk_size * 0.5:  # Only break if we're past halfway
                    chunk = chunk[:break_point + 1]
                    end = start + break_point + 1
            
            chunks.append(chunk.strip())
            start = end - overlap  # Overlap for context
        
        return chunks
    

    def _create_embedding(self, text: str) -> List[float]:
        """Create embedding for text using OpenAI."""
        response = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return response.data[0].embedding
    

    def load_and_index_talks(self, csv_path: str, batch_size: int = 100):
        """
        Load TED Talks from CSV and index them in Pinecone.
        
        Args:
            csv_path: Path to the CSV file
            batch_size: Number of vectors to upload in each batch
        """
        print(f"Loading talks from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        print(f"Found {len(df)} talks")
        
        vectors_to_upsert = []
        total_chunks = 0
        
        for idx, row in df.iterrows():
            talk_id = str(row['talk_id'])
            title = str(row.get('title', ''))
            speaker = str(row.get('speaker_1', ''))
            topics = str(row.get('topics', ''))
            transcript = str(row.get('transcript', ''))
            description = str(row.get('description', ''))
            
            if not transcript or transcript == 'nan':
                print(f"Skipping talk {talk_id}: No transcript")
                continue
            
            # Chunk the transcript
            chunks = self._chunk_text(transcript)
            
            for chunk_idx, chunk in enumerate(chunks):
                # Enrich chunk with metadata
                enriched_chunk = f"Title: {title}\nSpeaker: {speaker}\nTopics: {topics}\n\nTranscript excerpt:\n{chunk}"
                
                # Create embedding
                embedding = self._create_embedding(enriched_chunk)
                
                # Create unique ID for this chunk
                chunk_id = f"{talk_id}_chunk_{chunk_idx}"
                
                # Prepare metadata
                metadata = {
                    "talk_id": talk_id,
                    "title": title,
                    "speaker": speaker,
                    "topics": topics,
                    "chunk_index": chunk_idx,
                    "chunk_text": chunk,  # Store original chunk without enrichment
                    "description": description
                }
                
                vectors_to_upsert.append({
                    "id": chunk_id,
                    "values": embedding,
                    "metadata": metadata
                })
                
                total_chunks += 1
                
                # Upload in batches
                if len(vectors_to_upsert) >= batch_size:
                    self.index.upsert(vectors=vectors_to_upsert)
                    print(f"Uploaded batch: {total_chunks} chunks indexed so far...")
                    vectors_to_upsert = []
            
            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{len(df)} talks...")
        
        # Upload remaining vectors
        if vectors_to_upsert:
            self.index.upsert(vectors=vectors_to_upsert)
            print(f"Uploaded final batch: {total_chunks} chunks indexed")
        
        print(f"\nIndexing complete! Total chunks indexed: {total_chunks}")
    

    def retrieve_relevant_chunks(self, query: str, top_k: int = DEFAULT_RETRIEVE_TOP_K) -> List[Dict]:
        """
        Retrieve relevant chunks from the vector database.
        
        Args:
            query: User's query
            top_k: Number of chunks to retrieve
        
        Returns:
            List of relevant chunks with metadata
        """
        # Create query embedding
        query_embedding = self._create_embedding(query)
        
        # Search in Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        # Format results
        chunks = []
        for match in results.matches:
            chunks.append({
                "score": match.score,
                "talk_id": match.metadata.get("talk_id"),
                "title": match.metadata.get("title"),
                "speaker": match.metadata.get("speaker"),
                "topics": match.metadata.get("topics"),
                "chunk_text": match.metadata.get("chunk_text"),
                "description": match.metadata.get("description")
            })
        
        return chunks
    

    def query(self, user_question: str, top_k: int = DEFAULT_RETRIEVE_TOP_K) -> Dict:
        """
        Answer a user question using RAG.
        
        Args:
            user_question: The user's question
            top_k: Number of chunks to retrieve
        
        Returns:
            Dictionary with answer and source information
        """
        # Retrieve relevant chunks
        print(f"Retrieving relevant chunks for: '{user_question}'...")
        chunks = self.retrieve_relevant_chunks(user_question, top_k=top_k)
        
        if not chunks:
            return {
                "answer": "I couldn't find any relevant information in the TED Talks database.",
                "sources": []
            }
        
        # Build context from retrieved chunks
        context_parts = []
        sources = []
        
        for i, chunk in enumerate(chunks, 1):
            context_part = f"[Source {i}]\n"
            context_part += f"Title: {chunk['title']}\n"
            context_part += f"Speaker: {chunk['speaker']}\n"
            if chunk.get('topics'):
                context_part += f"Topics: {chunk['topics']}\n"
            context_part += f"Transcript excerpt:\n{chunk['chunk_text']}\n"
            context_parts.append(context_part)
            
            sources.append({
                "title": chunk['title'],
                "speaker": chunk['speaker'],
                "talk_id": chunk['talk_id'],
                "relevance_score": chunk['score']
            })
        
        context = "\n\n".join(context_parts)
        
        # Create prompt for LLM
        system_prompt = """
            You are a TED Talk assistant that answers questions strictly and 
            only based on the TED dataset context provided to you (metadata 
            and transcript passages). You must not use any external 
            knowledge, the open internet, or information that is not explicitly 
            contained in the retrieved context. If the answer cannot be 
            determined from the provided context, respond: "I don't know 
            based on the provided TED data." Always explain your answer 
            using the given context, quoting or paraphrasing the relevant 
            transcript or metadata when helpful.

            Please answer the question based ONLY on the information provided in the context above. If the context doesn't contain enough information, please say so explicitly.
            """

        user_prompt = f"""Question: {user_question}
            Context from TED Talks database:
            {context}
            """

        # Get answer from LLM
        print("Generating answer...")
        response = self.openai_client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=self.temperature  # Low temperature for more factual responses
        )
        
        answer = response.choices[0].message.content
        
        return {
            "answer": answer,
            "sources": sources,
            "num_sources": len(sources)
        }
    

    def get_index_stats(self) -> Dict:
        """Get statistics about the indexed data."""
        stats = self.index.describe_index_stats()
        return {
            "total_vectors": stats.total_vector_count,
            "dimension": stats.dimension,
            "index_fullness": stats.index_fullness
        }


def print_result(rag: TEDTalksRAG, question: str, top_k: int = DEFAULT_RETRIEVE_TOP_K):
    """Print the result of the query in a readable format."""
    print(f"\nQuestion: {question}\n")
    result = rag.query(question, top_k=top_k)
    print(f"Answer:\n{result['answer']}\n")
    print(f"Sources ({result['num_sources']}):")
    for i, source in enumerate(result['sources'], 1):
        print(f"  {i}. '{source['title']}' by {source['speaker']} (relevance: {source['relevance_score']:.3f})\n")


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="TED Talks RAG System")
    parser.add_argument("--mode", choices=["index", "query"], required=True,
                       help="Mode: 'index' to load data, 'query' to ask questions")
    parser.add_argument("--csv", default=CSV_FILE_PATH,
                       help=f"Path to CSV file (for index mode, default: {CSV_FILE_PATH})")
    parser.add_argument("--question", help="Question to ask (for query mode)")
    parser.add_argument("--top-k", type=int, default=DEFAULT_RETRIEVE_TOP_K,
                       help=f"Number of chunks to retrieve (default: {DEFAULT_RETRIEVE_TOP_K})")
    parser.add_argument("--temperature", type=float, default=0.1,
                       help=f"Temperature for the LLM (default: 0.1)")
    
    args = parser.parse_args()
    
    # Get API keys from environment
    with open('utils/api_keys.json', 'r') as f:
        api_keys = json.load(f)
        PINECONE_API_KEY = api_keys['PINECONE_API_KEY']
        OPENAI_API_KEY = api_keys['OPENAI_API_KEY']
    
    # Initialize RAG system
    rag = TEDTalksRAG(
        pinecone_api_key=PINECONE_API_KEY,
        openai_api_key=OPENAI_API_KEY,
        temperature=args.temperature
    )
    
    if args.mode == "index":
        # Index the talks
        rag.load_and_index_talks(args.csv)
        stats = rag.get_index_stats()
        print(f"\nIndex statistics: {stats}")
    
    elif args.mode == "query":
        if not args.question:
            # Interactive mode
            print("TED Talks RAG System - Interactive Mode")
            print("Type 'exit' to quit\n")
            
            while True:
                question = input("Your question: ").strip()
                if not question:
                    continue
                if question.lower() in ['exit', 'quit', 'q']:
                    break
                
                print_result(rag, question, args.top_k)
                print("\n" + "="*80 + "\n")
        else:
            # Single query mode
            question = args.question.strip()
            if not question:
                print("Question is empty. Please provide a question.")
                return

            print_result(rag, question, args.top_k)

if __name__ == "__main__":
    main()

