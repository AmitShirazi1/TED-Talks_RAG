"""
TED Talks RAG System
A Retrieval-Augmented Generation system for querying TED Talks.
"""

import os
import re
import pandas as pd
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from typing import List, Dict, Optional
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
                 llmod_api_key: str,
                 index_name: str = INDEX_NAME,
                 embedding_model: str = EMBEDDING_MODEL,
                 chat_model: str = CHAT_MODEL,
                 temperature: float = 0.1):
        """
        Initialize the RAG system.
        
        Args:
            pinecone_api_key: Pinecone API key
            llmod_api_key: LLMod API key
            index_name: Name of the Pinecone index
            embedding_model: LLMod embedding model to use
            chat_model: LLMod chat model to use
        """
        self.pinecone_api_key = pinecone_api_key
        self.llmod_api_key = llmod_api_key
        self.index_name = index_name
        self.embedding_model = embedding_model
        self.chat_model = chat_model
        self.temperature = temperature
        
        # Initialize OpenAI client
        self.llmod_client = OpenAI(api_key=llmod_api_key, base_url="https://api.llmod.ai")
        
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
        response = self.llmod_client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return response.data[0].embedding
    

    def load_and_index_talks(self, csv_path: str, batch_size: int = 100, force_reindex: bool = False):
        """
        Load TED Talks from CSV and index them in Pinecone.
        
        Args:
            csv_path: Path to the CSV file
            batch_size: Number of vectors to upload in each batch
            force_reindex: If True, reindex even if vectors already exist (default: False)
        """
        # Check if index already has vectors
        if not force_reindex:
            stats = self.get_index_stats()
            if stats["total_vectors"] > 0:
                print(f"Index already contains {stats['total_vectors']} vectors. Skipping indexing.")
                print("To force re-indexing, call with force_reindex=True")
                return
        
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
                # Embed only transcript chunk + title (keep speaker/topics as metadata only)
                # This prevents "Topics:" strings from dominating similarity for broad queries
                text_to_embed = f"{title}\n\n{chunk}" if title else chunk
                
                # Create embedding
                embedding = self._create_embedding(text_to_embed)
                
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
        
        # For fact retrieval queries, retrieve more chunks to ensure we have enough unique talks
        retrieve_k = top_k * 2 if self._is_fact_retrieval_query(query) else top_k
        
        # Search in Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=retrieve_k,
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
    
    def _is_fact_retrieval_query(self, query: str) -> bool:
        """
        Detect if the query is asking for a specific TED talk (fact retrieval).
        
        Args:
            query: User's query
        
        Returns:
            True if query is asking for a specific talk/talks
        """
        # If a number is extracted from the query AND it's talk-related, it's likely asking for specific talks
        # This prevents false positives like "top 5 reasons..." from being classified as fact retrieval
        n = self._extract_number_from_query(query)
        if n is not None and ("talk" in query.lower() or "ted" in query.lower() or "title" in query.lower()):
            return True
        
        query_lower = query.lower()
        fact_indicators = [
            "find a ted talk",
            "find one ted talk",
            "which ted talk",
            "what ted talk",
            "name a ted talk",
            "give me a ted talk",
            "give me",
            "recommend a ted talk",
            "suggest a ted talk",
            "provide the title",
            "return a list of",
            "list of exactly",
            "list",
            "talks about",
            "find a talk",
            "which talk",
            "what talk"
        ]
        
        return any(indicator in query_lower for indicator in fact_indicators)
    
    def _is_summary_extraction_query(self, query: str) -> bool:
        """
        Detect if the query is asking for a key idea summary extraction.
        
        Args:
            query: User's query
        
        Returns:
            True if query is asking for a summary of a talk's key idea
        """
        q = query.lower()
        indicators = [
            "short summary",
            "key idea",
            "main idea",
            "summarize",
            "summary of the key idea",
            "brief summary",
            "summary",
            "provide the title and a short summary",
        ]
        # Also treat "find a talk ... provide title and summary" as summary extraction
        # Stronger check: if query contains both "find a talk" and "summary", always treat as summary extraction
        has_find_talk = "find a talk" in q or "find a ted talk" in q
        has_summary = "summary" in q
        
        if has_find_talk and has_summary:
            return True
        
        return any(ind in q for ind in indicators) and ("title" in q or "talk" in q)
    
    def _is_recommendation_query(self, query: str) -> bool:
        """
        Detect if the query is asking for a recommendation with evidence-based justification.
        
        Args:
            query: User's query
        
        Returns:
            True if query is asking for a talk recommendation
        """
        q = query.lower()
        indicators = [
            "recommend",
            "which talk would you recommend",
            "what talk would you recommend",
            "suggest",
            "i'm looking for a ted talk",
            "i am looking for a ted talk",
            "can you recommend",
            "any recommendations",
        ]
        return any(ind in q for ind in indicators)
    
    def _extract_number_from_query(self, query: str) -> Optional[int]:
        """
        Extract a specific number from the query if present (e.g., "exactly 3 talks").
        
        Args:
            query: User's query
        
        Returns:
            Number extracted, or None if not found
        """
        query_lower = query.lower()
        
        NUMBER_WORDS = {
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
            "nine": 9,
            "ten": 10,
        }

        word_patterns = [
            r'\b(one|two|three|four|five|six|seven|eight|nine|ten)\s+talks?\b',
            r'\bexactly\s+(one|two|three|four|five|six|seven|eight|nine|ten)\b',
            r'give\s+me\s+(one|two|three|four|five|six|seven|eight|nine|ten)\s+talks?',
            r'list\s+(one|two|three|four|five|six|seven|eight|nine|ten)\s+talks?',
        ]

        for pattern in word_patterns:
            match = re.search(pattern, query_lower)
            if match:
                return NUMBER_WORDS[match.group(1)]


        # Look for patterns like "exactly 3", "3 talks", "list of 3"
        # Also catch "give me 2 talks", "list three talks", etc.
        # Note: We don't match "a ted talk" here to avoid false positives on QA queries
        # like "In a TED talk, what does X say..." - rely on clearer indicators in _is_fact_retrieval_query()
        patterns = [
            r'exactly\s+(\d+)',
            r'list\s+of\s+exactly\s+(\d+)',
            r'(\d+)\s+ted\s+talks',
            r'(\d+)\s+talks',
            r'give\s+me\s+(\d+)\s+talks?',
            r'list\s+(\d+)\s+talks?',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query_lower)
            if match:
                if match.lastindex:
                    return int(match.group(1))
                else:
                    return 1
        
        return None
    
    def _deduplicate_chunks_by_talk(self, chunks: List[Dict]) -> List[Dict]:
        """
        Deduplicate chunks by talk_id, keeping only the highest scoring chunk per talk.
        
        Args:
            chunks: List of chunks (may contain multiple chunks from same talk)
        
        Returns:
            Deduplicated list with one chunk per talk (highest score)
        """
        talk_dict = {}
        
        for chunk in chunks:
            talk_id = chunk.get("talk_id")
            if talk_id:
                # Keep the chunk with the highest score for each talk
                if talk_id not in talk_dict or chunk["score"] > talk_dict[talk_id]["score"]:
                    talk_dict[talk_id] = chunk
        
        # Return chunks sorted by score (descending)
        deduplicated = list(talk_dict.values())
        deduplicated.sort(key=lambda x: x["score"], reverse=True)
        
        return deduplicated
    
    def _select_top_talk_with_chunks(self, chunks: List[Dict], max_chunks: int = 2) -> List[Dict]:
        """
        Pick best talk_id by top score, then return up to max_chunks chunks from that talk.
        
        Args:
            chunks: List of chunks (may contain multiple chunks from same talk)
            max_chunks: Maximum number of chunks to return from the best talk
        
        Returns:
            List of chunks from the best talk (up to max_chunks), sorted by score
        """
        if not chunks:
            return []

        # Find best talk_id by max score
        best = max(chunks, key=lambda c: c["score"])
        best_id = best["talk_id"]

        # Take chunks from that talk, sorted by score
        same_talk = [c for c in chunks if c["talk_id"] == best_id]
        same_talk.sort(key=lambda c: c["score"], reverse=True)
        return same_talk[:max_chunks]

    def _no_chunks_found_response(self) -> Dict:
        """
        Return a response when no chunks are found.
        """
        return {
            "answer": "I don't know based on the provided TED data.",
            "sources": [],
            "num_sources": 0,
            "context": [],
            "augmented_prompt": {
                "System": "",
                "User": ""
            }
        }
    

    def query(self, user_question: str, top_k: int = DEFAULT_RETRIEVE_TOP_K) -> Dict:
        """
        Answer a user question using RAG.
        
        Args:
            user_question: The user's question
            top_k: Number of chunks to retrieve
        
        Returns:
            Dictionary with answer and source information
        """
        # Check query types - priority order: summary > recommendation > fact retrieval
        is_summary_query = self._is_summary_extraction_query(user_question)
        is_recommendation_query = (not is_summary_query) and self._is_recommendation_query(user_question)
        is_fact_query = (not is_summary_query) and (not is_reco_query) and self._is_fact_retrieval_query(user_question)
        
        print(f"Retrieving relevant chunks for: '{user_question}'...")
        
        # For summary extraction queries, retrieve more chunks to ensure we have multiple chunks from the best talk
        if is_summary_query:
            chunks = self.retrieve_relevant_chunks(user_question, top_k=top_k * 3)
            if not chunks:
                return self._no_chunks_found_response()

            # Select top talk and get 1-2 chunks from that talk
            chunks = self._select_top_talk_with_chunks(chunks, max_chunks=2)
            if not chunks:
                return self._no_chunks_found_response()
            print(f"Selected {len(chunks)} chunk(s) from best matching talk")
        
        # For recommendation queries, retrieve more chunks and select top talk with chunks
        elif is_recommendation_query:
            chunks = self.retrieve_relevant_chunks(user_question, top_k=top_k * 3)
            if not chunks:
                return self._no_chunks_found_response()
            chunks = self._select_top_talk_with_chunks(chunks, max_chunks=2)
            if not chunks:
                return self._no_chunks_found_response()
            print(f"Selected {len(chunks)} chunk(s) from best matching talk for recommendation")
            
        else:
            chunks = self.retrieve_relevant_chunks(user_question, top_k=top_k)
            if not chunks:
                return self._no_chunks_found_response()
        
        requested_number = None  # Initialize for use in prompt generation later
        
        # For fact retrieval queries, deduplicate by talk_id to get unique talks
        if is_fact_query:
            chunks = self._deduplicate_chunks_by_talk(chunks)
            
            # Extract requested number or default to 1 if not specified
            requested_number = self._extract_number_from_query(user_question) or 1
            
            # Enforce "exactly N" requirement: if we don't have enough unique talks, return "I don't know"
            if len(chunks) < requested_number:
                print(f"After deduplication: {len(chunks)} unique talk(s) found (requested: {requested_number}) - insufficient results")
                return self._no_chunks_found_response()
            
            # Limit to requested number (we know we have at least that many)
            chunks = chunks[:requested_number]
            print(f"After deduplication: {len(chunks)} unique talk(s) found (requested: {requested_number})")
        
        # Build context from retrieved chunks
        context_parts = []
        sources = []
        seen_talk_ids = set()  # Track unique talks for sources list
        
        for i, chunk in enumerate(chunks, 1):
            context_part = f"[Source {i}]\n"
            context_part += f"Title: {chunk['title']}\n"
            context_part += f"Speaker: {chunk['speaker']}\n"
            if chunk.get('topics'):
                context_part += f"Topics: {chunk['topics']}\n"
            context_part += f"Transcript excerpt:\n{chunk['chunk_text']}\n"
            context_parts.append(context_part)
            
            # Only add to sources if we haven't seen this talk_id yet
            talk_id = chunk['talk_id']
            if talk_id not in seen_talk_ids:
                sources.append({
                    "title": chunk['title'],
                    "speaker": chunk['speaker'],
                    "talk_id": chunk['talk_id'],
                    "relevance_score": chunk['score']
                })
                seen_talk_ids.add(talk_id)
        
        context = "\n\n".join(context_parts)
        
        # Create prompt for LLM
        system_prompt = """
            You are a TED Talk assistant that answers questions strictly and 
            only based on the TED dataset context provided to you (metadata 
            and transcript passages). You MUST NOT use any external 
            knowledge, the open internet, or information that is not explicitly 
            contained in the retrieved context. If the answer cannot be 
            determined from the provided context, respond: "I don't know 
            based on the provided TED data." Always explain your answer 
            using the given context, quoting or paraphrasing the relevant 
            transcript or metadata when helpful.
            IMPORTANT: Keep your answers SHORT and CONCISE, maximum 250 tokens per answer. Use bullet points 
            when possible. Focus on the key information only. Avoid lengthy 
            explanations unless absolutely necessary.
            When you answer, cite the source number(s) like [Source 1], [Source 2], etc. next to each claim. 
        """
        
        # Create specialized prompt for summary extraction queries
        if is_summary_query:
            system_prompt += """
                IMPORTANT: This is a Key Idea Summary Extraction task.
                You must:
                1) Choose ONE talk from the provided context.
                2) Return:
                - Title: [exact title from context]
                - Speaker: [exact speaker name from context]
                - Key idea (1-3 sentences): [concise summary of the main idea based on the transcript excerpt(s)]
                - Evidence: include 1-2 short quotes from the transcript excerpts with [Source #].
                Rules:
                - The key idea must be supported by the provided transcript excerpt(s), not outside knowledge.
                - Keep the key idea summary to 1-3 sentences maximum.
                - Always include evidence quotes from the transcript excerpts, tagged with [Source #].
                - If the excerpt(s) do not clearly support a key idea related to the user request, reply exactly:
                "I don't know based on the provided TED data."
            """
        
        # Create specialized prompt for recommendation queries
        elif is_recommendation_query:
            system_prompt += """
                IMPORTANT: This is a Recommendation with Evidence-Based Justification task.
                You must:
                1) Recommend ONE talk from the provided context.
                2) Output exactly this structure:
                   - Recommended talk: [Title] by [Speaker]
                   - Why this fits (2-4 bullet points): each bullet must reference [Source #]
                   - Evidence: include 1-2 short quotes from the transcript excerpts with [Source #]
                Rules:
                - The recommendation and justification must be grounded ONLY in the provided context.
                - Do NOT mention multiple talks.
                - If you cannot justify a recommendation from the excerpt(s), reply exactly:
                  "I don't know based on the provided TED data."
            """
        
        # Create specialized prompt for fact retrieval queries
        elif is_fact_query:
            if requested_number and requested_number > 1:
                system_prompt += f"""\n
                    IMPORTANT: The user is asking for EXACTLY {requested_number} specific TED talk(s). You must:
                    1. Identify EXACTLY {requested_number} talk(s) from the provided context that best match the query
                    2. Return the EXACT title and speaker name for each talk from the context
                    3. Provide exactly {requested_number} talk(s) - no more, no less
                    4. Be concise and direct - use bullet points: "• [title] by [speaker]"
                    5. Keep it brief - just the essential information
                    6. If the context doesn't contain enough matching talks, respond: "I don't know based on the provided TED data."
                """
            else:
                system_prompt += """\n
                    IMPORTANT: The user is asking for a SINGLE, SPECIFIC TED talk. You must:
                    1. Identify ONE specific talk from the provided context that best matches the query
                    2. Return the EXACT title and speaker name from the context
                    3. Do NOT mention multiple talks unless the query explicitly asks for multiple
                    4. Be concise and direct - format: "Title: [title] by [speaker]"
                    5. Keep it brief - just the essential information
                    6. If the context doesn't contain a matching talk, respond: "I don't know based on the provided TED data."
                """
        
        user_prompt = f"""\nQuestion: {user_question}\n\n
                        Context from TED Talks database:\n
                        {context}\n"""

        # Get answer from LLM
        print("Generating answer...")
        response = self.llmod_client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=self.temperature,  # Low temperature for more factual responses
            max_tokens=250  # Cap output to keep answers short
        )
        
        answer = response.choices[0].message.content
        
        # Prepare context list for API response
        # Note: chunks may have been deduplicated and sliced to N for fact queries,
        # so context_list will contain exactly those N chunks (one per unique talk)
        context_list = []
        for chunk in chunks:
            context_list.append({
                "talk_id": chunk.get("talk_id"),
                "title": chunk.get("title"),
                "speaker": chunk.get("speaker"),
                "topics": chunk.get("topics"),
                "chunk_text": chunk.get("chunk_text"),
                "score": chunk.get("score")
            })
        
        return {
            "answer": answer,
            "sources": sources,
            "num_sources": len(sources),
            "context": context_list,
            "augmented_prompt": {
                "System": system_prompt.strip(),
                "User": user_prompt.strip()
            }
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
        LLMOD_API_KEY = api_keys['LLMOD_API_KEY']
    
    # Initialize RAG system
    rag = TEDTalksRAG(
        pinecone_api_key=PINECONE_API_KEY,
        llmod_api_key=LLMOD_API_KEY,
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

