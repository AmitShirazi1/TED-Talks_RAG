CSV_FILE_PATH = "tiny_ted.csv"

EMBEDDING_MODEL = "RPRTHPB-text-embedding-3-small"
CHAT_MODEL = "RPRTHPB-gpt-5-mini"

"""
Chunking parameters choices explanation:

Default chunk size:
-   Ted talks could be considered to be in the same category as books, general text and articles.
    We learned in class that the recommended number of tokens per chunk is 512+.
-   We also learned that the recommended number of tokens per chunk in conversational data is 200-400.
    This also may apply to TED talks, as they are somewhat conversational.
-   In addition, considering the limited budget, it's best to choose a chunk size that is as small as possible (without losing too much information).
-   Based on the above and consulting GPT, I chose a chunk size of around 2000 characters (approximately 500 tokens).

Default overlap:
-   We learned in class that the recommended overlap long articles and books is 5-15%.
    TED talks best fit in this category, in my opinion.
-   Considering the limited budget, it's best to choose an overlap that is as small as possible (without losing too much context).
-   Based on the above and consulting GPT, I chose an overlap of around 7.5%.

Default retrieval K:
-   We learned in class that the recommended number of chunks to retrieve for research papers and long textis 8-12.
-   However, TED talks could be also fall to the general text category, in which case the recommended number of chunks to retrieve is 3-5.
    So initially, I chose K=8, which is a good compromise between the two.
-   Considering the limited budget, it's best to choose a retrieval K that is as small as possible (without losing too much context).
-   Based on the above and consulting GPT, I finally chose to lower K to 5.
"""
DEFAULT_CHUNK_SIZE = 2000  # characters (approx 500 tokens)
DEFAULT_OVERLAP = 150  # 7.5% of 2000 characters
DEFAULT_RETRIEVE_TOP_K = 5


def get_index_name(csv_file_path: str=CSV_FILE_PATH, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP) -> str:
    """
    Generate index name based on chunking parameters to avoid stale indices.
    
    When chunking parameters change, a new index name is generated, preventing
    accidental use of stale indices with old chunking settings.
    
    Args:
        chunk_size: Chunk size in characters
        overlap: Overlap in characters
        
    Returns:
        Index name string, e.g., "ted-talks-en-c2000-o150"
    """
    return f"{csv_file_path.split('.')[0].replace('_', '-')}-c{chunk_size}-o{overlap}"


# Index name includes chunking parameters to prevent stale indices
INDEX_NAME = get_index_name()