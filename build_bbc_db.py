# ==============================================================================
# BBC News Database Builder
# ==============================================================================
# This script builds a vector database of BBC news articles using Qdrant and LlamaIndex.
# It downloads the BBC news dataset, processes the articles, and creates embeddings
# for semantic search capabilities.

# Vector database and document processing imports
from qdrant_client import QdrantClient  # Vector database client
#from qdrant_client.models import VectorParams, Distance
from llama_index.core import VectorStoreIndex, Settings, Document, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # Text embeddings
from llama_index.vector_stores.qdrant import QdrantVectorStore  # Qdrant integration

# Dataset handling and utilities
from datasets import load_dataset  # HuggingFace datasets for BBC news
import os
import logging
from dotenv import load_dotenv  # Environment variable management

# Load environment variables from .env file
load_dotenv()

# Configure logging for better debugging and monitoring
logging.basicConfig(level=logging.INFO)

# ==============================================================================
# Vector Database and Embedding Configuration
# ==============================================================================
# Initialize Qdrant client for vector storage
client = QdrantClient(path=os.getenv('QDRANT_PATH'))

# Configure embedding model for converting text to vectors
embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5",  # Efficient English embedding model
    device="mps",  # Use Metal Performance Shaders on Mac (change to "cuda" for GPU)
    embed_batch_size=10,  # Process embeddings in batches
)

# Global LlamaIndex settings for document processing
Settings.embed_model = embed_model
Settings.chunk_size = 512  # Size of text chunks for processing
Settings.chunk_overlap = 50  # Overlap between chunks to maintain context

# ==============================================================================
# Database Building Functions
# ==============================================================================
def rebuild_index(client: QdrantClient):
    """Rebuild the vector index with BBC news articles.
    
    This function:
    1. Loads the BBC news dataset
    2. Processes and deduplicates articles
    3. Creates embeddings and stores them in Qdrant
    
    Args:
        client: QdrantClient instance for vector storage
        
    Returns:
        VectorStoreIndex: The created vector index
        
    Raises:
        ValueError: If dataset loading fails
    """
    try:
        # Load BBC news dataset from HuggingFace
        news_dataset = load_dataset(
            "RealTimeData/bbc_news_alltime", "2017-12", split="train"
        )
        logging.info(f"Loaded the BBC News dataset with {len(news_dataset)} rows")
        logging.info(f"Successfully loaded the BBC News dataset with {len(news_dataset)} rows.")
    except Exception as e:
        raise ValueError(f"Error loading the BBC News dataset: {str(e)}")

    # Extract and deduplicate news articles
    news_articles = news_dataset["content"]
    unique_articles = set()
    for article in news_articles:
        if article:
            unique_articles.add(article)
    unique_news_articles = list(unique_articles)
    logging.info(f"We have {len(unique_news_articles)} unique articles in our database.")

    # Filter out extremely long articles to prevent embedding issues
    articles = [article for article in unique_news_articles if article and len(article) <= 50000]

    # Convert articles to LlamaIndex Documents
    documents = [Document(text=t) for t in articles]
    
    # Set up vector store and storage context
    vector_store = QdrantVectorStore(client=client, collection_name=os.getenv('QDRANT_COLLECTION'))
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Create and return the vector index
    index = VectorStoreIndex.from_documents(
        documents, 
        storage_context=storage_context, 
        show_progress=True  # Show progress bar during indexing
    )
    return index

# ==============================================================================
# Main Execution
# ==============================================================================
# Check if collection exists and rebuild if necessary
if not client.collection_exists(os.getenv('QDRANT_COLLECTION')):
    logging.info(f"Collection {os.getenv('QDRANT_COLLECTION')} does not exist, rebuilding index...")
    index = rebuild_index(client)
    logging.info(f"Successfully rebuilt the index for collection {os.getenv('QDRANT_COLLECTION')}.")
else:
    logging.info(f"Collection {os.getenv('QDRANT_COLLECTION')} already exists.")