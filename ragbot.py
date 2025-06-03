# ==============================================================================
# RAG-based News Agent with PydanticAI and LlamaIndex
# ==============================================================================
# This script demonstrates how to build a Retrieval-Augmented Generation (RAG)
# system that combines PydanticAI agents with LlamaIndex for document retrieval.
# The agent can search through BBC news articles to answer questions about
# current events and news topics.

# Core PydanticAI imports for building intelligent agents
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

# Vector database and retrieval components
from qdrant_client import QdrantClient  # Vector database client
from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # Text embeddings
from llama_index.vector_stores.qdrant import QdrantVectorStore  # Qdrant integration
from llama_index.core.retrievers import BaseRetriever  # Base retriever interface

# Standard library and utility imports
from typing import List
import httpx
from dotenv import load_dotenv  # Environment variable management
import os
import base64
import logfire  # Observability and logging
from dataclasses import dataclass

# Load environment variables from .env file
load_dotenv()

# ==============================================================================
# Observability Setup with Langfuse Integration
# ==============================================================================
# Configure tracing and monitoring for the agent's performance
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_AUTH = base64.b64encode(f"{LANGFUSE_PUBLIC_KEY}:{LANGFUSE_SECRET_KEY}".encode()).decode()
 
# Set up OpenTelemetry exports to Langfuse for tracing
os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "https://us.cloud.langfuse.com/api/public/otel" # US data region
os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {LANGFUSE_AUTH}"

# Configure logfire for application monitoring
logfire.configure(
    service_name='henrys_chatbot',
    # Sending to Logfire is on by default regardless of the OTEL env vars.
    send_to_logfire=False,
)

# ==============================================================================
# Language Model Configuration
# ==============================================================================
# Set up the LLM using Google's Gemini model via OpenRouter
model = OpenAIModel(
    'google/gemini-2.0-flash-lite-001',  # Fast and efficient Gemini model
    provider=OpenAIProvider(
        base_url='https://openrouter.ai/api/v1',  # OpenRouter provides unified API access
        api_key=os.getenv("OPENROUTER_API_KEY"),  # API key from environment
    ),
)

# ==============================================================================
# Vector Database and Embedding Setup
# ==============================================================================
# Initialize Qdrant client for vector storage and similarity search
client = QdrantClient(path=os.getenv('QDRANT_PATH'))  # Local Qdrant instance

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
# Vector Store and Retriever Initialization
# ==============================================================================
# Check if the vector collection exists and set up retriever
if client.collection_exists(os.getenv('QDRANT_COLLECTION')):
    # Connect to existing vector store collection
    vector_store = QdrantVectorStore(client=client, collection_name=os.getenv('QDRANT_COLLECTION'))
    
    # Create vector index from existing collection
    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
    
    # Create retriever to search for relevant documents
    retriever = index.as_retriever(similarity_top_k=5)  # Return top 5 most similar documents
else:
    # Raise error if collection doesn't exist - you need to create it first
    raise ValueError(f"Collection {os.getenv('QDRANT_COLLECTION')} does not exist")

# ==============================================================================
# Agent Dependencies Definition
# ==============================================================================
@dataclass
class RagDeps:
    """Dependencies for the RAG (Retrieval-Augmented Generation) agents.
    
    Attributes:
        retriever: A LlamaIndex BaseRetriever instance used to search through
                  the BBC News database for relevant articles and information.
    """
    retriever: BaseRetriever

# ==============================================================================
# PydanticAI Agent Configuration
# ==============================================================================
# Create a specialized news agent with retrieval capabilities
bbc_agent = Agent(
    model=model,  # Use the configured Gemini model
    deps_type=RagDeps,  # Specify the dependency type for type safety
    # Define the agent's role and behavior
    system_prompt=(
        'Please answer everything in Traditional Chinese.'
        'You are a news assistant that helps users find relevant news information. '
        'When users ask questions about current events, news, politics, sports, technology, '
        'business, entertainment, or any topic that would require up-to-date information, '
        'use the retriever tool to search for relevant news articles first. '
        'Base your response on the retrieved information and cite the sources when possible. '
        'If the question is not related to news or current events (like general knowledge, '
        'personal advice, or simple calculations), answer directly without using the retriever. '
    ),
    instrument=True,  # Enable instrumentation for observability
)

# ==============================================================================
# Tool Function Definition
# ==============================================================================
@bbc_agent.tool
async def get_bbc_news(ctx: RunContext[RagDeps], query: str) -> List[str]:
    """Search for relevant BBC news articles based on a query.
    
    Use this tool when users ask about current events, news, politics, economics,
    international relations, business developments, or any topic that would benefit
    from recent news information. This tool searches through a database of BBC news
    articles to find the most relevant content.
    
    Args:
        query: A search query describing the news topic or information needed.
               Should be specific and use relevant keywords but keep it as full sentence.
               (e.g., "Latest news about China and US trade tariff",
               "What is the latest developement about Brexit negotiations", etc.). 
    
    Returns:
        List[str]: A list of relevant news article excerpts or summaries that match
                  the search query. Each item contains text content from BBC news articles.
    
    Examples of when to use this tool:
    - "What's the latest on China-US trade relations?"
    - "Tell me about recent climate change policies"
    - "What happened in the UK elections?"
    - "Any news about technology companies and AI regulation?"
    """
    # Use the retriever to find relevant documents based on the query
    results = ctx.deps.retriever.retrieve(query)
    
    # Extract text content from retrieved documents
    return [result.text for result in results]

# ==============================================================================
# Agent Execution Example
# ==============================================================================
# Create dependencies instance with the configured retriever
deps = RagDeps(retriever=retriever)

# Run the agent with a sample query about trade relations
result_sync = bbc_agent.run_sync('Can you tell me the news about trading tariff between China and US?', deps=deps)

# Display the agent's response
print(result_sync.output)
