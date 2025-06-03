# ==============================================================================
# Serper.dev Web Search Agent with PydanticAI
# ==============================================================================
# This script implements a web search agent using Serper.dev's API through PydanticAI.
# It provides structured search results and handles various types of web queries.

from dataclasses import dataclass
from typing import List, Optional
import os
from dotenv import load_dotenv
import httpx
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
import base64
import logfire  # Observability and logging

# Load environment variables
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
    service_name='serper_search_agent',
    # Sending to Logfire is on by default regardless of the OTEL env vars.
    send_to_logfire=False,
)

# ==============================================================================
# Data Models
# ==============================================================================
class SearchResult(BaseModel):
    """Structured model for individual search results."""
    title: str = Field(description="The title of the search result")
    link: str = Field(description="The URL of the search result")
    snippet: str = Field(description="A brief description or excerpt from the result")
    position: int = Field(description="The position of the result in the search results")

class SearchResponse(BaseModel):
    """Structured model for the search response containing only organic results."""
    organic_results: List[SearchResult] = Field(
        description="List of organic search results",
        default_factory=list
    )

@dataclass
class SearchDeps:
    """Dependencies for the search agent."""
    client: httpx.AsyncClient
    api_key: str

# ==============================================================================
# Agent Configuration
# ==============================================================================
# Initialize the language model
model = OpenAIModel(
    'google/gemini-2.0-flash-lite-001',
    provider=OpenAIProvider(
        base_url='https://openrouter.ai/api/v1',
        api_key=os.getenv("OPENROUTER_API_KEY"),
    ),
)

# Create the search agent
search_agent = Agent(
    model=model,
    deps_type=SearchDeps,
    output_type=str,
    system_prompt=(
        "You are a web search assistant that helps users find information from the internet. "
        "Your primary function is to use the web search tool to find relevant information. "
        "You should use the search tool whenever possible including but not limited to following cases: "
        "1. Users provide any topic or subject they want to learn about "
        "2. Users provide specific keywords or search terms "
        "3. Users ask about current events or recent developments "
        "4. Users need up-to-date information from the web "
        "Always structure your response based on the search results and cite your sources. "
        "If the search results don't contain enough information, say so clearly. "
        "Please answer everything in Traditional Chinese."
    ),
    instrument=True,
)

# ==============================================================================
# Search Tool Implementation
# ==============================================================================
@search_agent.tool
async def web_search(ctx: RunContext[SearchDeps], query: str) -> SearchResponse:
    """Perform a web search using Serper.dev API.
    
    This is your primary tool for finding information. Use it whenever users:
    - Provide any topic they want to learn about
    - Give specific keywords or search terms
    - Ask about current events or recent developments
    - Need up-to-date information from the web
    
    The tool will search the internet and return relevant web pages and articles.
    
    Args:
        query: The search query to look up. Can be:
               - A specific topic (e.g., "中聯辦人事變動")
               - Keywords (e.g., "中聯辦 最新消息")
               - A question (e.g., "中聯辦最近有什麼變化")
        
    Returns:
        SearchResponse: List of search results with title, link, and snippet
    """
    headers = {
        "X-API-KEY": ctx.deps.api_key,
        "Content-Type": "application/json"
    }
    
    payload = {
        "q": query,
        "gl": "hk",  # Set to Hong Kong for Traditional Chinese results
        "hl": "zh-hk"  # Set language to Traditional Chinese
    }
    
    response = await ctx.deps.client.post(
        "https://google.serper.dev/search",
        json=payload,
        headers=headers
    )
    response.raise_for_status()
    
    data = response.json()
    
    # Convert only organic results to our structured format
    organic_results = [
        SearchResult(
            title=result.get("title", ""),
            link=result.get("link", ""),
            snippet=result.get("snippet", ""),
            position=idx + 1
        )
        for idx, result in enumerate(data.get("organic", []))
    ]
    
    return SearchResponse(organic_results=organic_results)

# ==============================================================================
# Example Usage
# ==============================================================================
async def main():
    """Example of how to use the search agent."""
    async with httpx.AsyncClient() as client:
        deps = SearchDeps(
            client=client,
            api_key=os.getenv("SERPER_API_KEY")
        )
        
        # Example search query
        result = await search_agent.run(
            "do a search and summarize the news about match between Hong Kong and Manchester United in 2025",
            deps=deps
        )
        print(result.output)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
