# graph_utils.py
import streamlit as st
from langchain.chains import GraphCypherQAChain
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI
from functools import lru_cache
from typing import Tuple, Optional
import logging
   
   # Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
   
@lru_cache(maxsize=1)
def get_neo4j_connection() -> Neo4jGraph:
       """
       Create and cache Neo4j connection.
       Returns cached connection on subsequent calls.
       """
       try:
           graph = Neo4jGraph(
               url=st.secrets["NEO4J_URI"],
               username=st.secrets["NEO4J_USER"],
               password=st.secrets["NEO4J_PASSWORD"],
               enhanced_schema=True
           )
           graph.refresh_schema()
           return graph
       except Exception as e:
           logger.error(f"Failed to initialize Neo4j connection: {e}")
           raise
def create_chain(api_key: str, graph: Neo4jGraph) -> GraphCypherQAChain:
       """Create the GraphCypherQAChain with the given API key and graph."""
       try:
           return GraphCypherQAChain.from_llm(
               ChatOpenAI(
                   api_key=api_key, 
                   model="gpt-4o",
                   temperature=0.7,  # Add temperature for better response variety
                   request_timeout=30  # Add timeout for better error handling
               ),
               graph=graph,
               verbose=True,
               show_intermediate_steps=True,
               allow_dangerous_requests=True,
           )
       except Exception as e:
           logger.error(f"Failed to create chain: {e}")
           raise
   
@st.cache_resource(show_spinner=False, ttl=3600)  # Cache for 1 hour
def init_resources(api_key: str) -> Tuple[Neo4jGraph, GraphCypherQAChain]:
       """
       Initialize and cache Neo4j graph and chain resources.
       Uses caching for better performance.
       """
       graph = get_neo4j_connection()
       chain = create_chain(api_key, graph)
       return graph, chain
   
def query_graph(chain: GraphCypherQAChain, query: str) -> Optional[str]:
       """
       Execute a query against the graph with error handling and timeout.
       """
       try:
           result = chain.invoke(
               {
                   "query": query
               },
               config={
                   "timeout": 30  # Add timeout for long-running queries
               }
           )["result"]
           return result
       except Exception as e:
           logger.error(f"Query execution failed: {e}")
           return None
   
   # Connection cleanup
def cleanup_resources(graph: Neo4jGraph):
       """Cleanup Neo4j connection when no longer needed."""
       try:
           if graph:
               graph.close()
       except Exception as e:
           logger.error(f"Failed to cleanup resources: {e}")
   