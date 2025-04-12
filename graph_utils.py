# graph_utils.py
import streamlit as st
from langchain.chains import GraphCypherQAChain
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI
@st.cache_resource(show_spinner=False)
def init_resources(api_key):
   graph = Neo4jGraph(
       url=st.secrets["NEO4J_URI"],
       username=st.secrets["NEO4J_USER"],
       password=st.secrets["NEO4J_PASSWORD"],
       enhanced_schema=True,
   )
   graph.refresh_schema()
   chain = GraphCypherQAChain.from_llm(
       ChatOpenAI(api_key=api_key, model="gpt-4o"),
       graph=graph,
       verbose=True,
       show_intermediate_steps=True,
       allow_dangerous_requests=True,
   )
   return graph, chain
def query_graph(chain, query):
   result = chain.invoke({"query": query})["result"]
   return result