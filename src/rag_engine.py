import os
import sys
# Add parent directory to path to import config if run from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from config import Config

class RAGEngine:
    def __init__(self):
        self.chain = None
        
        # Check if vector store exists
        if not os.path.exists(Config.VECTOR_STORE_PATH):
            print(f"Warning: Vector store not found at {Config.VECTOR_STORE_PATH}. Please run ingest.py first.")
            return

        print(f"Loading vector store from {Config.VECTOR_STORE_PATH}...")
        embeddings = OllamaEmbeddings(model=Config.EMBEDDING_MODEL)
        try:
            vectorstore = FAISS.load_local(Config.VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
            print("Vector store loaded.")
            
            print(f"Initializing LLM {Config.LLM_MODEL}...")
            # Configure LLM with hyperparameters
            self.llm = ChatGoogleGenerativeAI(
                model=Config.LLM_MODEL,
                temperature=Config.LLM_TEMPERATURE,
                google_api_key=Config.GOOGLE_API_KEY
            )
            
            retriever = vectorstore.as_retriever(search_kwargs={"k": Config.RETRIEVAL_K})
            
            # RAG Prompt
            prompt = ChatPromptTemplate.from_template(Config.PROMPT_TEMPLATE)
            
            print("Creating LCEL chain...")
            self.chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
            )
            print("RAG Engine Ready.")
        except Exception as e:
            print(f"Error loading RAG Engine: {e}")

    def query(self, query_text):
        """Retrieves relevant context and generates an answer using LangChain LCEL."""
        if not self.chain:
            return "Knowledge base not loaded. Please run ingest.py."
            
        print(f"Querying: {query_text}")
        
        try:
            response = self.chain.invoke(query_text)
            return response
        except Exception as e:
            return f"Error generating response: {e}"
