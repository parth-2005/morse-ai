import os
import sys
from config import Config
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import OllamaEmbeddings

def verify_hybrid():
    print("Verifying Hybrid Configuration (Gemini LLM + Ollama Embeddings)...")
    print(f"API Key present: {bool(Config.GOOGLE_API_KEY) and Config.GOOGLE_API_KEY != 'YOUR_GOOGLE_API_KEY_HERE'}")
    
    if Config.GOOGLE_API_KEY == 'YOUR_GOOGLE_API_KEY_HERE':
        print("ERROR: Please set GOOGLE_API_KEY in config.py or as an environment variable.")
        return

    try:
        print(f"Initializing ChatGoogleGenerativeAI ({Config.LLM_MODEL})...")
        llm = ChatGoogleGenerativeAI(
            model=Config.LLM_MODEL,
            google_api_key=Config.GOOGLE_API_KEY,
            temperature=0
        )
        response = llm.invoke("Hello, are you working?")
        print(f"LLM Response: {response.content}")
        print("LLM Verification Successful.")
    except Exception as e:
        print(f"LLM Verification Failed: {e}")

    try:
        print(f"Initializing OllamaEmbeddings ({Config.EMBEDDING_MODEL})...")
        embeddings = OllamaEmbeddings(
            model=Config.EMBEDDING_MODEL
        )
        vec = embeddings.embed_query("test query")
        print(f"Embedding generated. Length: {len(vec)}")
        print("Embedding Verification Successful.")
    except Exception as e:
        print(f"Embedding Verification Failed: {e}")

if __name__ == "__main__":
    verify_hybrid()
