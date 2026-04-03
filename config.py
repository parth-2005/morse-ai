
"""
RAG System Configuration
------------------------
This file contains all the hyperparameters and configuration settings for the Educational RAG system.
Modify these values to tune the performance of the system.
"""

import os
from dotenv import load_dotenv

load_dotenv(".env.local")

class Config:
    # ==========================================
    # 1. System Paths
    # ==========================================
    # Directory containing the knowledge base (PDFs, TXT files)
    DATA_DIR = "data"
    
    # Directory where the FAISS vector store will be saved/loaded
    VECTOR_STORE_PATH = "vectorstore/db_faiss"

    # ==========================================
    # 2. LLM (Large Language Model) Settings
    # ==========================================
    # Google API Key 
    # Ensure you set this in your environment variables or .env file
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
    print(f"Using Google API Key: {'SET' if GOOGLE_API_KEY else 'NOT SET'}")
    # Model name served by Google Gemini
    LLM_MODEL = "gemini-2.5-flash"
    
    # Temperature (0.0 to 1.0)
    LLM_TEMPERATURE = 0.8
    
    # ==========================================
    # 3. Embedding & Retrieval Settings
    # ==========================================
    # Embedding model name
    EMBEDDING_MODEL = "nomic-embed-text"
    
    # Text Splitter Settings
    # How to chunk the documents before embedding.
    CHUNK_SIZE = 500       # Number of characters per chunk
    CHUNK_OVERLAP = 50     # Number of characters of overlap between chunks (preserves context)
    
    # Retrieval Settings
    # Number of relevant document chunks to retrieve for each query.
    # Increasing this gives the model more context but might confuse smaller models.
    RETRIEVAL_K = 3

    # ==========================================
    # 4. RAG Prompt Template
    # ==========================================
    # The template used to construct the prompt for the LLM.
    # {context} will be replaced by retrieved documents.
    # {question} will be replaced by the user's query.
    PROMPT_TEMPLATE = """You are a helpful educational assistant. Use the following pieces of context to answer the question at the end.
Keep the answer easy to understand.

Context:
{context}

Question: {question}

Answer:"""

    # ==========================================
    # 5. TTS (Text-to-Speech) Settings
    # ==========================================
    # Audio output device name (run 'aplay -L' to see available devices)
    # Common options: 'default', 'bluealsa', 'hw:0,0'
    # TTS_DEVICE = "bluealsa"
    # In config.py
    TTS_DEVICE = "bluealsa:SRV=org.bluealsa,DEV=36:D5:7C:06:A2:28,PROFILE=a2dp"    
    # Volume (0 to 100, though implementation might depend on system)
    TTS_VOLUME = 80
    
    # Audio Sample Rate (Hz)
    # 24000 is a good balance for quality and performance with gTTS/ffmpeg
    TTS_SAMPLE_RATE = 24000

    # ==========================================
    # 6. Input (Morse/GPIO) Settings
    # ==========================================
    # GPIO Pins (BCM numbering)
    GPIO_BUTTON_PIN = 17
    GPIO_BUZZER_PIN = 27
    
    # Morse Code Timing (seconds)
    # Threshold to distinguish between a DOT and a DASH
    DOT_THRESHOLD = 0.20
    
    # Pause between parts of the same letter
    # (Usually not explicitly checked if just using button up/down times, but for reference)
    
    # Pause to identify end of a letter
    LETTER_PAUSE = 0.50
    
    # Pause to identify end of a word (inserts a space)
    WORD_PAUSE = 1.20
    
    # Time of silence to assume the message is complete and submit it
    SUBMIT_TIMEOUT = 10


    # Debounce time for button press (seconds)
    BOUNCE_TIME = 0.01
