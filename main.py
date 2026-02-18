import sys
import os
import time
from src.input_handler import MorseInput, TextInput, InputPostProcessor
from src.rag_engine import RAGEngine
from src.tts_engine import TTSEngine
import argparse

def main():
    print("Initializing Educational RAG System...")
    
    # 1. Initialize TTS
    try:
        tts = TTSEngine()
        tts.speak("System Initializing")
    except Exception as e:
        print(f"Error initializing TTS: {e}")
        return

    # 2. Initialize RAG
    try:
        print("Loading Knowledge Base...")
        rag = RAGEngine()
        # Ensure data directory exists
        if not os.path.exists("data"):
            os.makedirs("data")
            print("Warning: 'data' directory created but empty. Please add .txt files.")
            tts.speak("Data directory empty. Please add text files.")
        else:
            if not rag.chain:
                print("Warning: RAG Chain not initialized. Please run 'ingest.py' to ingest documents.")
                tts.speak("Please run ingest script first.")
            else:
                tts.speak("Knowledge Base Loaded")
    except Exception as e:
        print(f"Error initializing RAG: {e}")
        tts.speak("Error loading knowledge base")
        return

    # 3. Initialize Input
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--text", action="store_true", help="Use text input instead of Morse code")
        args = parser.parse_args()

        if args.text:
            input_handler = TextInput()
            tts.speak("Text Input Ready")
            print("Text Input Mode Selected.")
        else:
            # Use Board numbering or BCM? gpiozero uses BCM by default.
            # Pin 17 for Button, 27 for Buzzer as per test.py
            input_handler = MorseInput(button_pin=17, buzzer_pin=27)
            tts.speak("Morse Input Ready")
    except Exception as e:
        print(f"Error initializing Input: {e}")
        tts.speak("Error initializing input")
        return

    print("\n--- System Ready ---\n")

    # Initialize Post Processor
    post_processor = InputPostProcessor()

    if args.text:
        print("Type your query and press Enter.")
    else:
        print("Press the button to input Morse code.")
        print("Long pause (2s) submits the query.")
    print("Ctrl+C to exit.")

    try:
        while True:
            # Get input (blocking)
            query = input_handler.get_input()
            
            if query:
                print(f"\nReceived Query: {query}")
                tts.speak(f"Received input")

                # Post-process input (Word Segmentation)
                corrected_query = post_processor.process_input(query)
                print(f"Corrected Query: {corrected_query}")
                tts.speak(f"Corrected to: {corrected_query}")
                
                # Query RAG
                answer = rag.query(corrected_query)
                print(f"Answer: {answer}")
                
                # Speak Answer
                tts.speak(answer)
            else:
                # Small sleep to prevent tight loop if get_input returns immediately (e.g. error)
                time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nExiting...")
        tts.speak("System Shutting Down")
    finally:
        input_handler.close()

if __name__ == "__main__":
    main()
