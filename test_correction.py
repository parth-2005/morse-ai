import sys
import os
import time

# Add parent directory to path to import src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.input_handler import InputPostProcessor

def test_correction():
    print("Initializing InputPostProcessor...")
    processor = InputPostProcessor()
    
    test_cases = [
        "WHATISSOLENOID",
        "W H A T I S S O L E N O I D",
        "thequickbrownfox",
        "morse code is fun",
        "HELLOWORLD"
    ]

    print("\n--- Starting Tests ---\n")
    for case in test_cases:
        print(f"Original: '{case}'")
        
        start = time.time()
        result = processor.process_input(case)
        duration = time.time() - start
        
        print(f"Result:   '{result}'")
        print(f"Time:     {duration:.2f}s")
        print("-" * 30)

if __name__ == "__main__":
    test_correction()
