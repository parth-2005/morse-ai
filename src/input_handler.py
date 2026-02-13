import time
from abc import ABC, abstractmethod
from threading import Timer, Lock
from queue import Queue, Empty
import sys
import os

# Add parent directory to path to import config if run from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

# Try to import gpiozero, handle if not present (for non-Pi testing)
try:
    from gpiozero import Button, Buzzer
except ImportError:
    Button = None
    Buzzer = None
    print("Warning: gpiozero not found. MorseInput will not work without hardware.")

MORSE_CODE_DICT = {
    '.-': 'A', '-...': 'B', '-.-.': 'C', '-..': 'D', '.': 'E',
    '..-.': 'F', '--.': 'G', '....': 'H', '..': 'I', '.---': 'J',
    '-.-': 'K', '.-..': 'L', '--': 'M', '-.': 'N', '---': 'O',
    '.--.': 'P', '--.-': 'Q', '.-.': 'R', '...': 'S', '-': 'T',
    '..-': 'U', '...-': 'V', '.--': 'W', '-..-': 'X', '-.--': 'Y',
    '--..': 'Z', '.----': '1', '..---': '2', '...--': '3', '....-': '4',
    '.....': '5', '-....': '6', '--...': '7', '---..': '8', '----.': '9',
    '-----': '0', '.-.-.-': '.', '--..--': ',', '..--..': '?'
}

class InputSource(ABC):
    @abstractmethod
    def get_input(self) -> str:
        """
        Blocking call to get the next input string.
        Should raise KeyboardInterrupt if interrupted.
        """
        pass

    @abstractmethod
    def close(self):
        """Cleanup resources."""
        pass

class MorseInput(InputSource):
    def __init__(self, button_pin=Config.GPIO_BUTTON_PIN, buzzer_pin=Config.GPIO_BUZZER_PIN):
        if Button is None:
            raise ImportError("gpiozero is required for MorseInput")
            
        # Using Button class for cleaner event handling
        self.button = Button(button_pin, pull_up=True, bounce_time=Config.BOUNCE_TIME)
        self.buzzer = Buzzer(buzzer_pin)
        
        self.press_time = 0
        self.buffer = ""
        self.current_message = ""
        self.message_queue = Queue()
        
        self.timer = None
        self.lock = Lock()

        self.button.when_pressed = self.handle_press
        self.button.when_released = self.handle_release
        
        print("Morse Code Input Ready.")

    def handle_press(self):
        with self.lock:
            self.buzzer.on() # Instant feedback
            if self.timer:
                self.timer.cancel()
            self.press_time = time.time()

    def handle_release(self):
        with self.lock:
            self.buzzer.off()
            duration = time.time() - self.press_time

            if duration < Config.DOT_THRESHOLD:
                self.buffer += "."
            else:
                self.buffer += "-"

            self.timer = Timer(Config.LETTER_PAUSE, self.decode_current_buffer)
            self.timer.start()

    def decode_current_buffer(self):
        with self.lock:
            char = MORSE_CODE_DICT.get(self.buffer, "")
            if char:
                print(char, end="", flush=True)
                self.current_message += char
            self.buffer = ""

            self.timer = Timer(Config.WORD_PAUSE - Config.LETTER_PAUSE, self.end_of_message_check)
            self.timer.start()

    def end_of_message_check(self):
        """
        Called after WORD_PAUSE.
        """
        with self.lock:
            # First, add space if needed
            if self.current_message and not self.current_message.endswith(" "):
                print(" ", end="", flush=True)
                self.current_message += " "
            
            # Start a timer to submit if no more input comes
            self.timer = Timer(Config.SUBMIT_TIMEOUT, self.submit_message)
            self.timer.start()

    def submit_message(self):
        with self.lock:
            final_msg = self.current_message.strip()
            if final_msg:
                print("\nSubmitting:", final_msg)
                self.message_queue.put(final_msg)
                self.current_message = ""

    def get_input(self) -> str:
        # Block until a message is ready
        return self.message_queue.get()

    def close(self):
        self.button.close()
        self.buzzer.close()

class TextInput(InputSource):
    """Simple console input for testing purposes."""
    def get_input(self) -> str:
        try:
            return input("\nEnter text query: ").strip()
        except EOFError:
            return ""
            
    def close(self):
        pass
            
    def close(self):
        pass
