import time
from gpiozero import Button, Buzzer
from threading import Timer, Lock

# --- TUNING ---
DOT_THRESHOLD = 0.20
LETTER_PAUSE = 0.50
WORD_PAUSE = 1.20

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

class MorseToEnglish:
    def __init__(self):
        # Using Button class for cleaner event handling
        self.button = Button(17, pull_up=True, bounce_time=0.01)
        self.buzzer = Buzzer(27)
        
        self.press_time = 0
        self.buffer = ""
        self.full_message = ""
        self.timer = None
        self.lock = Lock()

        self.button.when_pressed = self.handle_press
        self.button.when_released = self.handle_release

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

            if duration < DOT_THRESHOLD:
                self.buffer += "."
            else:
                self.buffer += "-"

            self.timer = Timer(LETTER_PAUSE, self.decode_current_buffer)
            self.timer.start()

    def decode_current_buffer(self):
        with self.lock:
            char = MORSE_CODE_DICT.get(self.buffer, "")
            if char:
                print(char, end="", flush=True)
                self.full_message += char
            self.buffer = ""

            self.timer = Timer(WORD_PAUSE - LETTER_PAUSE, self.add_space)
            self.timer.start()

    def add_space(self):
        with self.lock:
            if self.full_message and not self.full_message.endswith(" "):
                print(" ", end="", flush=True)
                self.full_message += " "

decoder = MorseToEnglish()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    # Save the message to a file for the RAG system to read
    with open("query.txt", "w") as f:
        f.write(decoder.full_message.strip())
    print("\nMessage saved for RAG:", decoder.full_message.strip())
