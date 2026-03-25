import os
import sys
import time
from abc import ABC, abstractmethod
from queue import Empty, Queue
from threading import Lock, Timer

import cv2
import wordsegment

# Add parent directory to path to import config if run from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

# Try to import gpiozero, handle if not present (for non-Pi testing)
try:
    from gpiozero import Button, Buzzer
except ImportError:
    Button = None
    Buzzer = None
    print("Warning: gpiozero not found. Hardware inputs will not work.")

try:
    import speech_recognition as sr
except ImportError:
    sr = None

# Make vision_reader importable when running from src/
VISION_READER_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vision_reader")
if VISION_READER_PATH not in sys.path:
    sys.path.append(VISION_READER_PATH)

try:
    from assistive_reader import ImageProcessor, OCREngine
except ImportError:
    ImageProcessor = None
    OCREngine = None


MORSE_CODE_DICT = {
    ".-": "A", "-...": "B", "-.-.": "C", "-..": "D", ".": "E",
    "..-.": "F", "--.": "G", "....": "H", "..": "I", ".---": "J",
    "-.-": "K", ".-..": "L", "--": "M", "-.": "N", "---": "O",
    ".--.": "P", "--.-": "Q", ".-.": "R", "...": "S", "-": "T",
    "..-": "U", "...-": "V", ".--": "W", "-..-": "X", "-.--": "Y",
    "--..": "Z", ".----": "1", "..---": "2", "...--": "3", "....-": "4",
    ".....": "5", "-....": "6", "--...": "7", "---..": "8", "----.": "9",
    "-----": "0", ".-.-.-": ".", "--..--": ",", "..--..": "?",
}


class InputSource(ABC):
    @abstractmethod
    def get_input(self) -> str:
        """Blocking call to get the next input string."""

    @abstractmethod
    def close(self):
        """Cleanup resources."""


class InputPostProcessor:
    def __init__(self):
        print("Loading WordSegment...")
        wordsegment.load()
        print("WordSegment loaded.")

    def process_input(self, raw_text):
        if not raw_text:
            return ""
        clean_text = raw_text.replace(" ", "")
        segmentation = wordsegment.segment(clean_text)
        corrected_text = " ".join(segmentation)
        print(f"WordSegment Result: {corrected_text}")
        return corrected_text


class MorseInput(InputSource):
    def __init__(self, button_pin=Config.GPIO_BUTTON_PIN, buzzer_pin=Config.GPIO_BUZZER_PIN):
        if Button is None:
            raise ImportError("gpiozero is required for MorseInput")

        self.button = Button(button_pin, pull_up=True, bounce_time=Config.BOUNCE_TIME)
        self.buzzer = Buzzer(buzzer_pin)
        self.press_time = 0
        self.buffer = ""
        self.current_message = ""
        self.message_queue = Queue()
        self.timer = None
        self.lock = Lock()
        self.closed = False

        self.button.when_pressed = self.handle_press
        self.button.when_released = self.handle_release
        print("Morse Code Input Ready.")

    def handle_press(self):
        if self.closed:
            return
        with self.lock:
            self.buzzer.on()
            if self.timer:
                self.timer.cancel()
            self.press_time = time.time()

    def handle_release(self):
        if self.closed:
            return
        with self.lock:
            self.buzzer.off()
            duration = time.time() - self.press_time
            self.buffer += "." if duration < Config.DOT_THRESHOLD else "-"
            self.timer = Timer(Config.LETTER_PAUSE, self.decode_current_buffer)
            self.timer.start()

    def decode_current_buffer(self):
        if self.closed:
            return
        with self.lock:
            char = MORSE_CODE_DICT.get(self.buffer, "")
            if char:
                print(char, end="", flush=True)
                self.current_message += char
            self.buffer = ""
            self.timer = Timer(Config.WORD_PAUSE - Config.LETTER_PAUSE, self.end_of_message_check)
            self.timer.start()

    def end_of_message_check(self):
        if self.closed:
            return
        with self.lock:
            if self.current_message and not self.current_message.endswith(" "):
                print(" ", end="", flush=True)
                self.current_message += " "
            self.timer = Timer(Config.SUBMIT_TIMEOUT, self.submit_message)
            self.timer.start()

    def submit_message(self):
        if self.closed:
            return
        with self.lock:
            final_msg = self.current_message.strip()
            if final_msg:
                print("\nSubmitting:", final_msg)
                self.message_queue.put(final_msg)
                self.current_message = ""

    def get_input(self) -> str:
        while not self.closed:
            try:
                return self.message_queue.get(timeout=0.2)
            except Empty:
                continue
        return ""

    def close(self):
        self.closed = True
        with self.lock:
            if self.timer:
                self.timer.cancel()
        try:
            self.message_queue.put_nowait("")
        except Exception:
            pass
        self.button.close()
        self.buzzer.close()


class TextInput(InputSource):
    def get_input(self) -> str:
        try:
            return input("\nEnter text query: ").strip()
        except EOFError:
            return ""

    def close(self):
        pass


class OCRInput(InputSource):
    """Button-triggered camera OCR input source for assistive usage."""

    def __init__(self, button_pin=22, buzzer_pin=None, camera_index=0):
        if ImageProcessor is None or OCREngine is None:
            raise ImportError("assistive_reader.ImageProcessor and OCREngine are required for OCRInput")

        self.closed = False
        self.button = None
        if button_pin is not None:
            if Button is None:
                raise ImportError("gpiozero is required for button-based OCRInput")
            self.button = Button(button_pin, pull_up=True, bounce_time=0.1)
        self.buzzer = Buzzer(buzzer_pin) if (buzzer_pin is not None and Buzzer is not None) else None
        self.cap = cv2.VideoCapture(camera_index)
        self.preview_enabled = True
        self.preview_window = "Live Camera Feed (OCR)"
        self.processed_window = "OCR Processed Frame"

        if not self.cap.isOpened():
            if self.button:
                self.button.close()
            if self.buzzer:
                self.buzzer.close()
            raise RuntimeError(f"Could not open camera index {camera_index}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        for _ in range(8):
            self.cap.read()

        self.image_processor = ImageProcessor()
        self.ocr_engine = OCREngine()
        if self.button:
            print(f"OCR input ready. Press hardware button on GPIO {button_pin} to capture.")
        else:
            print("OCR input ready. Press Space or Enter in the preview window to capture.")

    def _show_preview_frame(self, frame):
        if not self.preview_enabled:
            return
        try:
            preview = cv2.resize(frame, (960, 540))
            if self.button is None:
                instruction = "Press Space/Enter to capture | Q to cancel"
            else:
                instruction = "Press capture button | Q to cancel"
            cv2.putText(preview, instruction, (20, 35), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow(self.preview_window, preview)
        except cv2.error:
            self.preview_enabled = False

    def _wait_for_capture_with_preview(self):
        last_frame = None
        while not self.closed:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                last_frame = frame
                self._show_preview_frame(last_frame)

            key = cv2.waitKey(1) & 0xFF if self.preview_enabled else -1
            if key in (ord("q"), ord("Q"), 27):
                return None

            if self.button is None:
                if key in (13, 32):
                    return last_frame
            elif self.button.is_pressed:
                while self.button.is_pressed and not self.closed:
                    time.sleep(0.01)
                    if self.preview_enabled:
                        cv2.waitKey(1)
                return last_frame
        return None

    def _feedback(self, message: str, beep_seconds: float = 0.0):
        if self.buzzer and beep_seconds > 0:
            self.buzzer.on()
            time.sleep(beep_seconds)
            self.buzzer.off()
        print(message)

    def _wait_for_press_or_close(self):
        if self.button is None:
            try:
                input("Press Enter to capture image... ")
                return not self.closed
            except (EOFError, KeyboardInterrupt):
                return False
        while not self.closed:
            if self.button.wait_for_press(timeout=0.2):
                return True
        return False

    def get_input(self) -> str:
        self._feedback("Waiting for capture button press...")
        frame = self._wait_for_capture_with_preview() if self.preview_enabled else None

        if frame is None and not self.preview_enabled:
            if not self._wait_for_press_or_close():
                return ""
            if self.button is not None:
                while self.button.is_pressed and not self.closed:
                    time.sleep(0.01)
            if self.closed:
                return ""
            ret, frame = self.cap.read()
            if not ret:
                frame = None

        if self.closed or frame is None:
            return ""

        self._feedback("Photo captured. Processing text...", beep_seconds=0.08)

        processed = self.image_processor.preprocess(frame)
        if self.preview_enabled:
            try:
                cv2.imshow(self.processed_window, processed)
                cv2.waitKey(1)
            except cv2.error:
                self.preview_enabled = False

        ocr_result = self.ocr_engine.extract_text(processed)
        extracted_text = ocr_result[0] if isinstance(ocr_result, tuple) else ocr_result
        extracted_text = (extracted_text or "").strip()

        if extracted_text:
            self._feedback("Text extraction complete.", beep_seconds=0.05)
        else:
            self._feedback("No readable text found.", beep_seconds=0.15)
        return extracted_text

    def close(self):
        self.closed = True
        try:
            if self.cap:
                self.cap.release()
                cv2.destroyAllWindows()
        finally:
            if self.button:
                self.button.close()
            if self.buzzer:
                self.buzzer.close()


class VoiceInput(InputSource):
    """Push-to-talk speech input for assistive usage with GPIO button."""

    def __init__(
        self,
        button_pin=23,
        buzzer_pin=None,
        recognizer_backend="google",
        language="en-US",
        device_index=None,
    ):
        if sr is None:
            raise ImportError("SpeechRecognition is required for VoiceInput")

        self.closed = False
        self.button = None
        if button_pin is not None:
            if Button is None:
                raise ImportError("gpiozero is required for button-based VoiceInput")
            self.button = Button(button_pin, pull_up=True, bounce_time=0.05)
        self.buzzer = Buzzer(buzzer_pin) if (buzzer_pin is not None and Buzzer is not None) else None
        self.recognizer = sr.Recognizer()
        self.backend = recognizer_backend.lower().strip()
        self.language = language
        self.device_index = device_index
        if self.button:
            print(f"Voice input ready. Hold button on GPIO {button_pin} to talk.")
        else:
            print("Voice input ready. Press Enter to start listening.")

    def _feedback(self, message: str, beep_seconds: float = 0.0):
        if self.buzzer and beep_seconds > 0:
            self.buzzer.on()
            time.sleep(beep_seconds)
            self.buzzer.off()
        print(message)

    def _transcribe(self, audio) -> str:
        if self.backend == "vosk":
            if not hasattr(self.recognizer, "recognize_vosk"):
                self._feedback("Vosk backend not available in SpeechRecognition build.")
                return ""
            try:
                text = self.recognizer.recognize_vosk(audio)
                return (text or "").strip()
            except sr.UnknownValueError:
                self._feedback("Could not understand speech.")
                return ""
            except sr.RequestError as exc:
                self._feedback(f"Vosk recognition error: {exc}")
                return "Speech recognition engine error"

        try:
            text = self.recognizer.recognize_google(audio, language=self.language)
            return (text or "").strip()
        except sr.UnknownValueError:
            self._feedback("Could not understand speech.")
            return ""
        except sr.RequestError as exc:
            self._feedback(f"Speech recognition request failed: {exc}")
            return "Speech recognition service unavailable"

    def _wait_for_press_or_close(self):
        if self.button is None:
            try:
                input("Press Enter and speak... ")
                return not self.closed
            except (EOFError, KeyboardInterrupt):
                return False
        while not self.closed:
            if self.button.wait_for_press(timeout=0.2):
                return True
        return False

    def get_input(self) -> str:
        self._feedback("Hold button to speak. Release to transcribe.")
        if not self._wait_for_press_or_close():
            return ""

        with sr.Microphone(device_index=self.device_index) as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=0.2)
            self._feedback("Listening...", beep_seconds=0.05)

            if self.button is None:
                try:
                    audio = self.recognizer.listen(source, timeout=1.0, phrase_time_limit=12.0)
                except Exception:
                    return ""
            else:
                frames = []
                sample_rate = source.SAMPLE_RATE
                sample_width = source.SAMPLE_WIDTH
                while self.button.is_pressed and not self.closed:
                    frames.append(source.stream.read(source.CHUNK, exception_on_overflow=False))

                if self.closed:
                    return ""
                if not frames:
                    return ""
                audio = sr.AudioData(b"".join(frames), sample_rate, sample_width)
        self._feedback("Processing speech...", beep_seconds=0.05)
        return self._transcribe(audio)

    def close(self):
        self.closed = True
        if self.button:
            self.button.close()
        if self.buzzer:
            self.buzzer.close()
