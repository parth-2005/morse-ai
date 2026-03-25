import argparse
import os
import platform
import queue
import threading
import time

try:
    from gpiozero import Button
except ImportError:
    Button = None

from src.input_handler import InputPostProcessor, MorseInput, OCRInput, TextInput, VoiceInput
from src.rag_engine import RAGEngine
from src.tts_engine import TTSEngine


class InputWorker:
    def __init__(self, mode_name, input_handler, output_queue):
        self.mode_name = mode_name
        self.input_handler = input_handler
        self.output_queue = output_queue
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def _run(self):
        while not self.stop_event.is_set():
            try:
                text = self.input_handler.get_input()
            except Exception as exc:
                if self.stop_event.is_set():
                    return
                self.output_queue.put({"mode": self.mode_name, "text": "", "error": str(exc)})
                time.sleep(0.1)
                continue

            if self.stop_event.is_set():
                return
            if text:
                self.output_queue.put({"mode": self.mode_name, "text": text, "error": None})

    def stop(self):
        self.stop_event.set()
        try:
            self.input_handler.close()
        except Exception:
            pass
        self.thread.join(timeout=2.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", action="store_true", help="Use keyboard text mode only")
    parser.add_argument(
        "--mode",
        choices=["auto", "text", "morse", "voice", "ocr"],
        default="auto",
        help="Select input mode. Use auto for GPIO cycling on Linux/Raspberry Pi.",
    )
    parser.add_argument("--voice-backend", choices=["google", "vosk"], default="google")
    args = parser.parse_args()

    print("Initializing Educational RAG System...")

    try:
        tts = TTSEngine()
    except Exception as e:
        print(f"Error initializing TTS: {e}")
        return

    tts_lock = threading.Lock()

    def safe_speak(text):
        if not text:
            return
        with tts_lock:
            tts.speak(text)

    safe_speak("System initializing")

    try:
        print("Loading Knowledge Base...")
        rag = RAGEngine()
        if not os.path.exists("data"):
            os.makedirs("data")
            print("Warning: data directory created but empty.")
            safe_speak("Data directory is empty. Please add text files.")
        elif not rag.chain:
            print("Warning: retrieval chain not initialized. OCR mode can still work.")
            safe_speak("Knowledge base retrieval is not ready. OCR mode is still available.")
        else:
            safe_speak("Knowledge base loaded")
    except Exception as e:
        print(f"Error initializing RAG: {e}")
        safe_speak("Error loading knowledge base")
        return

    post_processor = InputPostProcessor()
    result_queue = queue.Queue()
    state_lock = threading.Lock()

    is_gpio_env = Button is not None and platform.system().lower() != "windows"
    selected_mode = "text" if args.text else args.mode

    if selected_mode == "auto":
        mode_order = ["morse", "voice", "ocr"] if is_gpio_env else ["text"]
        if not is_gpio_env:
            print("GPIO mode switching is unavailable on this platform. Defaulting to text mode.")
            safe_speak("GPIO is unavailable. Defaulting to text mode.")
    else:
        mode_order = [selected_mode]

    mode_index = 0
    input_worker = None
    mode_button = None

    def build_handler(mode_name):
        if mode_name == "morse":
            return MorseInput(button_pin=17, buzzer_pin=27)
        if mode_name == "voice":
            if is_gpio_env:
                return VoiceInput(button_pin=23, recognizer_backend=args.voice_backend)
            return VoiceInput(button_pin=None, recognizer_backend=args.voice_backend)
        if mode_name == "ocr":
            if is_gpio_env:
                return OCRInput(button_pin=22, buzzer_pin=27, camera_index=0)
            return OCRInput(button_pin=None, buzzer_pin=None, camera_index=0)
        return TextInput()

    def activate_mode(new_mode_name):
        nonlocal input_worker
        if input_worker is not None:
            input_worker.stop()
            input_worker = None

        try:
            handler = build_handler(new_mode_name)
        except Exception as exc:
            print(f"Mode '{new_mode_name}' is unavailable: {exc}")
            return False

        input_worker = InputWorker(new_mode_name, handler, result_queue)
        input_worker.start()
        print(f"Active mode: {new_mode_name}")
        safe_speak(f"{new_mode_name} mode")
        return True

    def cycle_mode():
        nonlocal mode_index
        with state_lock:
            for _ in range(len(mode_order)):
                mode_index = (mode_index + 1) % len(mode_order)
                new_mode = mode_order[mode_index]
                if activate_mode(new_mode):
                    return
            print("No input mode is currently available.")
            safe_speak("No input mode is available")

    if len(mode_order) > 1:
        if not is_gpio_env:
            print("Mode cycle button is unavailable on this platform.")
            safe_speak("Mode button is unavailable")
            return
        mode_button = Button(24, pull_up=True, bounce_time=0.2)
        mode_button.when_pressed = cycle_mode

    try:
        if not activate_mode(mode_order[mode_index]):
            for idx, mode in enumerate(mode_order):
                if activate_mode(mode):
                    mode_index = idx
                    break
            else:
                print("Could not initialize any input mode.")
                safe_speak("Could not initialize any input mode")
                return
    except Exception as e:
        print(f"Error initializing first input mode: {e}")
        safe_speak("Error initializing input mode")
        return

    if len(mode_order) > 1:
        print("\nSystem ready. Press mode button on GPIO 24 to cycle modes. Ctrl+C to exit.")
    else:
        print(f"\nSystem ready in {mode_order[0]} mode. Ctrl+C to exit.")

    try:
        while True:
            try:
                item = result_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            mode_name = item.get("mode")
            query_text = (item.get("text") or "").strip()
            error_text = item.get("error")

            with state_lock:
                active_mode = mode_order[mode_index]

            if mode_name != active_mode:
                continue

            if error_text:
                print(f"Input error ({mode_name}): {error_text}")
                safe_speak("Input error")
                continue

            if not query_text:
                continue

            print(f"\nReceived ({mode_name}): {query_text}")
            safe_speak("Received input")

            query_for_rag = query_text
            if mode_name == "morse":
                query_for_rag = post_processor.process_input(query_text)
                print(f"Corrected Morse Query: {query_for_rag}")

            answer = rag.query(query_for_rag, input_type=mode_name)
            print(f"Answer: {answer}")
            safe_speak(answer)

    except KeyboardInterrupt:
        print("\nExiting...")
        safe_speak("System shutting down")
    finally:
        if mode_button:
            mode_button.close()
        if input_worker:
            try:
                input_worker.stop()
            except KeyboardInterrupt:
                pass


if __name__ == "__main__":
    main()
