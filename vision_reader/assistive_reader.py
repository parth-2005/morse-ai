"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           ACCESSIBLE DOCUMENT READER — Production Grade                      ║
║           For Blind & Physically Disabled Users                              ║
║           OCR + LLM (Ollama) + TTS Pipeline                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

USAGE:
  python3 assistive_reader.py                         # Interactive camera mode
  python3 assistive_reader.py --image path/to/img     # From image file
  python3 assistive_reader.py --demo                  # Demo with a test image

REQUIREMENTS:  See requirements.txt
"""

import cv2
import numpy as np
import pytesseract
import requests
import json
import time
import argparse
import sys
import os
import platform
import subprocess
import threading
import logging
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import base64

# ──────────────────────────────────────────────────────────────
# CONFIGURATION  (edit these to match your hardware / model)
# ──────────────────────────────────────────────────────────────
from reader_config import (
    LLM_MODEL, LLM_TEMPERATURE, LLM_CONTEXT_SIZE,
    LLM_NUM_THREADS, LLM_REPEAT_PENALTY,
    OLLAMA_BASE_URL, TTS_RATE, TTS_VOLUME,
    CAMERA_INDEX, MIN_TEXT_LENGTH, OCR_CONFIDENCE_THRESHOLD,
    LOG_LEVEL, SAVE_INTERMEDIATE_IMAGES
)

# ──────────────────────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("accessible_reader.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# TEXT-TO-SPEECH ENGINE  (cross-platform, truly blocking)
# ══════════════════════════════════════════════════════════════
class TTSEngine:
    """
    Cross-platform TTS that ACTUALLY blocks until speech finishes.

    On macOS, it uses the built-in `say` command which is reliable and high-quality.
    On Linux, it tries `espeak-ng` first (more modern), then falls back
    to `espeak` if `espeak-ng` is not available.
    On Windows, it uses `pyttsx3` which interfaces with SAPI5.
    All methods are thread-safe and will block until the audio has finished playing
    before returning, ensuring proper synchronization with the LLM response streaming.
    """

    def __init__(self):
        self._os = platform.system()          # 'Darwin', 'Linux', 'Windows'
        self._lock = threading.Lock()
        self._rate = TTS_RATE
        self._volume = TTS_VOLUME
        self._engine = None                   # only used on Windows

        if self._os == "Windows":
            self._init_windows()

        log.info("TTS backend: %s (%s)", self._os, self._backend_name())

    # ── init ───────────────────────────────────────────────
    def _init_windows(self):
        import pyttsx3
        self._engine = pyttsx3.init()
        self._engine.setProperty("rate",   self._rate)
        self._engine.setProperty("volume", self._volume)
        voices = self._engine.getProperty("voices")
        for v in voices:
            if "english" in v.name.lower() or "en_" in (v.id or "").lower():
                self._engine.setProperty("voice", v.id)
                break

    def _backend_name(self) -> str:
        if self._os == "Darwin":  return "macOS say"
        if self._os == "Linux":   return "espeak-ng / espeak"
        return "pyttsx3 (Windows SAPI)"

    # ── public API ─────────────────────────────────────────
    def speak(self, text: str):
        """Speak *text* and BLOCK until playback is fully complete."""
        text = text.strip()
        if not text:
            return
        log.info("TTS ▶ %s", text[:120])
        with self._lock:
            if self._os == "Darwin":
                self._speak_macos(text)
            elif self._os == "Linux":
                self._speak_linux(text)
            else:
                self._speak_windows(text)

    # ── platform implementations ───────────────────────────
    def _speak_macos(self, text: str):
        """
        macOS `say` blocks the subprocess until audio finishes — perfectly reliable.
        Rate: `say` accepts words-per-minute via -r flag.
        """
        # Sanitise: escape double-quotes for the shell
        safe = text.replace('"', '\\"')
        cmd = ["say", "-r", str(self._rate), safe]
        try:
            subprocess.run(cmd, check=True, timeout=300)
        except subprocess.TimeoutExpired:
            log.warning("TTS timeout on: %s", text[:60])
        except Exception as e:
            log.error("macOS TTS error: %s", e)

    def _speak_linux(self, text: str):
        """Use espeak-ng if available, fall back to espeak."""
        # words-per-minute → espeak speed (roughly 1:1 mapping)
        speed = max(80, min(self._rate, 450))
        for binary in ("espeak-ng", "espeak"):
            try:
                subprocess.run(
                    [binary, "-s", str(speed), "-a", str(int(self._volume * 200)), text],
                    check=True, timeout=300
                )
                return
            except FileNotFoundError:
                continue
            except subprocess.TimeoutExpired:
                log.warning("TTS timeout on: %s", text[:60])
                return
            except Exception as e:
                log.error("Linux TTS error (%s): %s", binary, e)
                return
        log.error("No TTS binary found. Install espeak-ng: sudo apt install espeak-ng")

    def _speak_windows(self, text: str):
        """
        pyttsx3 on Windows (SAPI5) does block correctly.
        We still run it in an isolated thread + Event to be safe.
        """
        done = threading.Event()

        def _run():
            try:
                self._engine.say(text)
                self._engine.runAndWait()
            finally:
                done.set()

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        done.wait(timeout=300)   # never hang forever


# ══════════════════════════════════════════════════════════════
# IMAGE PRE-PROCESSING  (high-accuracy OCR pipeline)
# ══════════════════════════════════════════════════════════════
class ImageProcessor:
    """
    Multi-stage image enhancement designed for document capture under
    real-world lighting conditions (shadows, glare, skew, low contrast).
    """

    def __init__(self):
        self.debug_dir = Path("debug_images")
        if SAVE_INTERMEDIATE_IMAGES:
            self.debug_dir.mkdir(exist_ok=True)

    # ── public entry point ──────────────────────────────────
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Full pipeline:
          1. Upscale  →  2. Deskew  →  3. Denoise  →  4. Adaptive threshold
          →  5. Morphological clean-up  →  6. Border crop
        Returns a binary (black/white) image ready for Tesseract.
        """
        img = self._upscale(image)
        img = self._correct_skew(img)
        img = self._denoise(img)
        img = self._binarise(img)
        img = self._morph_cleanup(img)
        img = self._crop_border(img)

        if SAVE_INTERMEDIATE_IMAGES:
            path = self.debug_dir / f"processed_{int(time.time())}.png"
            cv2.imwrite(str(path), img)
            log.debug("Saved processed image → %s", path)

        return img

    # ── private helpers ──────────────────────────────────────
    @staticmethod
    def _upscale(img: np.ndarray) -> np.ndarray:
        """Upscale small captures so Tesseract has enough pixel density."""
        h, w = img.shape[:2]
        target_w = max(w, 2400)      # at least 2400 px wide
        if w < target_w:
            scale = target_w / w
            img = cv2.resize(img, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_CUBIC)
            log.debug("Upscaled ×%.2f → %dx%d", scale, img.shape[1], img.shape[0])
        return img

    @staticmethod
    def _correct_skew(img: np.ndarray) -> np.ndarray:
        """Detect and correct page tilt using Hough-line angle estimation."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=200)

        if lines is None:
            return img

        angles = []
        for rho, theta in lines[:, 0]:
            angle = (theta * 180 / np.pi) - 90
            if abs(angle) < 45:
                angles.append(angle)

        if not angles:
            return img

        median_angle = float(np.median(angles))
        if abs(median_angle) < 0.5:   # negligible skew
            return img

        log.debug("Correcting skew: %.2f°", median_angle)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), median_angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h),
                                 flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE)
        return rotated

    @staticmethod
    def _denoise(img: np.ndarray) -> np.ndarray:
        """Remove sensor noise while preserving text edges."""
        if img.ndim == 3:
            return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        return cv2.fastNlMeansDenoising(img, None, 10, 7, 21)

    @staticmethod
    def _binarise(img: np.ndarray) -> np.ndarray:
        """
        Adaptive Gaussian threshold on a CLAHE-equalised grayscale image.
        Far more robust than Otsu for uneven lighting (e.g. hand-held camera).
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

        # CLAHE for local contrast normalisation
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # Adaptive threshold
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=31,
            C=15,
        )
        return binary

    @staticmethod
    def _morph_cleanup(img: np.ndarray) -> np.ndarray:
        """Closing + opening to remove small speckles and fill gaps in letters."""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN,  kernel)
        return img

    @staticmethod
    def _crop_border(img: np.ndarray, margin: int = 20) -> np.ndarray:
        """Add a thin white border — helps Tesseract not clip edge characters."""
        return cv2.copyMakeBorder(img, margin, margin, margin, margin,
                                  cv2.BORDER_CONSTANT, value=255)


# ══════════════════════════════════════════════════════════════
# OCR ENGINE
# ══════════════════════════════════════════════════════════════
class OCREngine:
    """
    Runs Tesseract in multiple passes (different PSM modes) and picks
    the highest-confidence result.
    """

    # Page Segmentation Modes to try in order of preference
    PSM_MODES = [
        ("Auto OSD",     "--oem 3 --psm 1"),
        ("Auto",         "--oem 3 --psm 3"),
        ("Sparse text",  "--oem 3 --psm 11"),
        ("Single block", "--oem 3 --psm 6"),
    ]

    def extract_text(self, image: np.ndarray) -> tuple[str, float]:
        """
        Returns (best_text, confidence_score).
        Tries multiple PSM modes and returns the result with the
        most high-confidence words.
        """
        best_text = ""
        best_conf = 0.0

        for label, config in self.PSM_MODES:
            try:
                data = pytesseract.image_to_data(
                    image,
                    config=config,
                    lang="eng",
                    output_type=pytesseract.Output.DICT,
                )
                words = [
                    w for w, c in zip(data["text"], data["conf"])
                    if isinstance(c, (int, float)) and int(c) >= OCR_CONFIDENCE_THRESHOLD
                    and str(w).strip()
                ]
                text = " ".join(words)
                avg_conf = (
                    float(np.mean([
                        int(c) for c in data["conf"]
                        if isinstance(c, (int, float)) and int(c) > 0
                    ])) if any(int(c) > 0 for c in data["conf"]
                              if isinstance(c, (int, float))) else 0.0
                )

                log.debug("PSM %-14s → %d words, avg conf %.1f", label, len(words), avg_conf)

                if avg_conf > best_conf:
                    best_conf = avg_conf
                    best_text = text

            except Exception as exc:
                log.warning("OCR pass '%s' failed: %s", label, exc)

        return best_text.strip(), round(best_conf, 1)


# ══════════════════════════════════════════════════════════════
# LLM ENGINE  (Ollama)
# ══════════════════════════════════════════════════════════════
class LLMEngine:
    """
    Sends OCR text to a locally-running Ollama model and streams
    the contextual explanation back.
    """

    SYSTEM_PROMPT = (
        "You are a helpful assistant for visually impaired and physically disabled users. "
        "You will be given raw text extracted from a physical document via OCR. "
        "Your task is to:\n"
        "1. Understand what the document is (letter, receipt, form, book page, prescription, notice, etc.).\n"
        "2. Explain its content clearly and concisely in plain spoken language.\n"
        "3. Highlight any important details, deadlines, names, amounts, or actions required.\n"
        "4. Keep your response natural-sounding because it will be read aloud by a text-to-speech engine.\n"
        "5. Do NOT use bullet points, markdown, or special characters.\n"
        "6. Speak as if you are talking directly to the person holding the document.\n"
        "7. If the text is unclear or incomplete, say so gently and describe what you can infer."
    )

    def explain(self, raw_text: str, tts: TTSEngine) -> str:
        """
        Call Ollama /api/generate with streaming enabled.
        Speaks sentence chunks aloud as they arrive so the user gets
        progressive feedback without waiting for the full response.
        """
        url = f"{OLLAMA_BASE_URL}/api/generate"
        payload = {
            "model":  LLM_MODEL,
            "system": self.SYSTEM_PROMPT,
            "prompt": (
                f"Here is the text extracted from the document:\n\n"
                f"{raw_text}\n\n"
                f"Please explain what this document says."
            ),
            "stream": True,
            "options": {
                "temperature":    LLM_TEMPERATURE,
                "num_ctx":        LLM_CONTEXT_SIZE,
                "num_thread":     LLM_NUM_THREADS,
                "repeat_penalty": LLM_REPEAT_PENALTY,
            },
        }

        log.info("Sending %d chars to LLM (%s)…", len(raw_text), LLM_MODEL)

        full_response = ""
        sentence_buffer = ""
        sentence_enders = {".", "!", "?", "…"}

        try:
            with requests.post(url, json=payload, stream=True, timeout=120) as resp:
                resp.raise_for_status()

                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    token = chunk.get("response", "")
                    full_response    += token
                    sentence_buffer  += token

                    # Speak each complete sentence as soon as it arrives
                    if any(c in sentence_buffer for c in sentence_enders):
                        sentences = self._split_sentences(sentence_buffer)
                        for s in sentences[:-1]:   # all but the trailing fragment
                            if s.strip():
                                tts.speak(s.strip())
                        sentence_buffer = sentences[-1]   # keep the fragment

                    if chunk.get("done", False):
                        break

            # Speak any remaining text
            if sentence_buffer.strip():
                tts.speak(sentence_buffer.strip())

        except requests.exceptions.ConnectionError:
            msg = (
                "I could not connect to the Ollama service. "
                "Please make sure Ollama is running on your computer."
            )
            log.error(msg)
            tts.speak(msg)
            return ""
        except Exception as exc:
            msg = f"An error occurred while processing with the language model: {exc}"
            log.error(msg)
            tts.speak("An unexpected error occurred while explaining the document.")
            return ""

        return full_response.strip()

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Very fast sentence splitter — good enough for TTS chunking."""
        import re
        parts = re.split(r'(?<=[.!?…])\s+', text)
        return parts if parts else [text]


# ══════════════════════════════════════════════════════════════
# CAMERA CAPTURE
# ══════════════════════════════════════════════════════════════
class CameraCapture:
    """Captures a stable, well-lit frame from the default camera."""

    def __init__(self):
        self.cap = None

    def open(self) -> bool:
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self.cap.isOpened():
            log.error("Could not open camera index %d", CAMERA_INDEX)
            return False
        # Warm up — discard the first few frames (auto-exposure settling)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        for _ in range(10):
            self.cap.read()
        return True

    def capture(self) -> np.ndarray | None:
        if self.cap is None:
            return None
        ret, frame = self.cap.read()
        return frame if ret else None

    def release(self):
        if self.cap:
            self.cap.release()
            cv2.destroyAllWindows()


# ══════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ══════════════════════════════════════════════════════════════
class AccessibleDocumentReader:
    """
    Orchestrates the full pipeline:
      Camera/image  →  Pre-process  →  OCR  →  LLM  →  TTS
    with step-by-step spoken guidance for the end user.
    """

    def __init__(self):
        log.info("Initialising Accessible Document Reader…")
        self.tts      = TTSEngine()
        self.img_proc = ImageProcessor()
        self.ocr      = OCREngine()
        self.llm      = LLMEngine()
        self.camera   = CameraCapture()

    # ── guided workflow ──────────────────────────────────────
    def run_interactive(self):
        """Full guided session using the camera."""
        self._welcome()

        if not self.camera.open():
            self.tts.speak(
                "I could not access the camera. "
                "Please check that a camera is connected and try again."
            )
            return

        try:
            while True:
                self.tts.speak(
                    "When you are ready, hold your document flat under the camera "
                    "and press the SPACE bar to scan, or press Q to quit."
                )
                log.info("Waiting for keypress…")

                frame = None
                while True:
                    frame = self.camera.capture()
                    if frame is not None:
                        preview = cv2.resize(frame, (800, 600))
                        cv2.imshow("Accessible Document Reader  |  SPACE = scan  Q = quit",
                                   preview)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord(" "):
                        break
                    if key in (ord("q"), ord("Q"), 27):   # Q or ESC
                        self.tts.speak("Thank you for using the Accessible Document Reader. Goodbye.")
                        return

                if frame is None:
                    self.tts.speak("I could not capture an image. Please try again.")
                    continue

                self._process_image(frame)

        finally:
            self.camera.release()

    def run_from_file(self, path: str):
        """Process a saved image file directly."""
        self._welcome(camera_mode=False)

        img = cv2.imread(path)
        if img is None:
            msg = f"Could not read image file: {path}"
            log.error(msg)
            self.tts.speak("I could not open the image file you provided.")
            return

        self.tts.speak(f"I have loaded the image file. Let me begin scanning it now.")
        self._process_image(img)

    # ── core pipeline ────────────────────────────────────────
    def _process_image(self, image: np.ndarray):
        # Step 1 — pre-process
        self.tts.speak(
            "Step one. I am enhancing the image to improve text clarity. "
            "This may take a few seconds."
        )
        log.info("Pre-processing image…")
        processed = self.img_proc.preprocess(image)

        # Step 2 — OCR
        self.tts.speak(
            "Step two. I am now scanning the text from the document. "
            "Please keep the document still if you are using a camera."
        )
        log.info("Running OCR…")
        raw_text, confidence = self.ocr.extract_text(processed)

        log.info("OCR complete. Characters extracted: %d, Confidence: %.1f%%",
                 len(raw_text), confidence)

        if not raw_text or len(raw_text) < MIN_TEXT_LENGTH:
            self.tts.speak(
                "I was unable to find enough readable text on the document. "
                "Please ensure the document is well-lit, flat, and fully visible, "
                "then try again."
            )
            return

        self.tts.speak(
            f"Text scanning complete. I detected approximately {len(raw_text.split())} words "
            f"with a confidence of {int(confidence)} percent. "
        )

        if confidence < 50:
            self.tts.speak(
                "The confidence is lower than usual. "
                "The explanation may not be fully accurate. "
                "For best results, ensure good lighting and a clear, flat document."
            )

        # Step 3 — LLM explanation
        self.tts.speak(
            "Step three. I am now sending the text to the language model for understanding. "
            "I will read the explanation to you as it is being generated. "
            "Please listen carefully."
        )
        log.info("Calling LLM…")
        explanation = self.llm.explain(raw_text, self.tts)

        if explanation:
            self.tts.speak(
                "That is the full explanation of your document. "
                "If you would like to scan another document, press SPACE. "
                "To quit, press Q."
            )
        else:
            self.tts.speak(
                "I was unable to generate an explanation. "
                "Please check that Ollama is running with the correct model, then try again."
            )

        # Optionally log the extracted text for review
        log.info("═══ RAW OCR TEXT ═══\n%s", raw_text)
        log.info("═══ LLM EXPLANATION ═══\n%s", explanation)

    # ── welcome message ──────────────────────────────────────
    def _welcome(self, camera_mode: bool = True):
        msg = (
            "Welcome to the Accessible Document Reader. "
            "This tool will scan text from a physical document, "
            "understand its content using artificial intelligence, "
            "and explain it to you in plain spoken language. "
        )
        if camera_mode:
            msg += (
                "To use this tool: "
                "First, place your document flat in front of the camera. "
                "Second, press the SPACE bar when you are ready to scan. "
                "The system will guide you through each step. "
            )
        self.tts.speak(msg)


# ══════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Accessible Document Reader for blind / physically disabled users."
    )
    parser.add_argument("--image", type=str, default=None,
                        help="Path to an image file to process (skips camera).")
    parser.add_argument("--demo",  action="store_true",
                        help="Run a quick demo with a generated test image.")
    args = parser.parse_args()

    reader = AccessibleDocumentReader()

    if args.demo:
        _run_demo(reader)
    elif args.image:
        reader.run_from_file(args.image)
    else:
        reader.run_interactive()


def _run_demo(reader: AccessibleDocumentReader):
    """Create a synthetic test image and run the pipeline on it."""
    from PIL import Image as PILImage, ImageDraw, ImageFont
    import tempfile

    log.info("Creating demo image…")
    img = PILImage.new("RGB", (800, 600), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    sample_text = (
        "PRESCRIPTION\n\n"
        "Patient: John Smith\n"
        "Date: 22 March 2026\n\n"
        "Medication: Amoxicillin 500mg\n"
        "Dosage: Take one capsule three times daily\n"
        "Duration: 7 days\n\n"
        "Doctor: Dr. Priya Mehta\n"
        "Hospital: City Medical Centre\n\n"
        "IMPORTANT: Complete the full course.\n"
        "Do not stop early even if you feel better."
    )

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 26)
    except Exception:
        font = ImageFont.load_default()

    draw.multiline_text((40, 40), sample_text, fill=(0, 0, 0), font=font, spacing=8)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        img.save(tmp.name)
        demo_path = tmp.name

    log.info("Demo image saved → %s", demo_path)
    reader.run_from_file(demo_path)
    os.unlink(demo_path)


if __name__ == "__main__":
    main()