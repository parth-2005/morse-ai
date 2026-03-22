"""
reader_config.py  —  Accessible Document Reader
════════════════════════════════════════
Edit this file to tune every aspect of the system.
No other file needs to be touched for basic customisation.
"""

# ─────────────────────────────────────────────────────────────
# OLLAMA / LLM
# ─────────────────────────────────────────────────────────────

# Base URL for the Ollama API server.
# Change the host/port only if you run Ollama on a different machine.
OLLAMA_BASE_URL = "http://localhost:11434"

# Model name served by Ollama (e.g., "llama3.2:1b", "mistral", "llama2")
LLM_MODEL = "llama3.2:3b"

# Temperature (0.0 to 1.0)
# Lower values (e.g., 0.1) make the output more deterministic and focused.
# Higher values (e.g., 0.8) make the output more creative and random.
LLM_TEMPERATURE = 0.5

# Context text size (num_ctx / num_main_tokens)
# The maximum number of tokens the model can process (prompt + context + generation).
# Increase this if you have very long documents, but it requires more RAM.
LLM_CONTEXT_SIZE = 2048

# Number of CPU threads to use for generation.
# Set to the number of physical CPU cores you want to dedicate.
LLM_NUM_THREADS = 4

# Repeat penalty — penalise the model for repeating the same tokens.
LLM_REPEAT_PENALTY = 1.1


# ─────────────────────────────────────────────────────────────
# TEXT-TO-SPEECH
# ─────────────────────────────────────────────────────────────

# Words per minute (typical human speech ≈ 150).
# Lower for disabled users who need more time to process.
TTS_RATE = 155

# Volume 0.0 – 1.0
TTS_VOLUME = 1.0


# ─────────────────────────────────────────────────────────────
# CAMERA
# ─────────────────────────────────────────────────────────────

# OpenCV camera index. 0 = default webcam, 1 = second camera, etc.
CAMERA_INDEX = 0


# ─────────────────────────────────────────────────────────────
# OCR QUALITY CONTROL
# ─────────────────────────────────────────────────────────────

# Minimum number of characters needed before passing text to the LLM.
# Scans with fewer characters are rejected as "too little text found".
MIN_TEXT_LENGTH = 20

# Minimum per-word confidence (0–100) for a word to be included in the output.
# Words scoring below this threshold are discarded to reduce noise.
OCR_CONFIDENCE_THRESHOLD = 40


# ─────────────────────────────────────────────────────────────
# DEBUGGING / DEVELOPMENT
# ─────────────────────────────────────────────────────────────

# Logging level: "DEBUG", "INFO", "WARNING", "ERROR"
LOG_LEVEL = "INFO"

# Save intermediate processed images to ./debug_images/ for inspection.
SAVE_INTERMEDIATE_IMAGES = False