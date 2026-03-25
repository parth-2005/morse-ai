Based on my analysis of your repository, you have two somewhat separate systems right now:
1.  **The Main RAG System** (`main.py`, `src/`): Handles Morse code and text input, queries a LangChain RAG pipeline (Gemini + Ollama embeddings), and outputs via a custom ALSA/gTTS engine.
2.  **The Vision Reader** (`vision_reader/assistive_reader.py`): A standalone, production-grade script that captures images, processes them via Tesseract OCR, and sends the text directly to a local Ollama LLM for summarization, outputting via cross-platform TTS.

To build your unified system for a Raspberry Pi 5, you need to merge the OCR capabilities into your main input handler, add a new Voice Input module, and create a hardware-based mode-switching logic in `main.py`.

Here are the detailed, step-by-step prompts you can feed into GitHub Copilot (or any AI coding assistant) to generate the exact code you need.

### Prompt 1: Refactoring OCR into the Input Handler
**Goal:** Extract the camera and OCR logic from the standalone vision script and wrap it in your existing `InputSource` interface so `main.py` can use it just like Morse code.

**Copy and paste this into Copilot:**
> "I have an abstract base class `InputSource` in `src/input_handler.py`. I also have camera capture and OCR logic inside `vision_reader/assistive_reader.py` (specifically `CameraCapture`, `ImageProcessor`, and `OCREngine`). 
> 
> Please create a new class `OCRInput(InputSource)` in `src/input_handler.py`. 
> 1. It should initialize the camera using OpenCV (`cv2`). 
> 2. The `get_input()` method should block until a hardware button is pressed (use `gpiozero.Button` on a specific pin, say GPIO 22). 
> 3. Upon button press, it should capture a frame, run it through the `ImageProcessor.preprocess` pipeline, and extract text using `OCREngine.extract_text`. 
> 4. Return the extracted text as a string. Include appropriate audio feedback using a buzzer or print statements so a visually impaired user knows the photo was taken and is processing. Ensure it releases the camera properly in the `close()` method."

### Prompt 2: Adding Voice Input (Speech-to-Text)
**Goal:** Introduce a new input mechanism for voice, optimized for the Raspberry Pi.

**Copy and paste this into Copilot:**
> "Please create a new class called `VoiceInput(InputSource)` in `src/input_handler.py`. 
> 1. Use the `SpeechRecognition` Python library. 
> 2. The `get_input()` method should use `gpiozero.Button` (e.g., on GPIO 23) as a push-to-talk button. 
> 3. When the button is held down, record audio from the default ALSA microphone. 
> 4. When released, process the audio using `recognizer.recognize_google()` (or a fast local alternative like `vosk` if you suggest it for Raspberry Pi 5). 
> 5. Return the transcribed text. Handle `UnknownValueError` and `RequestError` gracefully, returning an empty string or an error message that the system can speak out loud."

### Prompt 3: Upgrading the RAG Engine to Handle General Context
**Goal:** Make sure the RAG engine can handle direct OCR text dumps, not just questions. If a user scans a document, they might just want it summarized, or they might want to ask a question *about* the scanned document.

**Copy and paste this into Copilot:**
> "In `src/rag_engine.py`, the `query` method currently assumes the input is a specific question to search against the vector database. I am now adding an OCR input mode. 
> 
> Please modify the `query(self, query_text, input_type="text")` method. 
> If `input_type` is 'ocr', the user has just scanned a document. Instead of searching the FAISS vector store, bypass the retriever and pass the OCR text directly to the Gemini LLM with a specific prompt instructing it to 'Summarize and explain the following scanned document for a visually impaired user: {query_text}'. 
> If `input_type` is 'morse' or 'voice', use the existing LCEL RAG chain to search the vector store. Update the LCEL chain definition if necessary to support this dynamic routing."

### Prompt 4: Unifying `main.py` with Hardware Mode Switching
**Goal:** Tie it all together so the user can physically switch between modes on the Raspberry Pi 5.

**Copy and paste this into Copilot:**
> "Rewrite my `main.py` to unify `MorseInput`, `VoiceInput`, and `OCRInput`. 
> 1. I am running this on a Raspberry Pi 5. I want to use a physical hardware switch or button to toggle between the three input modes. Use `gpiozero.Button` to implement a 'mode cycle' button (e.g., on GPIO 24). 
> 2. When the mode button is pressed, cycle the active input handler (Morse -> Voice -> OCR -> Morse) and use `tts.speak()` to announce the new mode to the user. 
> 3. The main `while True` loop should call `get_input()` on the *currently active* input handler. 
> 4. Pass the resulting query and the current mode type to the `rag.query()` method. 
> 5. Speak the answer using the existing `TTSEngine`. Ensure thread safety and that blocking calls in `get_input()` can be interrupted or cleanly exit if the mode changes."

### Raspberry Pi 5 Hardware Considerations (For You)
When executing these prompts, keep the following in mind for your Pi 5 setup:
* **Audio Setup:** The Pi 5 does not have a 3.5mm audio jack built-in. You will need a USB sound card, a Bluetooth speaker/mic (which your `config.py` seems to already target via `bluealsa`), or a GPIO DAC/I2S HAT.
* **Camera Interface:** The Pi 5 uses `libcamera` by default. OpenCV's standard `VideoCapture(0)` might struggle depending on your OS version (Bookworm). If Copilot's OpenCV code fails to grab a frame, tell Copilot: *"Update the OpenCV VideoCapture string to use the libcamera GStreamer pipeline for Raspberry Pi OS Bookworm."*
* **Local LLMs (Ollama):** The Pi 5 (especially the 8GB model) *can* run Ollama locally (`llama3.2:1b` or `qwen2.5:0.5b`), but inference will be slow (1-3 tokens/second). If speed is critical for the visually impaired user, stick to your Gemini API implementation in `src/rag_engine.py` for all text generation.