import io
import os
import subprocess
import alsaaudio
import sys
# Add parent directory to path to import config if run from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gtts import gTTS
from config import Config

class TTSEngine:
    def __init__(self, device=Config.TTS_DEVICE, volume=Config.TTS_VOLUME):
        self.device_name = device
        self.volume = volume
        print(f"Initializing TTS Engine on device: {self.device_name}")

    def speak(self, text):
        """Processes text to audio entirely in RAM using gTTS, ffmpeg and ALSA."""
        if not text:
            return

        print(f"Speaking: {text}")
        try:
            # 1. Generate Speech (In-Memory Buffer)
            mp3_fp = io.BytesIO()
            # Generate speech using gTTS
            tts = gTTS(text=text, lang='en')
            tts.write_to_fp(mp3_fp)
            mp3_fp.seek(0)
            mp3_data = mp3_fp.read()

            # 2. Decode MP3 to PCM (In-Memory) using ffmpeg subprocess
            # We enforce Config.TTS_SAMPLE_RATE mono S16LE to match ALSA config easily
            # -i pipe:0 : Read from stdin
            # -f s16le : Output format signed 16-bit little endian
            # -ac 1 : Mono (1 channel)
            # -ar rate : Sample rate
            # pipe:1 : Write to stdout
            # -loglevel quiet : Suppress ffmpeg output
            
            cmd = ['ffmpeg', '-i', 'pipe:0', '-f', 's16le', '-ac', '1', '-ar', str(Config.TTS_SAMPLE_RATE), 'pipe:1', '-loglevel', 'quiet']
            
            process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            pcm_data, stderr_data = process.communicate(input=mp3_data)
            
            if process.returncode != 0:
                print(f"ffmpeg error: {stderr_data.decode()}")
                raise RuntimeError("ffmpeg failed to decode audio")

            # 3. Initialize ALSA Hardware Connection
            try:
                device = alsaaudio.PCM(
                    type=alsaaudio.PCM_PLAYBACK,
                    device=self.device_name
                )
            except alsaaudio.ALSAAudioError as e:
                print(f"Error opening ALSA device '{self.device_name}': {e}")
                print("Falling back to 'default' device...")
                device = alsaaudio.PCM(
                    type=alsaaudio.PCM_PLAYBACK,
                    device='default'
                )
            
            # 4. Set Hardware Parameters (matching ffmpeg output)
            device.setchannels(1)
            device.setrate(Config.TTS_SAMPLE_RATE)
            device.setformat(alsaaudio.PCM_FORMAT_S16_LE)
            device.setperiodsize(320)

            # 5. Stream Raw Bytes to Hardware
            chunk_size = 640  # 320 frames * 2 bytes per sample
            for i in range(0, len(pcm_data), chunk_size):
                device.write(pcm_data[i:i+chunk_size])
            
        except Exception as e:
            print(f"TTS Error: {e}")
            # Fallback to espeak if gTTS/ffmpeg fails
            print("Falling back to espeak...")
            os.system(f"espeak '{text}' 2>/dev/null")

if __name__ == "__main__":
    tts = TTSEngine()
    tts.speak("Hello from the new ffmpeg TTS Engine")
