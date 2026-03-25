import io
import os
import platform
import subprocess
import sys

from gtts import gTTS

# Add parent directory to path to import config if run from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config


class TTSEngine:
    def __init__(self, device=Config.TTS_DEVICE, volume=Config.TTS_VOLUME):
        self.device_name = device
        self.volume = volume
        self.system = platform.system().lower()
        self._pyttsx3_engine = None

        if self.system == "windows":
            print("Initializing TTS Engine (Windows SAPI backend)")
        else:
            print(f"Initializing TTS Engine on device: {self.device_name}")

    def _init_pyttsx3(self):
        import pyttsx3

        self._pyttsx3_engine = pyttsx3.init()
        self._pyttsx3_engine.setProperty("volume", max(0.0, min(1.0, self.volume / 100.0)))

    def _speak_windows(self, text):
        # Native SAPI via PowerShell is more reliable across repeated calls on Windows.
        safe_text = text.replace("'", "''")
        command = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$s.Volume = {int(max(0, min(100, self.volume)))}; "
            f"$s.Speak('{safe_text}')"
        )
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def _speak_linux_alsa(self, text):
        try:
            import alsaaudio
        except ImportError as exc:
            raise RuntimeError("pyalsaaudio is not installed") from exc

        mp3_fp = io.BytesIO()
        tts = gTTS(text=text, lang="en")
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        mp3_data = mp3_fp.read()

        cmd = [
            "ffmpeg",
            "-i",
            "pipe:0",
            "-f",
            "s16le",
            "-ac",
            "1",
            "-ar",
            str(Config.TTS_SAMPLE_RATE),
            "pipe:1",
            "-loglevel",
            "quiet",
        ]
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        pcm_data, stderr_data = process.communicate(input=mp3_data)
        if process.returncode != 0:
            raise RuntimeError(f"ffmpeg failed to decode audio: {stderr_data.decode(errors='ignore')}")

        try:
            device = alsaaudio.PCM(type=alsaaudio.PCM_PLAYBACK, device=self.device_name)
        except alsaaudio.ALSAAudioError:
            device = alsaaudio.PCM(type=alsaaudio.PCM_PLAYBACK, device="default")

        device.setchannels(1)
        device.setrate(Config.TTS_SAMPLE_RATE)
        device.setformat(alsaaudio.PCM_FORMAT_S16_LE)
        device.setperiodsize(320)

        chunk_size = 640
        for i in range(0, len(pcm_data), chunk_size):
            device.write(pcm_data[i : i + chunk_size])

    def _fallback_speak(self, text):
        if self.system == "windows":
            try:
                if self._pyttsx3_engine is None:
                    self._init_pyttsx3()
                self._pyttsx3_engine.say(text)
                self._pyttsx3_engine.runAndWait()
            except Exception:
                print(text)
            return

        try:
            subprocess.run(["espeak", text], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            print(text)

    def speak(self, text):
        if not text:
            return

        print(f"Speaking: {text}")
        try:
            if self.system == "windows":
                self._speak_windows(text)
            else:
                self._speak_linux_alsa(text)
        except Exception as e:
            print(f"TTS Error: {e}")
            self._fallback_speak(text)


if __name__ == "__main__":
    tts = TTSEngine()
    tts.speak("Hello from the cross platform TTS Engine")
