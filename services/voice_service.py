import os
import io
import wave
import tempfile
from typing import Tuple, Optional
from utils.logger import logger


def transcribe_audio_file(file_path: str) -> Tuple[str, float]:
    """
    Transcribe audio file to text using SpeechRecognition.
    Returns (transcript, confidence).
    """
    try:
        import speech_recognition as sr
        recognizer = sr.Recognizer()
        recognizer.energy_threshold = 300
        recognizer.dynamic_energy_threshold = True

        with sr.AudioFile(file_path) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.record(source)

        # Try Google Speech (free, no API key needed)
        try:
            text = recognizer.recognize_google(audio, show_all=False)
            return text, 0.90
        except sr.UnknownValueError:
            return "", 0.0
        except sr.RequestError as e:
            logger.error(f"Google Speech API error: {e}")
            # Try Sphinx (offline) as fallback
            try:
                text = recognizer.recognize_sphinx(audio)
                return text, 0.60
            except Exception:
                return "", 0.0

    except ImportError:
        logger.error("speech_recognition not installed. pip install SpeechRecognition")
        return "", 0.0
    except Exception as e:
        logger.error(f"Audio transcription error: {e}")
        return "", 0.0


def transcribe_microphone(duration: int = 5, language: str = "en-US") -> Tuple[str, float]:
    """
    Record from microphone and transcribe.
    Returns (transcript, confidence).
    """
    try:
        import speech_recognition as sr
        recognizer = sr.Recognizer()

        with sr.Microphone() as source:
            logger.info("Adjusting for ambient noise...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            logger.info(f"Recording for {duration} seconds...")
            audio = recognizer.listen(source, timeout=duration, phrase_time_limit=duration)

        text = recognizer.recognize_google(audio, language=language)
        return text, 0.90

    except ImportError:
        logger.error("speech_recognition or PyAudio not installed.")
        return "", 0.0
    except Exception as e:
        logger.error(f"Microphone transcription error: {e}")
        return "", 0.0


def convert_audio_to_wav(input_path: str, output_path: Optional[str] = None) -> str:
    """Convert mp3/ogg/flac to wav using pydub."""
    if output_path is None:
        base = os.path.splitext(input_path)[0]
        output_path = base + "_converted.wav"

    ext = os.path.splitext(input_path)[1].lower().lstrip(".")

    try:
        from pydub import AudioSegment
        formats = {"mp3": "mp3", "ogg": "ogg", "flac": "flac", "m4a": "m4a", "aac": "aac"}
        fmt = formats.get(ext, ext)
        audio = AudioSegment.from_file(input_path, format=fmt)
        audio = audio.set_channels(1).set_frame_rate(16000)
        audio.export(output_path, format="wav")
        return output_path
    except ImportError:
        logger.warning("pydub not installed — returning input path as-is")
        return input_path
    except Exception as e:
        logger.error(f"Audio conversion error: {e}")
        return input_path


def save_upload_audio(file_storage, upload_dir: str = "data/uploads") -> str:
    """Save a Flask FileStorage object and return the saved path."""
    os.makedirs(upload_dir, exist_ok=True)
    filename = file_storage.filename
    safe_name = "".join(c for c in filename if c.isalnum() or c in "._-")
    path = os.path.join(upload_dir, safe_name)
    file_storage.save(path)

    # Convert to WAV if needed
    ext = os.path.splitext(path)[1].lower()
    if ext != ".wav":
        path = convert_audio_to_wav(path)

    return path
