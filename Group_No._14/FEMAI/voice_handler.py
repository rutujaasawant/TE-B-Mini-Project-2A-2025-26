"""
VoiceHandler for Google Speech-to-Text
Supports transcription of uploaded files and streaming audio
Handles WEBM/OPUS and WAV seamlessly
"""
import io
import logging
from google.cloud import speech_v1p1beta1 as speech
import pyaudio
import wave
from voice_config import AUDIO_CONFIG, LANGUAGE_CONFIGS

logger = logging.getLogger(__name__)

class VoiceHandler:
    def __init__(self, language='hindi'):
        self.client = speech.SpeechClient()
        self.language_code = LANGUAGE_CONFIGS.get(language, LANGUAGE_CONFIGS['hindi'])['code']

    def set_language(self, language: str):
        """Set language dynamically"""
        self.language_code = LANGUAGE_CONFIGS.get(language, LANGUAGE_CONFIGS['hindi'])['code']

    def transcribe_audio_file(self, audio_file_path: str) -> str:
        """
        Transcribe an uploaded audio file to text.
        Supports WEBM/OPUS and WAV automatically.
        """
        try:
            with io.open(audio_file_path, 'rb') as audio_file:
                content = audio_file.read()

            audio = speech.RecognitionAudio(content=content)

            # Let Google auto-detect encoding & sample rate for WEBM/OPUS
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.ENCODING_UNSPECIFIED,
                language_code=self.language_code,
                enable_automatic_punctuation=True,
                model='default',
                use_enhanced=True
            )

            response = self.client.recognize(config=config, audio=audio)
            transcripts = [result.alternatives[0].transcript for result in response.results]

            return ' '.join(transcripts) if transcripts else ""

        except Exception as e:
            logger.error(f"Error transcribing audio: {e}", exc_info=True)
            return None

    def transcribe_streaming(self, audio_stream):
        """
        Transcribe streaming audio in real-time (audio_stream yields raw bytes)
        """
        try:
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=AUDIO_CONFIG['sample_rate'],
                language_code=self.language_code,
                enable_automatic_punctuation=True
            )
            streaming_config = speech.StreamingRecognitionConfig(
                config=config, interim_results=True
            )

            requests = (speech.StreamingRecognizeRequest(audio_content=chunk) for chunk in audio_stream)
            responses = self.client.streaming_recognize(streaming_config, requests)

            final_transcript = ""
            for response in responses:
                for result in response.results:
                    if result.is_final:
                        final_transcript += result.alternatives[0].transcript + " "
            return final_transcript.strip()

        except Exception as e:
            logger.error(f"Error in streaming transcription: {e}", exc_info=True)
            return None


class AudioRecorder:
    """Handles recording audio from microphone"""
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.frames = []
        self.is_recording = False
        self.stream = None

    def start_recording(self):
        """Start recording audio from microphone"""
        self.frames = []
        self.is_recording = True
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=AUDIO_CONFIG['channels'],
            rate=AUDIO_CONFIG['sample_rate'],
            input=True,
            frames_per_buffer=AUDIO_CONFIG['chunk_size']
        )
        logger.info("Recording started...")
        while self.is_recording:
            data = self.stream.read(AUDIO_CONFIG['chunk_size'], exception_on_overflow=False)
            self.frames.append(data)
        self.stream.stop_stream()
        self.stream.close()
        logger.info("Recording stopped")

    def stop_recording(self):
        """Stop recording"""
        self.is_recording = False

    def save_recording(self, filepath):
        """Save recorded audio to WAV file"""
        wf = wave.open(filepath, 'wb')
        wf.setnchannels(AUDIO_CONFIG['channels'])
        wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(AUDIO_CONFIG['sample_rate'])
        wf.writeframes(b''.join(self.frames))
        wf.close()
        return filepath
