from voice_handler import VoiceHandler
from hindi_nlp_processor import HindiNLPProcessor

# Test transcription
voice_handler = VoiceHandler(language_code='hindi')
nlp = HindiNLPProcessor()

# Test with a sample audio file
audio_file = "test_audio.wav"  # Record yourself saying "मेरी उम्र पच्चीस साल है"

transcribed = voice_handler.transcribe_audio_file(audio_file)
print(f"Transcribed: {transcribed}")

# Test NLP extraction
value = nlp.process_answer(transcribed, 'age')
print(f"Extracted value: {value}")