
import os
import torch
from f5_tts.api import F5TTS

def test_transcribe():
    print("Testing F5TTS.transcribe...")
    f5tts = F5TTS(device="cpu")
    audio_path = "or.wav"
    if not os.path.exists(audio_path):
        print(f"File {audio_path} not found")
        return
        
    print(f"Transcribing {audio_path}...")
    try:
        text = f5tts.transcribe(audio_path)
        print(f"RESULT: '{text}'")
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    test_transcribe()
