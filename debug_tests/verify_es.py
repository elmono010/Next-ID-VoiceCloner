
import os
import torch
import whisper

def verify_audio():
    model = whisper.load_model("base", device="cpu")
    audio_path = "output/test_f5_es_final/audio_001.wav"
    
    if not os.path.exists(audio_path):
        print(f"File {audio_path} not found")
        return

    print(f"Transcribing {audio_path}...")
    result = model.transcribe(audio_path)
    print("TRANSCRIPTION:")
    print(result["text"])
    print(f"Detected language: {result['language']}")

if __name__ == "__main__":
    verify_audio()
