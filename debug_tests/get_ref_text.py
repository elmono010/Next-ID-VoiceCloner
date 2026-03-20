
import os
import torch
import whisper
import soundfile as sf

def get_ref_transcription():
    model = whisper.load_model("base", device="cpu")
    audio_path = "or.wav"
    
    # Leer los primeros 12 segundos
    data, sr = sf.read(audio_path)
    data_12s = data[:int(sr * 12)]
    
    temp_path = "tmp_ref_12s.wav"
    sf.write(temp_path, data_12s, sr)
    
    print(f"Transcribing {temp_path}...")
    result = model.transcribe(temp_path)
    print("EXACT REFERENCE TEXT:")
    print(result["text"])
    print(f"Detected language: {result['language']}")

if __name__ == "__main__":
    get_ref_transcription()
