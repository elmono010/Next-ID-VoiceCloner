
import os
import torch
import soundfile as sf
from f5_tts.api import F5TTS

def test_base_multilingual_es():
    device = "cpu"
    print(f"Testing Base Multilingual F5-TTS on {device} with EXACT REF_TEXT")
    
    # Usando el modelo BASE (multilingue)
    f5tts = F5TTS(
        model="F5TTS_Base",
        device=device
    )
    
    ref_audio = "or.wav"
    ref_text = "Oigan, trabajo sea madre porque les genera un karma y que la madre hay muchas charlatanes pejos que no saben ni lo que hablan. La vence los 5 antes de hablar. Primero que nada, un trabajo."
    
    gen_text = "Hola, esta es una prueba con el modelo base multilingue. Vamos a ver si con el texto de referencia correcto puede hablar español."
    
    print("Inference...")
    # Aseguramos que no hay parámetros de idioma raros
    audio, sr, _ = f5tts.infer(ref_audio, ref_text, gen_text)
    
    output = "output/debug_base_exact.wav"
    os.makedirs("output", exist_ok=True)
    sf.write(output, audio, sr)
    print(f"Success! Saved to {output}")

if __name__ == "__main__":
    test_base_multilingual_es()
