
import os
import sys
import torch
from tts.f5_cloner import F5Cloner

def test():
    # Probar F5-Spanish (el nuevo estándar)
    cloner = F5Cloner()
    
    # Intentar cargar referencia (esto gatillará _load_model)
    try:
        cloner.set_reference("or.wav")
    except Exception as e:
        print(f"Error en set_reference: {e}")
        return

    text = "Probando el motor E2 TTS. Este modelo debería entender mejor el español latino."
    output = "test_e2_es.wav"
    
    if os.path.exists(output):
        os.remove(output)
        
    print(f"Iniciando sintesis con {cloner.model_name} en {cloner.device}...")
    try:
        cloner.synthesize(text=text, output_path=output, speed=1.0)
        if os.path.exists(output):
            print(f"EXITO: Archivo generado en {output} ({os.path.getsize(output)} bytes)")
        else:
            print(f"FALLO: El archivo no fue generado.")
    except Exception as e:
        print(f"EXCEPCION FINAL: {e}")

if __name__ == "__main__":
    test()
