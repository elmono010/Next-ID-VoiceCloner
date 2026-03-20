
import os
import sys
from tts.f5_cloner import F5Cloner

def test():
    cloner = F5Cloner()
    # set_reference ahora carga el modelo y transcribe
    cloner.set_reference("or.wav")
    
    text = "Esta es una prueba de sintesis con F5 TTS en Español para verificar que no suene portugués."
    output = "test_f5_es.wav"
    
    if os.path.exists(output):
        os.remove(output)
        
    print(f"Iniciando sintesis...")
    try:
        cloner.synthesize(text=text, output_path=output, speed=1.0)
        if os.path.exists(output):
            print(f"EXITO: Archivo generado en {output} ({os.path.getsize(output)} bytes)")
        else:
            print(f"FALLO: El archivo no fue generado.")
    except Exception as e:
        print(f"EXCEPCION: {e}")

if __name__ == "__main__":
    test()
