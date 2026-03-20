import os, sys
from pathlib import Path

# Ajustar CWD
ROOT = Path(r"d:\voice-cloner")
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import gradio as gr

# Mock Progress
class MockProgress:
    def __call__(self, pct, desc=None):
        pass

try:
    from gui import full_pipeline
    
    audio_path = str(ROOT / "or.wav")
    model_name = "LUIS_HERNANDEZ"
    num_audios = 60
    language   = "ES"
    speed      = 1.0
    epochs     = 1000
    batch      = 8
    
    print(f"--- Iniciando Pipeline: {model_name} ---")
    print(f"Audio: {audio_path}")
    print(f"Audios a generar: {num_audios}")
    print(f"Epocas: {epochs}")
    print("------------------------------------------\n")
    
    last_msg = ""
    for msg in full_pipeline(audio_path, model_name, num_audios, language, 
                             speed, epochs, batch, progress=MockProgress()):
        # Solo imprimir si el mensaje cambió sustancialmente (para no saturar consola)
        new_lines = msg.replace(last_msg, "").strip()
        if new_lines:
            print(new_lines)
            last_msg = msg

    print("\n--- Pipeline Finalizado con Éxito ---")

except Exception as e:
    print(f"\n❌ ERROR FATAL en el Pipeline: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
