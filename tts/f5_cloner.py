"""
tts/f5_cloner.py
Motor F5-TTS optimizado PARA ESPAÑOL LATINO.
Finetune: jpgallegoar/F5-Spanish (Juan Gallego)
"""

import os
import torch
import soundfile as sf
import sys
import whisper
import torchaudio
from pathlib import Path
from rich.console import Console
from cached_path import cached_path

# Ajustes de entorno fundamentales
os.environ["PYTHONHASHSEED"] = "0"

def patched_torchaudio_load(filepath, **kwargs):
    """Parche agresivo para evitar el uso de torchcodec."""
    import soundfile as sf
    import torch
    data, sr = sf.read(filepath)
    if len(data.shape) == 1:
        data = data[None, :]
    else:
        data = data.T
    return torch.from_numpy(data).float(), sr

torchaudio.load = patched_torchaudio_load

try:
    torchaudio.set_audio_backend("soundfile")
except Exception:
    pass

# Importaciones diferidas para evitar errores si no está instalado
try:
    from f5_tts.api import F5TTS
    import f5_tts.infer.utils_infer as f5_utils_infer
    import f5_tts.model.utils as f5_model_utils
except ImportError:
    F5TTS = None
    f5_utils_infer = None
    f5_model_utils = None

console = Console()

class F5Cloner:
    """
    Motor exclusivo F5-Spanish para máxima fidelidad y acento latino.
    Forzado a trabajar en GPU (CUDA).
    """

    def __init__(self, device: str = "cuda"):
        self.language = "ES" # Forzado a Español
        self.model_name = "F5-Spanish (Juan Gallego)"
        self.device = "cuda" # FORZADO A GPU por petición del usuario
        self.f5tts = None
        self.ref_audio = None
        self.ref_text = None
        
        console.print(f"  [bold cyan]Motor Único:[/bold cyan] [white]{self.model_name}[/white]")
        console.print(f"  [bold cyan]Hardware:[/bold cyan] [yellow]{self.device.upper()} (FORZADO)[/yellow]")

        # Aplicar Monkey Patch para español inmediatamente
        if f5_model_utils:
            self._apply_spanish_patch()

    def _apply_spanish_patch(self):
        """Evita la conversión a Pinyin para español."""
        def patched_convert_char_to_pinyin(text_list, polyphone=True):
            return text_list
        f5_model_utils.convert_char_to_pinyin = patched_convert_char_to_pinyin
        console.print("  [dim]F5 Patch:[/dim] [cyan]Pinyin desactivado (Optimizado Español)[/cyan]")

    def _load_model(self):
        """Carga el modelo. Intenta GPU, si falla por incompatibilidad (Blackwell), usa CPU."""
        if self.f5tts is not None:
            return
            
        if F5TTS is None:
            raise ImportError("F5-TTS no está instalado. Ejecuta 'INSTALAR_F5.bat'.")

        try:
            self._load_model_internal(self.device)
        except Exception as e:
            err_msg = str(e).lower()
            if "no kernel image" in err_msg or "sm_120" in err_msg or "capability 12.0" in err_msg:
                console.print(f"  [yellow]⚠ TU GPU BLACKWELL NO ES COMPATIBLE AÚN CON ESTE TORCH.[/yellow]")
                console.print(f"  [yellow]Rebajando a CPU para que no se detenga el proceso...[/yellow]")
                self.device = "cpu"
                self._load_model_internal("cpu")
            else:
                console.print(f"  [red]CRÍTICO: Error en {self.device}:[/red] {e}")
                raise e

    def _load_model_internal(self, device):
        f5_utils_infer.device = device
        
        # Modelo especializado de Juan Gallego (jpgallegoar)
        repo_name = "jpgallegoar/F5-Spanish"
        ckpt_step = 1200000 
        model_type = "F5TTS_Base"
        
        console.print(f"  [dim]F5:[/dim] Cargando modelo [bold]jpgallegoar/F5-Spanish[/bold]...")
        
        ckpt_file = str(cached_path(f"hf://{repo_name}/model_{ckpt_step}.safetensors"))
        vocab_file = str(cached_path(f"hf://{repo_name}/vocab.txt"))

        self.f5tts = F5TTS(
            model=model_type,
            ckpt_file=ckpt_file,
            vocab_file=vocab_file,
            device=device
        )

    def set_reference(self, reference_audio: str):
        if not os.path.exists(reference_audio):
            raise FileNotFoundError(f"No se encontró: {reference_audio}")
        
        self.ref_audio = reference_audio
        self._load_model()
        
        console.print(f"  [dim]F5 Transcribiendo:[/dim] [cyan]{os.path.basename(reference_audio)}[/cyan]")
        
        try:
            # Recortar a 12s para alineación perfecta (Discovery 2026-03-19)
            from pydub import AudioSegment
            import tempfile
            
            aseg = AudioSegment.from_file(self.ref_audio)
            aseg_clipped = aseg[:12000] # Primeros 12 segundos
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                temp_clip_path = tmp.name
            
            aseg_clipped.export(temp_clip_path, format="wav")
            
            # Usamos whisper oficial en CPU para la transcripción inicial para evitar conflictos de memoria
            model_whisper = whisper.load_model("base", device="cpu")
            result = model_whisper.transcribe(temp_clip_path, verbose=False)
            self.ref_text = result["text"].strip()
            
            if os.path.exists(temp_clip_path):
                os.remove(temp_clip_path)
                
            console.print(f"  [dim]Texto (12s):[/dim] [italic]\"{self.ref_text[:60]}...\"[/italic]")
        except Exception as e:
            console.print(f"  [yellow]Error transcripción:[/yellow] {e}")
            self.ref_text = ""

    def extract_voice_profile(self, reference_audio: str):
        self.set_reference(reference_audio)

    def synthesize(self, text: str, output_path: str, speed: float = 1.0):
        if self.ref_audio is None:
            raise RuntimeError("Falta audio de referencia.")
        self._load_model()
        
        try:
            wav, sr, _ = self.f5tts.infer(
                ref_file=self.ref_audio,
                ref_text=self.ref_text, 
                gen_text=text
            )
            
            if abs(speed - 1.0) > 0.01:
                import librosa
                wav = librosa.effects.time_stretch(wav, rate=speed)

            sf.write(output_path, wav, sr)
        except Exception as e:
            console.print(f"  [red]Error síntesis GPU:[/red] {e}")
            raise e
