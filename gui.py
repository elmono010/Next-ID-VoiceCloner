#!/usr/bin/env python3
"""
gui.py — NEXT-ID VOICE CLONER: interfaz completa y autónoma
El usuario solo pone el audio. Todo lo demás es automático.
"""

import os
import sys

# Ajuste crucial para evitar errores en subprocesos de Python 3.10+
os.environ["PYTHONHASHSEED"] = "0"

def apply_audio_patch():
    try:
        import torchaudio
        import soundfile as sf
        import torch

        def patched_load(filepath, **kwargs):
            data, sr = sf.read(filepath)
            if len(data.shape) == 1:
                data = data[None, :]
            else:
                data = data.T
            return torch.from_numpy(data).float(), sr

        torchaudio.load = patched_load
        torchaudio.set_audio_backend("soundfile")
    except Exception:
        pass

apply_audio_patch()

import time, shutil, subprocess, queue, threading
from pathlib import Path

import numpy as np
import scipy.signal as signal
import torch
import gradio as gr

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
DATASET_DIR = ROOT / "output" / "dataset"
MODELS_DIR  = ROOT / "output" / "models"
LOGS_DIR    = ROOT / "output" / "logs"
APPLIO_DIR  = ROOT / "Applio"
ENV_PYTHON  = ROOT / "env" / "python.exe"

for d in [DATASET_DIR, MODELS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Estado global ────────────────────────────────────────────────────────────
state = {
    "training_active": False,
    "training_proc":   None,
    "rt_active":       False,
    "rt_stream":       None,
}

# ─── Helpers ─────────────────────────────────────────────────────────────────

def python_exe():
    return str(ENV_PYTHON) if ENV_PYTHON.exists() else sys.executable

def safe_env():
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "0"
    env["PYTHONIOENCODING"] = "utf-8"
    return env

def get_models():
    files = sorted([f.name for f in MODELS_DIR.glob("*.pth")])
    return files if files else ["(ningún modelo — crea uno primero)"]

def get_indexes():
    files = sorted([f.name for f in MODELS_DIR.glob("*.index")])
    return files if files else ["(ningún índice)"]

def get_audio_devices():
    try:
        import sounddevice as sd
        devs = sd.query_devices()
        inp  = [f"{i}: {d['name']}" for i, d in enumerate(devs) if d['max_input_channels']  > 0]
        out  = [f"{i}: {d['name']}" for i, d in enumerate(devs) if d['max_output_channels'] > 0]
        return inp or ["(sin dispositivos)"], out or ["(sin dispositivos)"]
    except Exception:
        return ["(instala sounddevice: pip install sounddevice)"], \
               ["(instala sounddevice: pip install sounddevice)"]

# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE COMPLETO
# ══════════════════════════════════════════════════════════════════════════════

def full_pipeline(audio_path, model_name, num_audios,
                  speed, mode, total_epochs, batch_size, progress=gr.Progress()):

    if audio_path is None:
        raise gr.Error("Sube tu archivo de audio primero.")
    if not model_name or not model_name.strip():
        raise gr.Error("Escribe un nombre para el modelo.")

    model_name = model_name.strip().replace(" ", "_")
    log = []

    def L(msg, pct=None):
        log.append(msg)
        if pct is not None:
            progress(pct)
        return "\n".join(log)

    # ── FASE 1: Validar audio ────────────────────────────────────────────────
    yield L("━━━ FASE 1/4  Validando audio ━━━", 0.0)
    try:
        import soundfile as sf
        data, sr = sf.read(audio_path)
        if len(data.shape) > 1:
            data = data.mean(axis=1)
        dur    = len(data) / sr
        rmsdb  = 20 * np.log10(np.sqrt(np.mean(data**2)) + 1e-10)
        if dur < 10:
            raise gr.Error(f"Audio muy corto: {dur:.1f}s  (mínimo 10s, ideal 30s–3min)")
        yield L(f"  ✅ Duración {dur:.1f}s  |  {sr}Hz  |  {rmsdb:.1f} dBFS", 0.05)
    except gr.Error:
        raise
    except Exception as e:
        raise gr.Error(f"Error leyendo el audio: {e}")

    # ── FASE 2: Generar dataset con F5-Spanish ─────────────────────────────
    yield L(f"\n━━━ FASE 2/4  Generando dataset sintético (F5-Spanish)  ({num_audios} audios) ━━━", 0.08)
    total_gen = 0
    try:
        yield L(f"  ━━━ MOTOR DE SÍNTESIS ━━━")
        yield L(f"  Motor: F5-Spanish (Juan Gallego)")
        
        yield L(f"  Cargando motor... (Zero-shot GPU)", 0.10)
        from tts.f5_cloner import F5Cloner
        cloner = F5Cloner()
        cloner.set_reference(audio_path)
        yield L(f"  ✅ Motor listo en {cloner.device.upper()}.")

        yield L(f"  Seleccionando {num_audios} textos fonéticos...")
        from texts.text_selector import select_texts
        texts = select_texts(language="ES", count=int(num_audios), mode=mode)
        yield L(f"  ✅ {len(texts)} textos seleccionados.")

        for f in DATASET_DIR.glob("*.wav"):
            f.unlink()

        yield L(f"\n  Sintetizando {len(texts)} audios...")
        for i, text in enumerate(texts):
            progress(0.12 + 0.33 * (i / len(texts)))
            out = str(DATASET_DIR / f"audio_{i+1:03d}.wav")
            try:
                cloner.synthesize(text=text, output_path=out, speed=speed)
                total_gen += 1
                yield L(f"  [{i+1:03d}/{len(texts)}] ✅ {text[:60]}")
            except Exception as e:
                yield L(f"  [{i+1:03d}/{len(texts)}] ⚠  {e}")

        yield L(f"\n  ✅ Dataset listo: {total_gen} audios en output/dataset/", 0.45)

    except ImportError as e:
        raise gr.Error(f"F5-TTS no instalado. Ejecuta INSTALAR_F5.bat\n{e}")
    except gr.Error:
        raise
    except Exception as e:
        raise gr.Error(f"Error generando dataset: {e}")

    # ── FASE 3: Entrenar con Applio ──────────────────────────────────────────
    yield L(f"\n━━━ FASE 3/4  Entrenando RVC  '{model_name}'  ({int(total_epochs)} épocas) ━━━", 0.47)

    trained_ok = False
    try:
        if not APPLIO_DIR.exists():
            yield L("  ⚠  Applio no encontrado. Instala con INSTALAR.bat")
            yield L("  → Puedes entrenar manualmente después.")
        else:
            # Copiar dataset a estructura Applio
            applio_ds = APPLIO_DIR / "assets" / "datasets" / model_name
            applio_ds.mkdir(parents=True, exist_ok=True)
            for wav in DATASET_DIR.glob("*.wav"):
                shutil.copy2(wav, applio_ds / wav.name)
            yield L(f"  ✅ Dataset copiado a Applio ({total_gen} archivos).")

            core = APPLIO_DIR / "core.py"
            if not core.exists():
                yield L("  ⚠  core.py no encontrado en Applio. Verifica la instalación.")
            else:
                # Preprocesar
                yield L("  Preprocesando dataset...")
                r = subprocess.run(
                    [python_exe(), str(core), "preprocess",
                     "--model_name", model_name,
                     "--dataset_path", str(applio_ds),
                     "--sample_rate", "40000",
                     "--cpu_cores", "4",
                     "--cut_preprocess", "Automatic"],
                    capture_output=True, text=True, cwd=str(APPLIO_DIR), timeout=600,
                    env=safe_env(), encoding="utf-8", errors="replace"
                )
                yield L("  ✅ Preprocesado OK." if r.returncode == 0
                        else f"  ⚠  Preprocess: {r.stderr[-300:]}")

                # Extraer features
                yield L("  Extrayendo features (RMVPE + ContentVec)...", 0.55)
                r = subprocess.run(
                    [python_exe(), str(core), "extract",
                     "--model_name", model_name,
                     "--f0_method", "rmvpe",
                     "--cpu_cores", "4",
                     "--gpu", "0",
                     "--sample_rate", "40000",
                     "--embedder_model", "contentvec",
                     "--include_mutes", "2"],
                    capture_output=True, text=True, cwd=str(APPLIO_DIR), timeout=900,
                    env=safe_env(), encoding="utf-8", errors="replace"
                )
                yield L("  ✅ Features extraídas." if r.returncode == 0
                        else f"  ⚠  Extract: {r.stderr[-300:]}")

                # Entrenar
                yield L(f"\n  Entrenando... ({int(total_epochs)} épocas, batch {int(batch_size)})")
                yield L("  El entrenamiento puede tardar 1–3 horas según la GPU.\n")

                proc = subprocess.Popen(
                    [python_exe(), str(core), "train",
                     "--model_name",            model_name,
                     "--save_every_epoch",      "10",
                     "--save_only_latest",      "False",
                     "--save_every_weights",    "True",
                     "--total_epoch",           str(int(total_epochs)),
                     "--sample_rate",           "40000",
                     "--batch_size",            str(int(batch_size)),
                     "--gpu",                   "0",
                     "--pretrained",            "True",
                     "--overtraining_detector", "True",
                     "--overtraining_threshold","50",
                     "--cleanup",               "False"],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, cwd=str(APPLIO_DIR), bufsize=1,
                    env=safe_env(), encoding="utf-8", errors="replace"
                )
                state["training_active"] = True
                state["training_proc"]   = proc

                buf = []
                for line in proc.stdout:
                    line = line.rstrip()
                    if any(k in line for k in ["Epoch","epoch","loss","Loss","Saving","saved"]):
                        buf.append(f"  {line}")
                        if len(buf) % 3 == 0:
                            yield L("\n".join(buf[-6:]))
                    elif "rror" in line:
                        yield L(f"  ⚠  {line}")

                proc.wait()
                state["training_active"] = False
                trained_ok = proc.returncode == 0
                yield L("\n  ✅ Entrenamiento finalizado." if trained_ok
                        else f"\n  ⚠  Entrenamiento terminó con código {proc.returncode}", 0.92)

    except subprocess.TimeoutExpired:
        yield L("  ⚠  Timeout en preprocesado. Intenta manualmente con Applio.")
    except Exception as e:
        yield L(f"  ⚠  Error: {e}")

    # ── FASE 4: Copiar modelo a output/models/ ──────────────────────────────
    yield L("\n━━━ FASE 4/4  Guardando modelo en output/models/ ━━━", 0.94)

    copied = _copy_best_model(model_name)
    if copied:
        yield L(f"  ✅ Modelo guardado: {copied}")
        yield L("  ✅ Índice (.index) copiado si existía.")
    else:
        yield L("  ⚠  No se encontró el modelo automáticamente.")
        yield L(f"  → Busca en Applio/logs/{model_name}/ y copia el .pth y .index a output/models/")

    progress(1.0)
    yield L(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅  PIPELINE COMPLETADO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Modelo     : {model_name}
  Dataset    : {total_gen} audios generados
  Guardado en: output/models/

  ▶  Ve a la pestaña  🔴 Tiempo Real
  ▶  Selecciona el modelo y tu micrófono
  ▶  Pulsa INICIAR — hablas tú, suena el modelo
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")


def _copy_best_model(model_name):
    search = [
        APPLIO_DIR / "logs"   / model_name / "weights",
        APPLIO_DIR / "logs"   / model_name,
        APPLIO_DIR / "assets" / "weights",
        APPLIO_DIR / "weights",
    ]
    best, best_t = None, 0
    for d in search:
        if not d.exists(): continue
        # Buscar .pth (excepto G_ o D_ que son pretraineds)
        for pth in d.glob("*.pth"):
            if "G_" in pth.name or "D_" in pth.name: continue
            t = pth.stat().st_mtime
            if t > best_t:
                best, best_t = pth, t
    
    if best:
        shutil.copy2(best, MODELS_DIR / f"{model_name}.pth")
        # Buscar el .index correspondiente
        idx_search = [
            APPLIO_DIR / "logs" / model_name,
            APPLIO_DIR / "assets" / "indices",
            APPLIO_DIR / "indices",
        ]
        for idx_dir in idx_search:
            if not idx_dir.exists(): continue
            for idx in idx_dir.glob("*.index"):
                shutil.copy2(idx, MODELS_DIR / f"{model_name}.index")
                break
        return best.name
    return None


def stop_training():
    p = state.get("training_proc")
    if p and p.poll() is None:
        p.terminate()
        state["training_active"] = False
        return "⏹  Entrenamiento detenido manualmente."
    return "No hay entrenamiento activo en este momento."


# ══════════════════════════════════════════════════════════════════════════════
# TIEMPO REAL
# ══════════════════════════════════════════════════════════════════════════════

def start_realtime(model_name, index_name, input_dev, output_dev, index_ratio, pitch_shift, noise_gate, gain, monitor):
    if not model_name or "ningún modelo" in model_name:
        return ("⚠  Selecciona un modelo primero.",
                gr.update(interactive=True), gr.update(interactive=False))

    # Resolvemos rutas absolutas
    model_path = MODELS_DIR / model_name
    index_path = str(MODELS_DIR / index_name) if index_name and "ningún" not in index_name else None

    try:
        import sounddevice as sd
        in_idx  = int(input_dev.split(":")[0])  if input_dev  and ":" in input_dev  else None
        out_idx = int(output_dev.split(":")[0]) if output_dev and ":" in output_dev else None

        # --- Iniciar Motor Nativo de Applio ---
        old_cwd = os.getcwd()
        try:
            os.chdir(str(APPLIO_DIR))
            sys.path.insert(0, str(APPLIO_DIR))
            from rvc.realtime.core import VoiceChanger
            
            # Iniciamos Motor con Post-Procesamiento Profesional (Pedalboard)
            motor = VoiceChanger(
                read_chunk_size=32,
                cross_fade_overlap_size=0.06,
                extra_convert_size=0.15,
                model_path=str(model_path),
                index_path=str(index_path) if index_path and os.path.exists(index_path) else None,
                f0_method="rmvpe",
                embedder_model="contentvec",
                # silent_threshold: el motor convierte dB a lineal internamente (línea 61 core.py)
                # Con -45 dB el motor ya filtra silencio real sin cortar sílabas débiles
                silent_threshold=float(noise_gate),
                # VAD interno: sensibilidad 1 para oficina con ruido de fondo
                # (3=máximo agresivo corta sílabas, 1=solo silencio real)
                vad_enabled=True,
                vad_sensitivity=1,
                vad_frame_ms=30,
                # TorchGate interno corre en GPU a 16kHz dentro del pipeline — no usar NR externo
                clean_audio=True,
                clean_strength=0.5,
                post_process=True,
                limiter=True,
                limiter_threshold=-3,
                compressor=True,
                compressor_threshold=-15,
                compressor_ratio=4,
            )
            print(f"--- RVC STUDIO ENGINE READY ---")
            
            # Ventana de suavizado (Fade 5ms)
            fade_len = int(48000 * 0.005)
            fade_in = np.sin(np.linspace(0, np.pi/2, fade_len))**2
            fade_out = fade_in[::-1]
            
        finally:
            os.chdir(old_cwd)

        # ── Gate externo: cubre los gaps que produce el VAD interno de Applio ──
        # El VAD interno (core.py línea 249) devuelve zeros cuando no hay voz.
        # Sin hold, esos zeros llegan al speaker como "blop". El hold los absorbe.
        _SR          = 48000
        _BLOCK       = 32 * 128                          # 4096 muestras
        _gate_linear = 10 ** (float(noise_gate) / 20.0)
        _hold_blocks = int(_SR * 0.30 / _BLOCK)         # 300 ms — cubre gaps de VAD
        _gs          = {"open": False, "hold": 0}

        def callback(indata, outdata, frames, time_info, status):
            try:
                audio_input = indata[:, 0].copy() * float(gain)

                # Gate de entrada: energía mínima para no amplificar silencio absoluto
                rms = float(np.sqrt(np.mean(audio_input ** 2)))
                if rms >= _gate_linear:
                    _gs["open"] = True
                    _gs["hold"] = _hold_blocks
                elif _gs["hold"] > 0:
                    _gs["hold"] -= 1
                else:
                    _gs["open"] = False

                if not _gs["open"]:
                    outdata[:] = 0
                    return

                # Normalización pre-RVC
                peak_in = np.max(np.abs(audio_input))
                if peak_in > 0.001:
                    audio_input = (audio_input / peak_in) * 0.7

                # El VAD interno (vad_sensitivity=1) + TorchGate hacen el resto
                audio_out, vol, _ = motor.on_request(
                    audio_input,
                    f0_up_key=int(pitch_shift),
                    index_rate=float(index_ratio),
                    protect=0.33,
                    volume_envelope=1,
                    f0_autotune=False,
                    f0_autotune_strength=1,
                    proposed_pitch=False,
                    proposed_pitch_threshold=155.0
                )

                # Normalización de salida
                peak_out = np.max(np.abs(audio_out))
                if peak_out > 0.001:
                    audio_out = (audio_out / peak_out) * 0.8

                if monitor:
                    outdata[:, 0] = audio_out[:frames]
                else:
                    outdata[:, 0] = 0

            except Exception as e:
                outdata[:] = 0

        # Bloque de 32*128 = 4096 (SINCRO TOTAL CON EL MOTOR)
        stream = sd.Stream(
            samplerate=48000, 
            blocksize=32*128, 
            dtype="float32",
            device=(in_idx, out_idx), 
            channels=(1, 1),
            callback=callback, 
            latency="low"
        )
        stream.start()
        state["rt_active"] = True
        state["rt_stream"] = stream

        return (
            f"🔴 ACTIVO (Nativo)\nMotor: VoiceChanger SOLA\nSR: 48000Hz\nLatency block: {16*128}",
            gr.update(interactive=False),
            gr.update(interactive=True)
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return (f"❌ Error fatal: {e}", gr.update(interactive=True), gr.update(interactive=False))


def stop_realtime():
    s = state.get("rt_stream")
    if s:
        try: s.stop(); s.close()
        except Exception: pass
    state["rt_active"] = False
    state["rt_stream"] = None
    return ("⚫ Detenido.",
            gr.update(interactive=True), gr.update(interactive=False))


def get_rt_status():
    if state["rt_active"]:
        s = state.get("rt_stream")
        if s:
            return f"🔴 ACTIVO  ·  CPU carga: {s.cpu_load*100:.1f}%"
    return "⚫ Inactivo"


# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════

CSS = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&family=Syne:wght@700;800&display=swap');
:root{
  --bg0:#07070d;--bg1:#0e0e19;--bg2:#141420;--bg3:#1b1b2c;
  --ac:#7c6af5;--ac2:#4a3fa0;--acg:rgba(124,106,245,.15);
  --tx:#e4e2f2;--tx2:#857fa8;--tx3:#3f3c60;
  --gr:#3dd68c;--wa:#f5a623;--er:#f05c5c;
  --bd:#22203a;--bda:#38356a;
}
*{box-sizing:border-box}
body,.gradio-container{background:var(--bg0)!important;font-family:'JetBrains Mono',monospace!important;color:var(--tx)!important}
.gradio-container .block{background:var(--bg2)!important;border:1px solid var(--bd)!important;border-radius:12px!important}
.gradio-container .block:hover{border-color:var(--bda)!important;transition:border-color .2s}

.hdr{background:linear-gradient(140deg,#0b0918 0%,#14113a 50%,#07070d 100%);
     border-bottom:1px solid var(--bda);padding:26px 34px 20px;position:relative;overflow:hidden}
.hdr::before{content:'';position:absolute;width:260px;height:260px;border-radius:50%;
     top:-100px;right:-60px;background:radial-gradient(circle,rgba(124,106,245,.12),transparent 70%);pointer-events:none}
.htitle{font-family:'Syne',sans-serif;font-size:2rem;font-weight:800;letter-spacing:-.025em;margin:0 0 4px}
.htitle em{color:var(--ac);font-style:normal}
.hsub{font-size:.7rem;color:var(--tx2);letter-spacing:.15em;text-transform:uppercase}
.badge{background:#18163a;color:var(--ac);font-size:.62rem;padding:2px 9px;
       border-radius:20px;margin-left:10px;border:1px solid var(--bda);vertical-align:middle}

.tabs .tab-nav{background:var(--bg1)!important;border-bottom:1px solid var(--bd)!important;padding:0 18px!important}
.tabs .tab-nav button{font-family:'Syne',sans-serif!important;font-size:.76rem!important;
  font-weight:700!important;color:var(--tx2)!important;background:transparent!important;
  border:none!important;border-bottom:2px solid transparent!important;
  padding:12px 16px!important;text-transform:uppercase!important;letter-spacing:.1em!important;transition:all .15s!important}
.tabs .tab-nav button:hover{color:var(--tx)!important}
.tabs .tab-nav button.selected{color:var(--ac)!important;border-bottom-color:var(--ac)!important}

label span{font-size:.73rem!important;color:var(--tx2)!important;letter-spacing:.04em!important}
textarea,input[type=text],input[type=number],select{
  background:var(--bg3)!important;border:1px solid var(--bd)!important;
  color:var(--tx)!important;font-family:'JetBrains Mono',monospace!important;
  font-size:.78rem!important;border-radius:8px!important}
textarea:focus,input:focus{border-color:var(--ac)!important}

.logbox textarea{background:#040408!important;border-color:var(--bda)!important;
  color:#9ef0c0!important;font-size:.72rem!important;line-height:1.8!important}

.gradio-container button.primary{
  background:var(--ac)!important;color:#fff!important;
  font-family:'Syne',sans-serif!important;font-weight:700!important;
  font-size:.8rem!important;letter-spacing:.09em!important;text-transform:uppercase!important;
  border:none!important;border-radius:8px!important;
  box-shadow:0 0 20px rgba(124,106,245,.3)!important;transition:all .18s!important}
.gradio-container button.primary:hover{background:#9080ff!important;
  box-shadow:0 0 32px rgba(124,106,245,.55)!important;transform:translateY(-1px)!important}
.gradio-container button.primary:disabled{background:var(--bg3)!important;
  color:var(--tx3)!important;box-shadow:none!important;transform:none!important}
.gradio-container button.secondary{background:var(--bg3)!important;color:var(--tx2)!important;
  border:1px solid var(--bd)!important;font-family:'Syne',sans-serif!important;font-size:.76rem!important;border-radius:8px!important}

.sl{font-family:'Syne',sans-serif;font-size:.66rem;font-weight:700;
    color:var(--tx2);text-transform:uppercase;letter-spacing:.15em;margin-bottom:8px;display:block}

.psteps{display:flex;gap:0;margin:14px 0 18px}
.pstep{flex:1;text-align:center;position:relative}
.pstep::after{content:'';position:absolute;top:14px;left:50%;width:100%;height:1px;background:var(--bd)}
.pstep:last-child::after{display:none}
.pnum{width:28px;height:28px;border-radius:50%;background:var(--bg3);border:1px solid var(--bd);
      color:var(--tx3);font-size:.7rem;font-family:'Syne',sans-serif;font-weight:700;
      display:inline-flex;align-items:center;justify-content:center;position:relative;z-index:1;margin-bottom:5px}
.pnum.a{background:var(--ac2);border-color:var(--ac);color:#fff;box-shadow:0 0 10px rgba(124,106,245,.5)}
.ptxt{font-size:.62rem;color:var(--tx3);line-height:1.4}

::-webkit-scrollbar{width:4px;height:4px}
::-webkit-scrollbar-track{background:var(--bg0)}
::-webkit-scrollbar-thumb{background:var(--bda);border-radius:2px}
"""


# ══════════════════════════════════════════════════════════════════════════════
# BUILD
# ══════════════════════════════════════════════════════════════════════════════

def build():
    inp_devs, out_devs = get_audio_devices()

    with gr.Blocks(title="NEXT-ID VOICE CLONER") as app:

        gr.HTML("""
        <div class="hdr">
          <div class="htitle">NEXT-ID <em>VOICE CLONER</em><span class="badge">100% LOCAL</span></div>
          <div class="hsub">F5-TTS / E2-TTS  ›  RVC Applio  ›  Tiempo Real</div>
        </div>""")

        with gr.Tabs(elem_classes="tabs"):

            # ────────────────────────────────────────────────────────────
            # TAB 1 — CREAR MODELO (F5-TTS / E2-TTS)
            # ────────────────────────────────────────────────────────────
            with gr.Tab("🎙  Crear Modelo"):
                gr.HTML("""
                <div style="padding:14px 0 2px">
                  <span class="sl">Pipeline automático 100% Local — F5/E2-TTS</span>
                  <div class="psteps">
                    <div class="pstep">
                      <div class="pnum a">1</div>
                      <div class="ptxt">Subes<br>tu voz</div>
                    </div>
                    <div class="pstep">
                      <div class="pnum">2</div>
                      <div class="ptxt">F5-Spanish<br>genera dataset</div>
                    </div>
                    <div class="pstep">
                      <div class="pnum">3</div>
                      <div class="ptxt">Applio<br>entrena RVC</div>
                    </div>
                    <div class="pstep">
                      <div class="pnum">4</div>
                      <div class="ptxt">Modelo listo<br>output/models/</div>
                    </div>
                  </div>
                </div>""")

                with gr.Row(equal_height=False):
                    with gr.Column(scale=1, min_width=320):

                        gr.HTML('<span class="sl">Audio de referencia</span>')
                        audio_in = gr.Audio(
                            label="Sube tu voz (WAV · 30s mínimo · ideal 1–3 min)",
                            type="filepath"
                        )
                        model_name_in = gr.Textbox(
                            label="Nombre del modelo",
                            placeholder="mi_voz  /  locutor_juan  /  narrador ...",
                            max_lines=1
                        )

                        gr.HTML('<span class="sl" style="margin-top:14px">Configuración Dataset (Español Latino)</span>')
                        num_audios_in = gr.Radio(
                            choices=[1, 30, 90, 180], value=30,
                            label="Audios a generar",
                            info="1=Test rápido  |  90 audios ≈ 30min de dataset"
                        )
                        with gr.Row():
                            speed_in = gr.Slider(
                                0.7, 1.3, value=1.0, step=0.05,
                                label="Velocidad"
                            )
                        
                        gr.HTML(
                            '<div style="background:rgba(124,106,245,.08);border:1px solid #38356a;'
                            'border-radius:8px;padding:10px 14px;margin:4px 0 8px;font-size:.7rem;'
                            'color:#857fa8;line-height:1.7">'
                            '<b style="color:#7c6af5">Control de Acento:</b> '
                            'El acento latino (colombiano, mexicano, etc.) lo define tu audio '
                            'de referencia. El motor clona el timbre y la cadencia exacta.'
                            '</div>'
                        )

                        mode_in = gr.Radio(
                            choices=["auto", "guiones", "frases", "mixto"],
                            value="auto",
                            label="Tipo de textos del dataset",
                            info="guiones=narrativo ~1min  |  frases=cortas variadas  |  mixto=ambos  |  auto=segun cantidad"
                        )

                        gr.HTML('<span class="sl" style="margin-top:14px">Entrenamiento (GPU / CPU)</span>')
                        with gr.Row():
                            epochs_in = gr.Number(
                                value=500, label="Épocas",
                                precision=0, info="500 para 30-90 audios"
                            )
                            batch_in  = gr.Number(
                                value=8, label="Batch size",
                                precision=0, info="8 para 8GB VRAM"
                            )

                        with gr.Row():
                            run_btn  = gr.Button(
                                "🚀  INICIAR PIPELINE COMPLETO",
                                variant="primary", scale=3
                            )
                            stop_btn = gr.Button(
                                "⏹", variant="secondary", scale=1,
                                elem_id="stop-btn"
                            )

                    with gr.Column(scale=2):
                        gr.HTML('<span class="sl">Log en tiempo real</span>')
                        log_out = gr.Textbox(
                            label="", lines=32, max_lines=50,
                            interactive=False,
                            placeholder="Aquí verás el progreso paso a paso cuando inicies el pipeline...",
                            elem_classes="logbox"
                        )

                run_btn.click(
                    full_pipeline,
                    inputs=[audio_in, model_name_in, num_audios_in,
                            speed_in, mode_in, epochs_in, batch_in],
                    outputs=[log_out]
                )
                stop_btn.click(stop_training, outputs=[log_out])

            # ────────────────────────────────────────────────────────────
            # TAB 2 — TIEMPO REAL
            # ────────────────────────────────────────────────────────────
            with gr.Tab("🔴  Tiempo Real"):
                with gr.Row(equal_height=False):

                    # ── Controles ─────────────────────────────────────────
                    with gr.Column(scale=1, min_width=320):

                        gr.HTML('<span class="sl">Modelo de voz</span>')
                        model_sel = gr.Dropdown(
                            choices=get_models(), value=get_models()[0],
                            label="Modelo .pth activo"
                        )
                        index_sel = gr.Dropdown(
                            choices=get_indexes(), value=get_indexes()[0],
                            label="Archivo .index (mejora la fidelidad del timbre)"
                        )
                        refresh_btn = gr.Button(
                            "🔄  Actualizar lista de modelos",
                            variant="secondary", size="sm"
                        )

                        gr.HTML('<span class="sl" style="margin-top:18px">Dispositivos de audio</span>')
                        in_dev_sel = gr.Dropdown(
                            choices=inp_devs,
                            value=inp_devs[0],
                            label="Micrófono de entrada (tu voz)"
                        )
                        out_dev_sel = gr.Dropdown(
                            choices=out_devs,
                            value=out_devs[0],
                            label="Salida (altavoces / VB-Audio Cable)"
                        )

                        gr.HTML('<span class="sl" style="margin-top:18px">Ajustes RVC</span>')
                        with gr.Row():
                            pitch_shift = gr.Slider(
                                -24, 24, value=0, step=1,
                                label="Tono (Pitch)",
                                info="-12 = mujer a hombre · +12 = hombre a mujer"
                            )
                            noise_gate = gr.Slider(
                                -100, 0, value=-50, step=5,
                                label="Puerta de ruido (dB)",
                                info="Evita ruidos de fondo cuando no hablas"
                            )
                        with gr.Row():
                            idx_ratio = gr.Slider(
                                0.0, 1.0, value=0.75, step=0.05,
                                label="Index ratio",
                                info="0 = más natural  ·  1 = más fiel al modelo clonado"
                            )
                            gain_in = gr.Slider(
                                0.5, 5.0, value=1.0, step=0.1,
                                label="Ganancia (Boost)",
                                info="Sube el volumen si el modelo no te reconoce"
                            )
                        monitor_chk = gr.Checkbox(
                            value=True,
                            label="🔊 Monitorear — escuchar la voz de salida en tiempo real"
                        )

                        with gr.Row():
                            start_btn = gr.Button("▶  INICIAR", variant="primary", scale=2)
                            stop_rt   = gr.Button("⏹  DETENER", variant="secondary",
                                                  scale=1, interactive=False)

                        gr.HTML('<span class="sl" style="margin-top:14px">Estado</span>')
                        rt_stat = gr.Textbox(
                            value="⚫ Inactivo", label="",
                            interactive=False, lines=3, max_lines=4
                        )
                        timer = gr.Timer(value=1.5)
                        timer.tick(get_rt_status, outputs=[rt_stat])

                    # ── Info panel ────────────────────────────────────────
                    with gr.Column(scale=1):
                        gr.HTML("""
                        <span class="sl">Flujo de audio</span>
                        <div style="background:#040408;border:1px solid #22203a;border-radius:12px;
                             padding:22px;font-size:.78rem;line-height:2.6">

                          <div style="display:flex;align-items:center;gap:14px">
                            <div style="font-size:1.5rem;width:40px;text-align:center">🎙️</div>
                            <div>
                              <div style="color:#e4e2f2;font-weight:500">Micrófono físico</div>
                              <div style="color:#3f3c60;font-size:.7rem">Tu voz entra aquí</div>
                            </div>
                          </div>
                          <div style="margin-left:20px;color:#2a2848;font-size:1.1rem;
                               padding:2px 0;line-height:1.2">↓</div>

                          <div style="display:flex;align-items:center;gap:14px;
                               background:rgba(124,106,245,.09);border:1px solid #38356a;
                               border-radius:10px;padding:12px 16px">
                            <div style="font-size:1.5rem;width:40px;text-align:center">🤖</div>
                            <div>
                              <div style="color:#7c6af5;font-weight:600">Voice Cloner RVC</div>
                              <div style="color:#3f3c60;font-size:.7rem">
                                RMVPE  ·  Chunk 512  ·  ~100–150ms
                              </div>
                            </div>
                          </div>
                          <div style="margin-left:20px;color:#2a2848;font-size:1.1rem;
                               padding:2px 0;line-height:1.2">↓</div>

                          <div style="display:flex;align-items:center;gap:14px">
                            <div style="font-size:1.5rem;width:40px;text-align:center">🔊</div>
                            <div>
                              <div style="color:#e4e2f2;font-weight:500">Salida de audio</div>
                              <div style="color:#3f3c60;font-size:.7rem">
                                Altavoces · VB-Cable · Discord · OBS
                              </div>
                            </div>
                          </div>

                          <div style="margin-top:18px;padding-top:14px;border-top:1px solid #14142a;
                               font-size:.68rem;color:#3f3c60;line-height:1.9">
                            <span style="color:#7c6af5">Para Discord / OBS:</span><br>
                            1. Instala <strong style="color:#857fa8">VB-Audio Virtual Cable</strong><br>
                            2. Selecciona <em>CABLE Input</em> como salida aquí<br>
                            3. En Discord/OBS → entrada → <em>CABLE Output</em>
                          </div>
                        </div>

                        <span class="sl" style="margin-top:18px">Modelos en output/models/</span>
                        """)

                        models_md = gr.Markdown(
                            value="\n".join([f"- `{m}`" for m in get_models()])
                        )

                # Eventos Tab 2
                def refresh_fn():
                    m = get_models(); i = get_indexes()
                    return (
                        gr.update(choices=m, value=m[0]),
                        gr.update(choices=i, value=i[0]),
                        "\n".join([f"- `{x}`" for x in m])
                    )

                refresh_btn.click(refresh_fn, outputs=[model_sel, index_sel, models_md])

                start_btn.click(
                    start_realtime,
                    inputs=[model_sel, index_sel, in_dev_sel, out_dev_sel, idx_ratio, pitch_shift, noise_gate, gain_in, monitor_chk],
                    outputs=[rt_stat, start_btn, stop_rt]
                )
                stop_rt.click(
                    stop_realtime,
                    outputs=[rt_stat, start_btn, stop_rt]
                )

        gr.HTML("""
        <div style="border-top:1px solid #1a1830;padding:12px 24px;
             display:flex;justify-content:space-between;align-items:center">
          <div style="font-size:.65rem;color:#3f3c60">
            Voice Cloner · 100% local · sin telemetría
          </div>
          <div style="display:flex;gap:18px">
            <a href="https://github.com/SWivid/F5-TTS" target="_blank"
               style="font-size:.65rem;color:#7c6af5;text-decoration:none">F5-TTS / E2-TTS</a>
            <a href="https://github.com/IAHispano/Applio" target="_blank"
               style="font-size:.65rem;color:#7c6af5;text-decoration:none">Applio RVC</a>
          </div>
        </div>""")

    return app


if __name__ == "__main__":
    build().launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
        show_error=True,
        css=CSS,
        theme=gr.themes.Base()
    )