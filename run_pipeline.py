#!/usr/bin/env python3
"""
NEXT-ID VOICE CLONER — Pipeline optimizado para F5-Spanish (Juan Gallego)
Uso: python run_pipeline.py --reference mi_voz.wav --num_audios 30
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
            import soundfile as sf
            import torch
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

import time, argparse
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich import print as rprint
from colorama import init

init()
console = Console()

# ─────────────────────────────────────────────
#  Banner
# ─────────────────────────────────────────────
BANNER = """
[bold cyan]██╗   ██╗ ██████╗ ██╗ ██████╗███████╗     ██████╗██╗      ██████╗ ███╗   ██╗███████╗██████╗[/bold cyan]
[bold cyan]██║   ██║██╔═══██╗██║██╔════╝██╔════╝    ██╔════╝██║     ██╔═══██╗████╗  ██║██╔════╝██╔══██╗[/bold cyan]
[bold cyan]██║   ██║██║   ██║██║██║     █████╗      ██║     ██║     ██║   ██║██╔██╗ ██║█████╗  ██████╔╝[/bold cyan]
[bold cyan]╚██╗ ██╔╝██║   ██║██║██║     ██╔══╝      ██║     ██║     ██║   ██║██║╚██╗██║██╔══╝  ██╔══██╗[/bold cyan]
[bold cyan] ╚████╔╝ ╚██████╔╝██║╚██████╗███████╗    ╚██████╗███████╗╚██████╔╝██║ ╚████║███████╗██║  ██║[/bold cyan]
[bold cyan]  ╚═══╝   ╚═════╝ ╚═╝ ╚═════╝╚══════╝     ╚═════╝╚══════╝ ╚═════╝ ╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝[/bold cyan]
[dim]Pipeline 100% local: F5-TTS (Juan Gallego) → Dataset → Applio RVC[/dim]
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="NEXT-ID VOICE CLONER: Pipeline TTS → RVC optimizado para ESPAÑOL LATINO"
    )
    parser.add_argument(
        "--reference", "-r",
        type=str,
        required=True,
        help="Ruta al audio de referencia (WAV)"
    )
    parser.add_argument(
        "--num_audios", "-n",
        type=int,
        default=30,
        help="Cantidad de audios a generar (default: 30)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="output/dataset",
        help="Carpeta de salida"
    )
    parser.add_argument(
        "--speed", "-s",
        type=float,
        default=1.0,
        help="Velocidad (default: 1.0)"
    )
    parser.add_argument(
        "--skip_check",
        action="store_true",
        help="Saltar validación del audio de referencia"
    )
    return parser.parse_args()


def print_summary(args):
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="dim")
    table.add_column(style="bold white")
    table.add_row("Audio referencia", args.reference)
    table.add_row("Audios a generar", str(args.num_audios))
    table.add_row("Idioma", "ESPAÑOL (Latino)")
    table.add_row("Motor", "F5-Spanish (Juan Gallego)")
    table.add_row("Soporte GPU", "SÍ (Forzado)")
    table.add_row("Carpeta salida", args.output)
    dur_min = args.num_audios * 1
    table.add_row("Dataset total aprox.", f"~{dur_min} minutos")
    console.print(Panel(table, title="[bold cyan]Configuración de Generación[/bold cyan]", border_style="cyan"))


def run_pipeline(args):
    console.print(BANNER)
    print_summary(args)
    console.print()

    # ── 1. Validar ────────────────────────────────────────────────
    if not args.skip_check:
        console.rule("[bold]Paso 1 — Validando audio[/bold]")
        from utils.audio_check import check_reference_audio
        if not check_reference_audio(args.reference):
            sys.exit(1)
        console.print("[green]✓ Audio okay[/green]\n")

    # ── 2. Motor (Hardcoded F5-Spanish) ───────────────────────────
    console.rule("[bold]Paso 2 — Iniciando motor F5-Spanish (GPU)[/bold]")
    from tts.f5_cloner import F5Cloner
    cloner = F5Cloner() # Usa valores por defecto (ES + GPU)
    
    # ── 3. Perfil ─────────────────────────────────────────────────
    console.rule("[bold]Paso 3 — Extrayendo perfil de voz[/bold]")
    cloner.extract_voice_profile(args.reference)
    console.print("[green]✓ Perfil extraído[/green]\n")

    # ── 4. Textos ─────────────────────────────────────────────────
    console.rule("[bold]Paso 4 — Seleccionando textos en español[/bold]")
    from texts.text_selector import select_texts
    texts = select_texts(language="ES", count=args.num_audios)
    console.print(f"[green]✓ {len(texts)} textos seleccionados[/green]\n")

    # ── 5. Dataset ────────────────────────────────────────────────
    console.rule("[bold]Paso 5 — Generando dataset sintético[/bold]")
    from tts.generator import DatasetGenerator
    generator = DatasetGenerator(cloner=cloner, output_dir=args.output)

    os.makedirs(args.output, exist_ok=True)

    start_time = time.time()
    generated = generator.generate_batch(texts=texts, speed=args.speed)
    elapsed = time.time() - start_time

    console.print()
    console.print(Panel(
        f"[bold green]✓ Dataset terminado[/bold green]\n\n"
        f"  [white]Ubicación:[/white] [cyan]{args.output}[/cyan]\n"
        f"  [white]Tiempo:[/white] [cyan]{elapsed:.1f}s[/cyan]\n",
        title="[bold]Éxito[/bold]",
        border_style="green"
    ))

    console.print("\n[bold cyan]1.[/bold cyan] Abre Applio [bold]Train[/bold] → Apunta a: [cyan]output/dataset/[/cyan]")
    console.print("[bold cyan]2.[/bold cyan] Entrena y usa tu voz en tiempo real.")


if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.reference):
        console.print(f"[red]Error: {args.reference} no existe[/red]")
        sys.exit(1)
    run_pipeline(args)
