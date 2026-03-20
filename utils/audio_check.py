"""
utils/audio_check.py
Validador de calidad del audio de referencia antes de clonar.
Verifica duración, frecuencia de muestreo, nivel de ruido y recorte.
"""

import numpy as np
import soundfile as sf
from rich.console import Console
from rich.table import Table

console = Console()


def check_reference_audio(filepath: str) -> bool:
    """
    Valida que el audio de referencia cumple los requisitos mínimos.
    
    Criterios:
    - Duración: mínimo 10s, ideal 30s-3min
    - Sample rate: 22050 Hz o superior
    - No silencio excesivo (>40% del total)
    - Sin clipping grave (>1% de muestras saturadas)
    - Nivel de señal adecuado (RMS > -30 dBFS)
    
    Returns:
        True si el audio es válido, False si tiene problemas graves
    """
    try:
        data, sr = sf.read(filepath)
    except Exception as e:
        console.print(f"  [red]✗ No se pudo leer el archivo: {e}[/red]")
        return False

    # Convertir a mono si es estéreo
    if len(data.shape) > 1:
        data = data.mean(axis=1)

    duration = len(data) / sr
    issues = []
    warnings = []
    passed = []

    # ── Duración ────────────────────────────────────────────────────
    if duration < 10:
        issues.append(f"Audio muy corto: {duration:.1f}s (mínimo 10s)")
    elif duration < 30:
        warnings.append(f"Audio corto: {duration:.1f}s (recomendado 30s-3min)")
        passed.append(f"Duración: {duration:.1f}s")
    elif duration > 600:
        warnings.append(f"Audio muy largo: {duration:.1f}s (máximo recomendado 10min)")
        passed.append(f"Duración: {duration:.1f}s")
    else:
        passed.append(f"Duración: {duration:.1f}s ✓")

    # ── Sample rate ──────────────────────────────────────────────────
    if sr < 16000:
        issues.append(f"Sample rate muy bajo: {sr}Hz (mínimo 16000Hz)")
    elif sr < 22050:
        warnings.append(f"Sample rate bajo: {sr}Hz (recomendado 22050Hz+)")
    else:
        passed.append(f"Sample rate: {sr}Hz ✓")

    # ── Clipping ─────────────────────────────────────────────────────
    clip_threshold = 0.999
    clipped = np.sum(np.abs(data) >= clip_threshold) / len(data)
    if clipped > 0.05:
        issues.append(f"Clipping severo: {clipped:.1%} de muestras saturadas")
    elif clipped > 0.01:
        warnings.append(f"Clipping leve: {clipped:.1%} de muestras saturadas")
    else:
        passed.append(f"Clipping: {clipped:.2%} ✓")

    # ── Nivel RMS ────────────────────────────────────────────────────
    rms = np.sqrt(np.mean(data**2))
    rms_db = 20 * np.log10(rms + 1e-10)
    if rms_db < -40:
        issues.append(f"Nivel muy bajo: {rms_db:.1f} dBFS (mínimo -40 dBFS)")
    elif rms_db < -30:
        warnings.append(f"Nivel bajo: {rms_db:.1f} dBFS (recomendado > -30 dBFS)")
    else:
        passed.append(f"Nivel RMS: {rms_db:.1f} dBFS ✓")

    # ── Porcentaje de silencio ───────────────────────────────────────
    silence_threshold = 0.01
    silence_ratio = np.sum(np.abs(data) < silence_threshold) / len(data)
    if silence_ratio > 0.6:
        issues.append(f"Demasiado silencio: {silence_ratio:.0%} del audio")
    elif silence_ratio > 0.4:
        warnings.append(f"Mucho silencio: {silence_ratio:.0%} del audio")
    else:
        passed.append(f"Ratio silencio: {silence_ratio:.0%} ✓")

    # ── Mostrar resultados ───────────────────────────────────────────
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="dim", width=4)
    table.add_column()

    for p in passed:
        table.add_row("[green]✓[/green]", f"[green]{p}[/green]")
    for w in warnings:
        table.add_row("[yellow]⚠[/yellow]", f"[yellow]{w}[/yellow]")
    for e in issues:
        table.add_row("[red]✗[/red]", f"[red]{e}[/red]")

    console.print(table)

    if issues:
        console.print("\n  [red bold]El audio de referencia tiene problemas graves.[/red bold]")
        console.print("  [dim]Consejos:[/dim]")
        console.print("    [dim]• Graba en un ambiente silencioso, sin eco[/dim]")
        console.print("    [dim]• Habla claro y a volumen normal, sin gritar[/dim]")
        console.print("    [dim]• Duración ideal: 1-3 minutos de habla continua[/dim]")
        return False

    if warnings:
        console.print(
            "\n  [yellow]El audio tiene advertencias pero se puede usar.[/yellow]"
        )

    return True
