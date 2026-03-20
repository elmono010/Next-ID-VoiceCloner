"""
tts/generator.py
Generador de dataset en lote con barra de progreso y manejo de errores.
"""

import os
import time
import soundfile as sf
import numpy as np
from pathlib import Path
from rich.progress import (
    Progress, SpinnerColumn, BarColumn,
    TextColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
)
from rich.console import Console

console = Console()


class DatasetGenerator:
    """
    Genera lotes de audios para el dataset de entrenamiento RVC.
    
    Cada audio:
    - Formato WAV, 44100 Hz, mono (óptimo para RVC)
    - Nombre: audio_001.wav, audio_002.wav, ...
    - ~1 minuto de duración (depende del texto)
    """

    def __init__(self, cloner, output_dir: str = "output/dataset"):
        self.cloner = cloner
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_batch(
        self,
        texts: list[str],
        speed: float = 1.0,
        show_progress: bool = True
    ) -> int:
        """
        Genera todos los audios del dataset.
        
        Args:
            texts: lista de textos a sintetizar
            speed: velocidad de habla
            show_progress: mostrar barra de progreso
        
        Returns:
            Número de audios generados exitosamente
        """
        total = len(texts)
        generated = 0
        failed = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(bar_width=30),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=console,
            disable=not show_progress
        ) as progress:

            task = progress.add_task(
                f"Generando audios ({self.cloner.language})",
                total=total
            )

            for i, text in enumerate(texts):
                output_path = self.output_dir / f"audio_{i+1:03d}.wav"
                progress.update(
                    task,
                    description=f"[bold cyan]{i+1:03d}/{total:03d}[/bold cyan] [dim]{text[:55]}...[/dim]" if len(text) > 55
                    else f"[bold cyan]{i+1:03d}/{total:03d}[/bold cyan] [dim]{text}[/dim]"
                )

                try:
                    self.cloner.synthesize(
                        text=text,
                        output_path=str(output_path),
                        speed=speed
                    )

                    # Verificar que el audio se generó y tiene duración razonable
                    if output_path.exists():
                        data, sr = sf.read(str(output_path))
                        duration = len(data) / sr
                        if duration < 0.5:
                            raise ValueError(f"Audio muy corto: {duration:.1f}s")
                        generated += 1
                    else:
                        raise FileNotFoundError("El archivo no fue creado")

                except Exception as e:
                    failed.append((i + 1, text[:50], str(e)))
                    console.print(
                        f"\n  [yellow]⚠ Audio {i+1:03d} falló:[/yellow] [dim]{e}[/dim]"
                    )

                progress.advance(task)

        # Reporte final
        if failed:
            console.print(f"\n  [yellow]Fallidos ({len(failed)}):[/yellow]")
            for idx, txt, err in failed:
                console.print(f"    [dim]{idx:03d}: {txt}... → {err}[/dim]")

        # Generar archivo de metadatos para Applio
        self._write_metadata(texts, generated)

        return generated

    def _write_metadata(self, texts: list[str], count: int):
        """
        Escribe un archivo de metadatos con info del dataset.
        Útil para reproducibilidad y debugging.
        """
        meta_path = self.output_dir / "dataset_info.txt"
        with open(meta_path, "w", encoding="utf-8") as f:
            f.write(f"Dataset generado con Voice Cloner\n")
            f.write(f"Idioma: {self.cloner.language}\n")
            f.write(f"Audios generados: {count}\n")
            f.write(f"Formato: WAV 44100Hz mono\n")
            f.write(f"\n{'='*50}\n")
            f.write("TEXTOS:\n")
            f.write(f"{'='*50}\n\n")
            for i, text in enumerate(texts[:count], 1):
                f.write(f"{i:03d}: {text}\n\n")
        console.print(f"  [dim]Metadatos guardados en: {meta_path}[/dim]")