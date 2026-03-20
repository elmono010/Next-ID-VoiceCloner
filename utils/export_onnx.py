"""
utils/export_onnx.py
Convierte un modelo RVC .PTH a .ONNX para menor latencia en tiempo real.
Uso: python utils/export_onnx.py --model output/models/mi_modelo.pth
"""

import argparse
import sys
from pathlib import Path
from rich.console import Console

console = Console()


def export_to_onnx(pth_path: str, output_path: str = None):
    """
    Convierte un modelo RVC .pth a .onnx.
    
    Nota: Esta función requiere que Applio esté instalado,
    ya que utiliza la arquitectura de modelos de RVC.
    
    Alternativa recomendada: exportar desde W-Okada directamente
    (más confiable para uso en tiempo real).
    """
    pth_path = Path(pth_path)

    if not pth_path.exists():
        console.print(f"[red]Error: No se encuentra {pth_path}[/red]")
        sys.exit(1)

    if output_path is None:
        output_path = pth_path.with_suffix(".onnx")
    output_path = Path(output_path)

    console.print(f"\n[bold cyan]Convirtiendo .PTH → .ONNX[/bold cyan]")
    console.print(f"  Entrada: [cyan]{pth_path}[/cyan]")
    console.print(f"  Salida:  [cyan]{output_path}[/cyan]\n")

    try:
        import torch

        # Cargar el modelo PTH
        console.print("[dim]Cargando modelo PTH...[/dim]")
        checkpoint = torch.load(pth_path, map_location="cpu")

        # Intentar exportación usando las utilidades de RVC/Applio
        # Esto requiere que Applio esté en el Python path
        try:
            # Método 1: vía Applio (si está instalado)
            sys.path.insert(0, ".")
            from rvc.lib.tools.export_onnx import export_onnx
            export_onnx(str(pth_path), str(output_path))
            console.print(f"\n[green]✓ Modelo exportado: {output_path}[/green]")

        except ImportError:
            # Método 2: instrucciones para hacerlo desde W-Okada
            console.print(
                "\n[yellow]Applio no está en el path para exportación automática.[/yellow]\n"
                "\n[bold]Exportación desde W-Okada (recomendado):[/bold]\n"
                "  1. Abre W-Okada Voice Changer\n"
                "  2. Carga tu modelo .PTH en la interfaz\n"
                "  3. Haz clic en [bold]'Export to .onnx'[/bold]\n"
                "  4. El .onnx se guarda en la misma carpeta que el .pth\n"
                "\n[dim]El .onnx puede reducir la latencia en tiempo real[/dim]\n"
                "[dim]pero el .pth tiene mejor calidad de audio.[/dim]"
            )

    except Exception as e:
        console.print(f"[red]Error durante la exportación: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convertir modelo RVC .PTH a .ONNX")
    parser.add_argument("--model", "-m", required=True, help="Ruta al archivo .pth")
    parser.add_argument("--output", "-o", help="Ruta de salida .onnx (opcional)")
    args = parser.parse_args()

    export_to_onnx(args.model, args.output)
