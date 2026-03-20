"""
utils/download_checkpoints.py
Descarga los checkpoints de OpenVoice V2 desde HuggingFace Hub.
"""

import os
import sys
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, DownloadColumn, TransferSpeedColumn

console = Console()

CHECKPOINTS_DIR = Path("OpenVoice/checkpoints_v2")

# Archivos necesarios de OpenVoice V2
# Fuente: https://huggingface.co/myshell-ai/OpenVoiceV2
REQUIRED_FILES = {
    "converter/config.json": "https://huggingface.co/myshell-ai/OpenVoiceV2/resolve/main/converter/config.json",
    "converter/checkpoint.pth": "https://huggingface.co/myshell-ai/OpenVoiceV2/resolve/main/converter/checkpoint.pth",
}


def download_file(url: str, dest: Path):
    """Descarga un archivo con barra de progreso."""
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)

    def reporthook(count, block_size, total_size):
        pass

    urllib.request.urlretrieve(url, dest, reporthook)


def check_and_download():
    console.print("\n[bold cyan]Verificando checkpoints de OpenVoice V2...[/bold cyan]\n")

    all_present = True
    for rel_path, url in REQUIRED_FILES.items():
        dest = CHECKPOINTS_DIR / rel_path
        if dest.exists():
            size = dest.stat().st_size / (1024 * 1024)
            console.print(f"  [green]✓[/green] {rel_path} [dim]({size:.1f} MB)[/dim]")
        else:
            console.print(f"  [red]✗[/red] {rel_path} — [dim]no encontrado[/dim]")
            all_present = False

    if all_present:
        console.print("\n[green]Todos los checkpoints están presentes. Listo para usar.[/green]")
        return

    console.print(
        "\n[yellow]Faltan checkpoints. Descargando desde HuggingFace Hub...[/yellow]\n"
        "[dim]Asegúrate de tener conexión a internet para este paso.[/dim]\n"
    )

    try:
        # Intentar con huggingface_hub si está disponible
        try:
            from huggingface_hub import snapshot_download
            console.print("[dim]Usando huggingface_hub...[/dim]")
            snapshot_download(
                repo_id="myshell-ai/OpenVoiceV2",
                local_dir=str(CHECKPOINTS_DIR),
                repo_type="model"
            )
            console.print("\n[green]✓ Checkpoints descargados correctamente.[/green]")
            return
        except ImportError:
            pass

        # Fallback: descarga directa
        console.print("[dim]Descargando archivos individualmente...[/dim]\n")
        for rel_path, url in REQUIRED_FILES.items():
            dest = CHECKPOINTS_DIR / rel_path
            if dest.exists():
                continue
            console.print(f"  Descargando [cyan]{rel_path}[/cyan]...")
            try:
                download_file(url, dest)
                size = dest.stat().st_size / (1024 * 1024)
                console.print(f"  [green]✓[/green] {rel_path} [dim]({size:.1f} MB)[/dim]")
            except Exception as e:
                console.print(f"  [red]✗ Error descargando {rel_path}: {e}[/red]")

    except Exception as e:
        console.print(f"\n[red]Error durante la descarga: {e}[/red]")
        console.print(
            "\n[yellow]Descarga manual:[/yellow]\n"
            "  1. Ve a https://huggingface.co/myshell-ai/OpenVoiceV2\n"
            "  2. Descarga los archivos de la carpeta 'converter'\n"
            f"  3. Colócalos en: [cyan]{CHECKPOINTS_DIR}/converter/[/cyan]"
        )
        sys.exit(1)

    console.print("\n[green]✓ Checkpoints listos.[/green]")


if __name__ == "__main__":
    check_and_download()
