import os
import sys
import subprocess
from pathlib import Path

def main():
    root = Path(__file__).parent
    gui_script = root / "gui.py"
    python_exe = root / "env" / "python.exe"

    if not python_exe.exists():
        print(f"ERROR: No se encontro el entorno en {python_exe}")
        print("Por favor, ejecuta INSTALAR.bat primero.")
        input("Presiona una tecla para salir...")
        sys.exit(1)

    # Iniciar gui.py usando el python del entorno
    try:
        subprocess.run([str(python_exe), str(gui_script)], check=True)
    except Exception as e:
        print(f"Error al iniciar: {e}")
        input("Presiona una tecla para salir...")

if __name__ == "__main__":
    main()
