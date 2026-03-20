@echo off
setlocal enabledelayedexpansion
title NEXT-ID VOICE CLONER - Instalador Unificado

set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

echo.
echo  ================================================
echo  NEXT-ID VOICE CLONER - Instalacion Profesional
echo  Desarrollado por: Julian Posada
echo  ================================================
echo.

:: 1. Verificaciones de Sistema
echo [1/7] Verificando requisitos de sistema...
where conda >nul 2>&1
if !errorlevel! neq 0 (
    echo ERROR: Conda no encontrado. Por favor instala Miniconda o Anaconda.
    pause & exit /b 1
)
where git >nul 2>&1
if !errorlevel! neq 0 (
    echo ERROR: Git no encontrado. Por favor instala Git para Windows.
    pause & exit /b 1
)
echo OK - Requisitos base encontrados.

:: 2. Creacion de Entorno
echo [2/7] Configurando entorno virtual Python 3.10...
if not exist "%ROOT%\env" (
    call conda create -p "%ROOT%\env" python=3.10 -y
) else (
    echo Entorno ya existe, omitiendo creacion.
)
set "PY=%ROOT%\env\python.exe"
set "PIP=%ROOT%\env\Scripts\pip.exe"

:: 3. Instalacion de PyTorch (Optimizado para Blackwell/RTX 5060)
echo [3/7] Instalando PyTorch con soporte Nativo para RTX 5060 (Blackwell)...
echo (Esto instalara PyTorch Nightly con CUDA 12.8 para maximo rendimiento)
"!PIP!" uninstall torch torchvision torchaudio -y >nul 2>&1
"!PIP!" install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
if !errorlevel! neq 0 (
    echo ERROR: No se pudo instalar PyTorch Nightly.
    pause & exit /b 1
)

:: 4. Dependencias del Proyecto
echo [4/7] Instalando dependencias de NEXT-ID VOICE CLONER...
"!PIP!" install -r "%ROOT%\requirements.txt"
"!PIP!" install f5-tts gradio sounddevice soundfile numpy scipy librosa tqdm pydub colorama rich pyinstaller
echo OK - Dependencias base instaladas.

:: 5. Clonacion de Applio
echo [5/7] Configurando motor Applio RVC...
if not exist "%ROOT%\Applio" (
    git clone https://github.com/IAHispano/Applio.git "%ROOT%\Applio"
)
if exist "%ROOT%\Applio\requirements.txt" (
    "!PIP!" install -r "%ROOT%\Applio\requirements.txt"
)

:: 6. Estructura de Carpetas
echo [6/7] Creando estructura de directorios...
if not exist "%ROOT%\output\dataset" mkdir "%ROOT%\output\dataset"
if not exist "%ROOT%\output\models"  mkdir "%ROOT%\output\models"
if not exist "%ROOT%\output\logs"    mkdir "%ROOT%\output\logs"
if not exist "%ROOT%\debug_tests"    mkdir "%ROOT%\debug_tests"

:: 7. Finalizacion y Lanzador
echo [7/7] Creando archivos de inicio...

:: Crear INICIAR.bat
(
echo @echo off
echo title NEXT-ID VOICE CLONER
echo cd /d "%%~dp0"
echo echo Iniciando sistema...
echo start http://localhost:7860
echo "%%~dp0env\python.exe" "%%~dp0gui.py"
echo pause
) > "%ROOT%\INICIAR.bat"

:: Crear ACTUALIZAR.bat
(
echo @echo off
echo title NEXT-ID VOICE CLONER - Actualizador
echo cd /d "%%~dp0"
echo echo Actualizando dependencias y motores...
echo "%%~dp0env\Scripts\pip.exe" install --upgrade f5-tts
echo "%%~dp0env\Scripts\pip.exe" install --upgrade -r "%%~dp0requirements.txt"
echo if exist "%%~dp0Applio" (
echo     cd Applio
echo     git pull
echo     "../../env/Scripts/pip.exe" install -r requirements.txt
echo     cd ..
echo )
echo echo Actualizacion completada.
echo pause
) > "%ROOT%\ACTUALIZAR.bat"

echo.
echo ================================================
echo INSTALACION COMPLETADA EXITOSAMENTE
echo ================================================
echo Desarrollador: Julian Posada
echo.
echo Usa INICIAR.bat para arrancar el programa.
echo Usa ACTUALIZAR.bat para mantener el sistema al dia.
echo.
pause
