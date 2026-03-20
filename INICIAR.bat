@echo off
title Voice Cloner
set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"
set "PYTHONHASHSEED=0"
set "PYTHONIOENCODING=utf-8"
echo Iniciando Voice Cloner...
"%ROOT%\env\python.exe" "%ROOT%\gui.py"
pause
