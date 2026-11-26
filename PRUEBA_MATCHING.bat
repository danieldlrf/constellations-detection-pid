@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
title Prueba del matching de constelaciones
set PYTHONUTF8=1

echo ======================================================================
echo  Prueba del matching: original (main) frente a arreglado
echo ======================================================================

rem --- Entorno virtual en .venv (solo la primera vez) ---
if exist ".venv\Scripts\python.exe" goto deps

set "PYBASE="
py -3.13 -c "1" >nul 2>&1 && set "PYBASE=py -3.13"
if not defined PYBASE if exist "C:\Program Files\Python313\python.exe" set "PYBASE="C:\Program Files\Python313\python.exe""
if not defined PYBASE py -3 -c "1" >nul 2>&1 && set "PYBASE=py -3"
if not defined PYBASE python -c "1" >nul 2>&1 && set "PYBASE=python"
if not defined PYBASE (
    echo No se ha encontrado Python. Instala Python 3.13 desde python.org y vuelve a probar.
    goto fin
)
echo Creando entorno virtual en .venv con %PYBASE% ...
%PYBASE% -m venv .venv
if errorlevel 1 (
    echo No se pudo crear el entorno virtual.
    goto fin
)

:deps
echo Comprobando dependencias (opencv, numpy, scipy, matplotlib)...
".venv\Scripts\python.exe" -c "import cv2, numpy, scipy, matplotlib" >nul 2>&1
if errorlevel 1 (
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo Fallo instalando dependencias.
        goto fin
    )
)

rem --- La prueba ---
".venv\Scripts\python.exe" prueba\prueba_matching.py
if errorlevel 1 (
    echo.
    echo La prueba ha fallado. Revisa los mensajes de arriba.
    goto fin
)
if exist "resultados_prueba\ejemplos_matching.png" start "" "resultados_prueba\ejemplos_matching.png"
start "" "resultados_prueba"

:fin
echo.
pause
