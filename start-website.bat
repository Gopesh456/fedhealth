@echo off
rem One click: starts the three hospitals, the coordinator and the dashboard, then opens the website.
rem First run: creates a private Python environment in .venv and installs the packages (needs internet, a few minutes).
rem Run "start-website.bat --reset" from a command prompt to forget every budget and audit entry first.
setlocal
title FedHealth
cd /d "%~dp0"

if exist "scripts\start_website.py" goto :find_python
echo This file must stay inside the DBpro folder. If you opened it from a zip file, extract the whole zip first.
goto :fail

:find_python
set "PY="
py -3 -c "import sys; sys.exit(sys.version_info < (3, 11))" >nul 2>nul
if not errorlevel 1 set "PY=py -3"
if defined PY goto :venv
python -c "import sys; sys.exit(sys.version_info < (3, 11))" >nul 2>nul
if not errorlevel 1 set "PY=python"
if defined PY goto :venv
echo Python 3.11 or newer was not found.
echo Install it from https://www.python.org/downloads/ and tick "Add python.exe to PATH", then run this file again.
goto :fail

:venv
rem Reuse .venv only if it works; an interrupted first run or a removed Python leaves a broken one, which is rebuilt.
set "VPY=.venv\Scripts\python.exe"
if not exist "%VPY%" goto :make_venv
"%VPY%" -c "import sys, pip; sys.exit(sys.version_info < (3, 11))" >nul 2>nul
if not errorlevel 1 goto :check_arch
echo The private Python environment in .venv is broken or out of date. Rebuilding it ...
:make_venv
echo Creating a private Python environment in .venv ...
%PY% -m venv --clear .venv
if errorlevel 1 goto :fail

:check_arch
"%VPY%" -c "import sysconfig, sys; sys.exit(sysconfig.get_platform() == 'win-arm64')" >nul 2>nul
if not errorlevel 1 goto :packages
echo This Python is the ARM64 build, and the dashboard needs a package that has no ARM64 Windows version.
echo Install "Windows installer (64-bit)" from https://www.python.org/downloads/windows/ (it runs fine on ARM PCs),
echo delete the .venv folder, and run this file again.
goto :fail

:packages
rem -I hides the project folder, so this sees only what is really installed in .venv.
"%VPY%" -I -c "import fedhealth, fastapi, uvicorn, sqlalchemy, httpx, pydantic, cryptography, jwt, yaml, numpy, streamlit, pandas" >nul 2>nul
if not errorlevel 1 goto :start
echo Installing the packages (first run only, a few minutes) ...
"%VPY%" -m pip install --upgrade pip
"%VPY%" -m pip install -e ".[dashboard]"
if errorlevel 1 goto :fail
rem Optional: makes the first, private patient linkage several times faster. Skipped quietly if it cannot install.
"%VPY%" -m pip install "gmpy2>=2.1" >nul 2>nul

:start
"%VPY%" scripts\start_website.py %*
if errorlevel 1 goto :fail
exit /b 0

:fail
echo.
echo FedHealth did not start. The messages above say why.
echo If it keeps failing, close every other FedHealth window, delete the .venv folder and run this file again.
pause
exit /b 1
