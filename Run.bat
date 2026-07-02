@echo off
REM ============================================================================
REM  Cineforge one-click launcher (Windows).
REM  FLAT goto-label structure ONLY. Do NOT wrap logic in big parenthesized
REM  if-blocks: escaped parens / special chars inside a paren block make cmd.exe
REM  abort with ". was unexpected at this time" and the window closes instantly.
REM ============================================================================
setlocal EnableExtensions
cd /d "%~dp0"

if exist ".installing" goto BUSY
if not exist ".installed" goto SETUP
goto LAUNCH

:SETUP
echo [Cineforge] First-time setup. This downloads Python, torch, ComfyUI and models.
echo installing> ".installing"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1"
if errorlevel 1 goto SETUPFAIL
del ".installing" >nul 2>&1
echo [Cineforge] Setup complete.
goto LAUNCH

:SETUPFAIL
del ".installing" >nul 2>&1
echo [Cineforge] Setup FAILED. See docs\TROUBLESHOOTING.md and re-run Run.bat.
pause
exit /b 1

:BUSY
echo [Cineforge] Setup is already running in another window. Please wait for it to finish.
pause
exit /b 0

:LAUNCH
set "PYEXE=%~dp0python_embeded\Scripts\python.exe"
if not exist "%PYEXE%" set "PYEXE=%~dp0python_embeded\python.exe"
if not exist "%PYEXE%" set "PYEXE=python"
echo [Cineforge] Launching GUI...
"%PYEXE%" -m cineforge gui
exit /b %errorlevel%
