@echo off
setlocal DisableDelayedExpansion
title Federated Learning Presentation Demo

for %%I in ("%~dp0.") do set "REPO_DIR=%%~fI"
set "WIN_LAUNCHER=%REPO_DIR%\scripts\live\presentation_demo.sh"

where.exe wsl.exe >nul 2>&1
if errorlevel 1 (
  echo ERROR: WSL was not found.
  echo.
  pause
  exit /b 10
)

if not exist "%WIN_LAUNCHER%" (
  echo ERROR: presentation launcher was not found:
  echo "%WIN_LAUNCHER%"
  echo.
  pause
  exit /b 11
)

echo Starting the federated-learning presentation demo...
echo Repository: "%REPO_DIR%"
echo.

wsl.exe --cd "%REPO_DIR%" --exec /bin/bash ./scripts/live/presentation_demo.sh
set "DEMO_EXIT=%ERRORLEVEL%"

echo.
if "%DEMO_EXIT%"=="0" (
  echo Demo stopped normally.
) else (
  echo ERROR: demo exited with code %DEMO_EXIT%.
)
echo.
pause
endlocal & exit /b %DEMO_EXIT%
