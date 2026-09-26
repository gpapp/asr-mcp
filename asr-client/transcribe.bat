@echo off
chcp 65001 >nul
setlocal
set "CLIENT=%~dp0transcribe_client.py"

where py >nul 2>nul
if not errorlevel 1 (
    py "%CLIENT%" %*
) else (
    python "%CLIENT%" %*
)

if not errorlevel 1 exit /b 0
echo.
pause
