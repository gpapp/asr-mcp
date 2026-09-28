@echo off
chcp 65001 >nul
setlocal
set "HERE=%~dp0"
set "CLIENT=%HERE%transcribe_client.py"
set "VENV_PY=%HERE%.venv\Scripts\python.exe"

rem First run: create local .env from the template, then stop so it can be edited.
if not exist "%HERE%.env" (
    copy /y "%HERE%.env.example" "%HERE%.env" >nul
    echo Created .env from .env.example - edit SERVER_URL and TOKEN, then run again.
    echo.
    pause
    exit /b 1
)

rem First run: create the local virtual environment.
if exist "%VENV_PY%" goto :check_req
echo Creating local virtual environment in .venv ...
where py >nul 2>nul
if not errorlevel 1 goto :venv_py
python -m venv "%HERE%.venv"
goto :check_req

:venv_py
py -3 -m venv "%HERE%.venv"

:check_req
if not exist "%VENV_PY%" goto :run_fallback
rem Install only when requirements.txt has non-comment lines (pip errors on an empty set).
findstr /r /v /c:"^#" /c:"^$" "%HERE%requirements.txt" >nul 2>nul
if errorlevel 1 goto :run_venv
"%VENV_PY%" -m pip install --quiet --disable-pip-version-check -r "%HERE%requirements.txt"

:run_venv
"%VENV_PY%" "%CLIENT%" %*
goto :done

:run_fallback
where py >nul 2>nul
if not errorlevel 1 goto :run_py
python "%CLIENT%" %*
goto :done

:run_py
py "%CLIENT%" %*

:done
if not errorlevel 1 exit /b 0
echo.
pause
exit /b 1
