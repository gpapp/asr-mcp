@echo off
chcp 65001 >nul
setlocal
set "HERE=%~dp0"
set "CLIENT=%HERE%transcribe_client.py"
set "LIVE_CLIENT=%HERE%live_client.py"
set "VENV_PY=%HERE%.venv\Scripts\python.exe"

rem "live" runs the real-time client, which needs audio dependencies.
set "REQS=%HERE%requirements.txt"
set "SCRIPT=%CLIENT%"
if /i "%~1"=="live" (
    set "REQS=%HERE%requirements-live.txt"
    set "SCRIPT=%LIVE_CLIENT%"
    shift
)

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
rem Install only when the requirements file has non-comment lines (pip errors on an empty set).
findstr /r /v /c:"^#" /c:"^$" "%REQS%" >nul 2>nul
if errorlevel 1 goto :run_venv
"%VENV_PY%" -m pip install --quiet --disable-pip-version-check -r "%REQS%"

:run_venv
"%VENV_PY%" "%SCRIPT%" %*
goto :done

rem Fallback when the venv could not be created. Use %SCRIPT% here too, or
rem "live" would silently fall back to the file client.
:run_fallback
where py >nul 2>nul
if not errorlevel 1 goto :run_py
python "%SCRIPT%" %*
goto :done

:run_py
py "%SCRIPT%" %*

:done
if not errorlevel 1 exit /b 0
echo.
pause
exit /b 1
