@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0compare-step-times.ps1" %*
exit /b %errorlevel%
