@echo off
echo ======================================
echo LinkedIn CV Classifier wird gestoppt...
echo ======================================
echo.

REM Finde und stoppe alle Streamlit-Prozesse
for /f "tokens=2" %%i in ('tasklist ^| findstr "streamlit"') do (
    echo Stoppe Streamlit-Prozess %%i...
    taskkill /PID %%i /F
)

echo.
echo App wurde gestoppt.
echo.
pause
