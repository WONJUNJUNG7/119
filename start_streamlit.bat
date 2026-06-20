@echo off
echo ========================================
echo Fire Safety Analysis System Setup
echo ========================================
echo.
echo Checking Python installation...
C:\Users\Administrator\AppData\Local\Programs\Python\Python313\python.exe --version
echo.
echo Starting Streamlit server...
echo The application will open in your browser shortly.
echo.
echo If you see library errors, close this window and run this script again.
echo.
echo.
echo Checking and installing required packages...
echo.

REM Get the directory where this script is located
set SCRIPT_DIR=%~dp0

REM Install required packages if not already installed
echo Installing: streamlit, pandas, numpy, pydeck, plotly, folium, streamlit-folium
C:\Users\Administrator\AppData\Local\Programs\Python\Python313\python.exe -m pip install --upgrade --force-reinstall streamlit pandas numpy pydeck plotly folium streamlit-folium

C:\Users\Administrator\AppData\Local\Programs\Python\Python313\python.exe -m streamlit run "%SCRIPT_DIR%streamlit_app.py" --server.port 8501 --server.headless true
echo.
echo Server stopped. Press any key to exit.
pause > nul
