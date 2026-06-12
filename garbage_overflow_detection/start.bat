@echo off
setlocal EnableDelayedExpansion

echo ============================================================
echo   GARBAGE OVERFLOW DETECTION - STARTUP MENU
echo   India/Tamil Nadu Fine-Tuned Model
echo ============================================================
echo.
echo Please select the camera source you want to use for detection:
echo.
echo   [0] System Webcam (Default Laptop/PC Camera)
echo   [1] External Camera (USB Webcam, Android Cam, etc.)
echo   [2] Synthetic Test Video (test_video.mp4)
echo   [3] India Sample Video (sample_india_video.mp4) [RECOMMENDED]
echo.

set /p choice="Enter your choice (0/1/2/3) [default 3]: "

if "%choice%"=="" set choice=3

if "%choice%"=="0" (
    set source=0
    set source_name=System Webcam
) else if "%choice%"=="1" (
    set source=1
    set source_name=External Camera
) else if "%choice%"=="2" (
    set source=test_video.mp4
    set source_name=Synthetic Test Video
) else if "%choice%"=="3" (
    set source=sample_india_video.mp4
    set source_name=India Sample Video
) else (
    set source=sample_india_video.mp4
    set source_name=India Sample Video
)

echo.
echo ============================================================
echo Starting Backend and Frontend with: %source_name%
echo ============================================================
echo.
echo This will open your browser to the dashboard automatically.
echo Press CTRL+C in this window to stop the server when done.
echo.

:: Start the browser
start http://localhost:5000

:: Run the Flask app with the selected camera source
.\venv\Scripts\python.exe app.py --with-detection --source %source%

pause

