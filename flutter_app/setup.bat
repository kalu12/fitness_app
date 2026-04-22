@echo off
:: setup.bat — finishes the Flutter project setup.
:: Run this once after cloning/opening the project.
:: Requirements: Flutter SDK installed, Android SDK installed.

echo.
echo ===========================================================
echo  Fitness Form Checker — Flutter project setup
echo ===========================================================
echo.

where flutter >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Flutter not found in PATH.
    echo Download from https://flutter.dev/docs/get-started/install
    pause
    exit /b 1
)

flutter --version

echo.
echo [1/3] Running flutter pub get ...
flutter pub get
if %errorlevel% neq 0 (
    echo ERROR: flutter pub get failed.
    pause
    exit /b 1
)

echo.
echo [2/3] Checking connected devices ...
flutter devices

echo.
echo [3/3] All done!
echo.
echo To run on a connected Android device or emulator:
echo     flutter run
echo.
echo To build an APK:
echo     flutter build apk --release
echo.
echo Remember to start the Python server first:
echo     cd ..
echo     pip install fastapi uvicorn python-multipart
echo     python server.py
echo.
pause
