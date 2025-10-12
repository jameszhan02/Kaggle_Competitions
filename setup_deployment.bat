@echo off
echo ========================================
echo Setting up Digit Recognizer API
echo ========================================
echo.

REM Create models directory if it doesn't exist
if not exist "models" (
    echo Creating models directory...
    mkdir models
    echo [OK] models/ created
) else (
    echo [OK] models/ already exists
)

REM Copy model file from server/models to root models
if exist "server\models\digit_model.pth" (
    echo Copying model file...
    copy server\models\digit_model.pth models\digit_model.pth >nul
    echo [OK] Model file copied to models/digit_model.pth
) else (
    echo [WARNING] Model file not found at server/models/digit_model.pth
    echo Please ensure the model file exists!
)

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Files ready for deployment:
echo   [x] app.py
echo   [x] requirements.txt
echo   [x] Procfile
echo   [x] runtime.txt
echo   [x] models/digit_model.pth
echo.
echo Next steps:
echo   1. Test locally: python app.py
echo   2. Commit to git: git add . ^&^& git commit -m "Deploy API"
echo   3. Push to GitHub: git push
echo   4. Deploy on Railway: https://railway.app
echo.
pause

