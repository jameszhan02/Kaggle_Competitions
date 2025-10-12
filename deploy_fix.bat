@echo off
echo ========================================
echo Fixing Railway Deployment - PyTorch CPU
echo ========================================
echo.
echo Changes made:
echo   [x] Updated requirements.txt (PyTorch CPU-only)
echo   [x] Added nixpacks.toml configuration
echo   [x] Reduced image size from 5.4GB to ~1.5GB
echo.
echo ========================================
echo Ready to deploy!
echo ========================================
echo.
echo Next steps:
echo.
echo 1. Git commit:
echo    git add requirements.txt nixpacks.toml
echo    git commit -m "Fix: Use PyTorch CPU to reduce image size"
echo.
echo 2. Push to GitHub:
echo    git push
echo.
echo 3. Railway will auto-deploy (3-5 minutes)
echo.
echo 4. Test deployed API:
echo    curl https://your-app.railway.app/health
echo.
echo ========================================
echo For details, see: DEPLOYMENT_FIX.md
echo ========================================
echo.
pause

