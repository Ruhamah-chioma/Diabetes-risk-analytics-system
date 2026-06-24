@echo off
echo ========================================
echo Starting Diabetes Risk Ensemble API
echo ========================================
echo.

cd /d "%~dp0"

echo Installing dependencies if needed...
pip install -r requirements.txt

echo.
echo Starting server...
echo API will be available at http://localhost:8001
echo Interactive docs at http://localhost:8001/docs
echo.
echo Press Ctrl+C to stop the server
echo ========================================

python ensemble_api.py