@echo off
REM Generate status reports for GitHub visibility
REM Run from project root: scripts\status.bat

cd /d "%~dp0.."

echo Generating status reports...
python src/status_report.py

if "%1"=="push" (
    echo.
    echo Committing and pushing to GitHub...
    git add reports/STATUS.md reports/status.json
    git commit -m "chore: update status report"
    git push
    echo Done!
) else (
    echo.
    echo Status generated. To push to GitHub, run: scripts\status.bat push
)
