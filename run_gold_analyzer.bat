@echo off
echo WoW 골드 거래 분석기를 시작합니다...
echo.

cd /d "%~dp0"

echo 현재 디렉토리: %CD%
echo.

echo Gradio 웹 인터페이스를 실행중입니다...
echo 브라우저에서 자동으로 열립니다.
echo.
echo 종료하려면 이 창에서 Ctrl+C를 누르세요.
echo.

uv run gold_analyzer_gui.py

pause