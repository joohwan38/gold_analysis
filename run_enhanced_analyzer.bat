@echo off
echo WoW 골드 거래 분석기 Enhanced를 시작합니다...
echo.

cd /d "%~dp0"

echo 현재 디렉토리: %CD%
echo.

echo 포트 7860 사용 중인 프로세스를 종료합니다...
echo.

REM First, try to kill processes on port 7860 multiple times
:KILL_PORT
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :7860 ^| findstr LISTENING 2^>nul') do (
    echo 포트 7860 사용 중인 프로세스 ID %%a 강제 종료 중...
    taskkill //F //PID %%a >nul 2>&1
    if errorlevel 1 (
        echo 프로세스 종료 실패, 재시도 중...
        taskkill //F //PID %%a
    )
)

REM Wait a bit
timeout /t 1 /nobreak >nul

REM Check if port is still in use
netstat -ano | findstr :7860 | findstr LISTENING >nul
if %errorlevel% equ 0 (
    echo 포트가 여전히 사용 중입니다. 다시 시도 중...
    timeout /t 2 /nobreak >nul

    REM Try one more time
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr :7860 ^| findstr LISTENING 2^>nul') do (
        echo 프로세스 ID %%a 재시도 종료 중...
        taskkill //F //PID %%a
    )

    timeout /t 2 /nobreak >nul
)

REM Final check
netstat -ano | findstr :7860 | findstr LISTENING >nul
if %errorlevel% equ 0 (
    echo.
    echo [경고] 포트 7860을 사용 중인 프로세스를 종료할 수 없습니다.
    echo 수동으로 작업 관리자에서 프로세스를 종료하거나,
    echo 다음 명령을 실행해주세요:
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr :7860 ^| findstr LISTENING') do (
        echo   taskkill //F //PID %%a
    )
    echo.
    pause
    exit /b 1
) else (
    echo 포트 7860 사용 가능
)
echo.

echo 필요한 패키지를 설치합니다...
uv add plotly seaborn
echo.

echo Enhanced Gradio 웹 인터페이스를 실행중입니다...
echo 브라우저에서 자동으로 열립니다.
echo.
echo 새로운 기능:
echo - 시계열 분석 (일별 시세 추적)
echo - 데이터베이스 통합 (SQLite)
echo - 금액구간별 GPC 분석
echo - GPC 분포도 시각화
echo - 데이터 레이블 표시
echo.
echo 종료하려면 이 창에서 Ctrl+C를 누르세요.
echo.

uv run gold_analyzer_enhanced.py

pause