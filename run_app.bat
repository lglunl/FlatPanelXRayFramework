@echo off
chcp 65001 >nul
echo ============================================
echo  平板X射线去混叠成像算法框架 - 启动器
echo ============================================
cd /d "%~dp0"

REM 优先使用项目虚拟环境
if exist ".venv\Scripts\python.exe" (
    set PYTHON=.venv\Scripts\python.exe
) else (
    set PYTHON=python
)

%PYTHON% -m streamlit run app.py --server.headless false
pause
