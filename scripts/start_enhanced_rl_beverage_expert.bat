@echo off
echo ========================================
echo 增强版基于强化学习的自主现制饮品研发专家系统
echo ========================================
echo.

REM 检查Python环境
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误: 未找到Python环境
    echo 请先安装Python 3.7或更高版本
    pause
    exit /b 1
)

REM 检查必要的Python包
echo 检查必要的Python包...
python -c "import torch" >nul 2>&1
if %errorlevel% neq 0 (
    echo 安装PyTorch...
    pip install torch
)

python -c "import flask" >nul 2>&1
if %errorlevel% neq 0 (
    echo 安装Flask...
    pip install flask
)

python -c "import flask_cors" >nul 2>&1
if %errorlevel% neq 0 (
    echo 安装Flask-CORS...
    pip install flask-cors
)

echo.
echo 启动增强版基于强化学习的自主现制饮品研发专家系统...
echo 访问地址: http://localhost:5000
echo 按 Ctrl+C 停止服务
echo.

python enhanced_rl_beverage_expert_webui.py

pause