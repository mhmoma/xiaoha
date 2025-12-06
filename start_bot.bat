@echo off
echo 正在安装依赖...
pip install -r requirements.txt
echo 依赖安装完成。
echo 正在启动机器人...
python bot.py
pause
