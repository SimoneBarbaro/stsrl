@echo off
cd sts_lightspeed || exit /b
cmake -G Ninja -S%cd% -B%cd%\build || exit /b
cd build || exit /b
ninja || exit /b
move /y slaythespire* ..\..\stsrl || exit /b
cd ..
cd ..
pip install --force-reinstall .