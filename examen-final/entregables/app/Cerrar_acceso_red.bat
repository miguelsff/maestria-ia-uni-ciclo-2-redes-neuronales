@echo off
REM Quita la regla del firewall que abrio Iniciar_demo_red_local.bat.
net session >nul 2>&1
if %errorlevel% neq 0 (
    powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
    exit /b
)
netsh advfirewall firewall delete rule name="PeruFoodNet demo"
echo Listo: el puerto ya no acepta conexiones de otros equipos.
pause
