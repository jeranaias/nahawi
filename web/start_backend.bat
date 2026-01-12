@echo off
cd /d c:\nahawi\web\backend
C:\Users\Jesse\AppData\Local\Programs\Python\Python312\python.exe -m uvicorn main:app --reload --port 8000
pause
