# -*- coding: utf-8 -*-
"""
AI 이미지 분류기 웹사이트 - 포트 8080으로 실행
"""

import os
import sys
from pathlib import Path

# 필요한 디렉토리 생성
directories = [
    'static/uploads',
    'static/results', 
    'data/feedback',
    'templates'
]

for directory in directories:
    Path(directory).mkdir(parents=True, exist_ok=True)
    print(f"OK {directory} directory created")

print("\nStarting AI Image Classifier Web Application...")
print("=" * 60)
print("Please open your browser and go to: http://localhost:8080")
print("Press Ctrl+C to stop the server")
print("=" * 60)

try:
    # Flask 앱 실행 (포트 8080)
    from app import app
    app.run(debug=False, host='127.0.0.1', port=8080)
except KeyboardInterrupt:
    print("\n\nWeb application stopped.")
except Exception as e:
    print(f"\nERROR: {e}")
    print("Troubleshooting:")
    print("1. Check if all packages are installed")
    print("2. Check if port 8080 is available")
