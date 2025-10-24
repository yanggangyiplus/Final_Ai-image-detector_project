# -*- coding: utf-8 -*-
"""
AI 이미지 분류기 웹사이트 간단 실행 스크립트
watchdog 오류를 피하기 위해 debug=False로 실행
"""

import os
import sys
from pathlib import Path

def main():
    """메인 실행 함수"""
    print("AI 이미지 분류기 웹사이트 시작 중...")
    print("=" * 60)
    
    # 필요한 디렉토리 생성
    directories = [
        'static/uploads',
        'static/results', 
        'data/feedback',
        'templates'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"OK {directory} 디렉토리 생성됨")
    
    print("\n웹 애플리케이션 시작 중...")
    print("=" * 60)
    print("브라우저에서 http://localhost:5000 으로 접속하세요!")
    print("종료하려면 Ctrl+C를 누르세요")
    print("=" * 60)
    
    try:
        # Flask 앱 실행 (debug=False로 설정하여 watchdog 오류 방지)
        from app import app
        app.run(debug=False, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n\n웹사이트가 종료되었습니다.")
    except Exception as e:
        print(f"\nERROR 오류 발생: {e}")
        print("문제 해결 방법:")
        print("1. 모든 패키지가 설치되어 있는지 확인하세요")
        print("2. 포트 5000이 사용 중이 아닌지 확인하세요")

if __name__ == '__main__':
    main()
