# AI 이미지 분류기 로컬 실행 가이드

이 가이드는 AI 이미지 분류기 웹 애플리케이션을 로컬 컴퓨터에서 실행하는 방법을 설명합니다.

## 📋 시스템 요구사항

- Python 3.8 이상
- 최소 4GB RAM (8GB 권장)
- 인터넷 연결 (모델 다운로드용)

## 🚀 빠른 시작

### 1단계: 의존성 설치

```bash
pip install -r requirements.txt
```

### 2단계: 웹 애플리케이션 실행

다음 중 하나의 방법을 선택하세요:

#### 방법 1: 간단한 실행 (권장)
```bash
python start_app.py
```

#### 방법 2: 기본 실행
```bash
python app.py
```

#### 방법 3: 완전한 체크와 함께 실행
```bash
python run_website.py
```

### 3단계: 웹사이트 접속

브라우저에서 다음 주소로 접속하세요:
- http://localhost:5000
- 또는 http://127.0.0.1:5000

## 🔧 문제 해결

### 일반적인 문제들

#### 1. 유니코드 인코딩 오류
```
UnicodeEncodeError: 'cp949' codec can't encode character
```
**해결방법**: 이미 수정된 파일들을 사용하세요. 이모지가 제거되어 있습니다.

#### 2. watchdog 오류
```
ImportError: cannot import name 'EVENT_TYPE_OPENED' from 'watchdog.events'
```
**해결방법**: `start_app.py`를 사용하세요. 이 파일은 debug=False로 설정되어 있어 watchdog 오류를 방지합니다.

#### 3. 모델 로드 오류
```
모델 로드 실패: ...
```
**해결방법**: 
- 인터넷 연결을 확인하세요
- 방화벽이 모델 다운로드를 차단하지 않는지 확인하세요
- 충분한 디스크 공간이 있는지 확인하세요 (약 1GB 필요)

#### 4. 포트 사용 중 오류
```
Address already in use
```
**해결방법**: 
- 다른 포트를 사용하거나
- 포트를 사용하는 다른 프로그램을 종료하세요

### 포트 변경하기

다른 포트를 사용하려면 `app.py` 파일의 마지막 부분을 수정하세요:

```python
# 포트 8080으로 변경
port = 8080
app.run(debug=False, host='0.0.0.0', port=port)
```

## 📁 프로젝트 구조

```
ai-image-detector-project-main/
├── app.py                    # 메인 웹 애플리케이션
├── start_app.py             # 간단한 실행 스크립트 (권장)
├── run_website.py           # 완전한 체크와 함께 실행
├── requirements.txt         # 필요한 패키지 목록
├── static/                  # 정적 파일 (CSS, JS, 업로드된 이미지)
│   ├── css/
│   ├── js/
│   └── uploads/
├── templates/               # HTML 템플릿
│   ├── index.html
│   ├── about.html
│   └── stats.html
└── data/                    # 데이터 저장소
    └── feedback/
```

## 🎯 사용 방법

1. **이미지 업로드**: 메인 페이지에서 이미지 파일을 선택하고 업로드하세요
2. **결과 확인**: AI가 이미지를 분석하여 "REAL" 또는 "FAKE"로 분류합니다
3. **피드백 제공**: 결과가 맞는지 피드백을 제공할 수 있습니다
4. **통계 확인**: 통계 페이지에서 전체 성능을 확인할 수 있습니다

## 🔍 지원되는 이미지 형식

- PNG
- JPG/JPEG
- GIF
- BMP
- TIFF

최대 파일 크기: 16MB

## 💡 추가 정보

- **모델**: 사전 훈련된 ViT (Vision Transformer) 모델 사용
- **성능**: CPU에서도 실행 가능하지만, GPU가 있으면 더 빠릅니다
- **데이터**: 업로드된 이미지는 `static/uploads/` 폴더에 저장됩니다
- **피드백**: 사용자 피드백은 `data/feedback/` 폴더에 JSON 형태로 저장됩니다

## 🆘 도움이 필요하신가요?

문제가 지속되면 다음을 확인해보세요:

1. Python 버전이 3.8 이상인지 확인
2. 모든 패키지가 올바르게 설치되었는지 확인
3. 충분한 메모리와 디스크 공간이 있는지 확인
4. 방화벽이나 안티바이러스가 차단하지 않는지 확인

---

**성공적으로 실행되면 브라우저에서 http://localhost:5000 으로 접속하여 AI 이미지 분류기를 사용할 수 있습니다!**
