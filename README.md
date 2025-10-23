# AI 이미지 분류기 (AI Image Detector)

AI가 생성한 이미지와 실제 사진을 구분하는 웹 애플리케이션입니다.

## 🎯 주요 기능

- **이미지 분류**: 업로드된 이미지가 AI 생성 이미지인지 실제 사진인지 판별
- **실시간 분석**: ViT (Vision Transformer) 모델을 사용한 고정밀 분석
- **사용자 피드백**: 예측 결과에 대한 피드백 수집으로 모델 개선
- **통계 모니터링**: 전체 성능 및 사용 통계 확인
- **반응형 웹 UI**: 직관적이고 사용하기 쉬운 인터페이스

## 🚀 빠른 시작

### 1. 저장소 클론
```bash
git clone https://github.com/your-username/ai-image-detector.git
cd ai-image-detector
```

### 2. 의존성 설치
```bash
pip install -r requirements.txt
```

### 3. 웹 애플리케이션 실행
```bash
python run_simple.py
```

### 4. 브라우저에서 접속
http://localhost:5000

## 📋 시스템 요구사항

- Python 3.8 이상
- 최소 4GB RAM (8GB 권장)
- 인터넷 연결 (모델 다운로드용)

## 🔧 기술 스택

- **Backend**: Flask (Python)
- **AI Model**: Vision Transformer (ViT)
- **Frontend**: HTML, CSS, JavaScript
- **ML Libraries**: PyTorch, Transformers, PIL

## 📁 프로젝트 구조

```
ai-image-detector/
├── app.py                    # 메인 웹 애플리케이션
├── run_simple.py            # 간단한 실행 스크립트
├── requirements.txt         # 필요한 패키지 목록
├── .gitignore              # Git 무시 파일 목록
├── static/                 # 정적 파일 (CSS, JS, 업로드된 이미지)
│   ├── css/
│   ├── js/
│   └── uploads/
├── templates/              # HTML 템플릿
│   ├── index.html
│   ├── about.html
│   └── stats.html
└── data/                   # 데이터 저장소
    └── feedback/
```

## 🎨 사용 방법

1. **이미지 업로드**: 메인 페이지에서 이미지 파일을 선택하고 업로드
2. **결과 확인**: AI가 이미지를 분석하여 "REAL" 또는 "FAKE"로 분류
3. **피드백 제공**: 결과가 맞는지 피드백을 제공하여 모델 개선에 기여
4. **통계 확인**: 통계 페이지에서 전체 성능을 확인

## 🔍 지원되는 이미지 형식

- PNG, JPG/JPEG, GIF, BMP, TIFF
- 최대 파일 크기: 16MB

## ⚡ 주요 개선사항

- **이미지 전처리**: 모든 이미지를 32×32로 리사이즈하여 모델 훈련 조건과 일치
- **유니코드 문제 해결**: 한글 출력 문제 완전 해결
- **사전 훈련된 모델 사용**: 로컬 훈련 없이도 즉시 사용 가능
- **사용자 친화적 UI**: 직관적인 웹 인터페이스

## 🛠️ 문제 해결

자세한 문제 해결 방법은 [LOCAL_SETUP_GUIDE.md](LOCAL_SETUP_GUIDE.md)를 참고하세요.

## 📊 모델 성능

- **모델**: 사전 훈련된 ViT (Vision Transformer)
- **정확도**: 높은 정확도로 AI 생성 이미지와 실제 이미지 구분
- **처리 속도**: CPU에서도 빠른 분석 가능

## 🤝 기여하기

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참고하세요.

## 📞 연락처

프로젝트 링크: [https://github.com/your-username/ai-image-detector](https://github.com/your-username/ai-image-detector)

## 🙏 감사의 말

- [Hugging Face](https://huggingface.co/) - 사전 훈련된 모델 제공
- [PyTorch](https://pytorch.org/) - 딥러닝 프레임워크
- [Flask](https://flask.palletsprojects.com/) - 웹 프레임워크

---

**⭐ 이 프로젝트가 도움이 되었다면 Star를 눌러주세요!**