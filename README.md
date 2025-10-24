# AI 이미지 검출기 (AI Image Detector) 🤖

AI가 생성한 이미지와 실제 사진을 구분하는 딥러닝 기반 웹 애플리케이션입니다.

Vision Transformer (ViT) 모델을 사용하여 높은 정확도로 AI 생성 이미지를 탐지합니다.

## 🎯 주요 기능

- **이미지 분류**: 업로드된 이미지가 AI 생성 이미지인지 실제 사진인지 판별
- **실시간 분석**: ViT (Vision Transformer) 모델을 사용한 고정밀 분석
- **시각화**: 원본 이미지와 전처리된 이미지 비교 시각화
- **사용자 피드백**: 예측 결과에 대한 피드백 수집으로 모델 개선
- **통계 모니터링**: 실시간 성능 및 사용 통계 대시보드
- **반응형 웹 UI**: 직관적이고 사용하기 쉬운 인터페이스

## 🚀 빠른 시작

### 1. 저장소 클론
```bash
git clone https://github.com/yanggangyiplus/Final_Ai-image-detector_project.git
cd Final_Ai-image-detector_project
```

### 2. 의존성 설치
```bash
pip install -r requirements.txt
```

### 3. 웹 애플리케이션 실행

**일반 실행 (권장):**
```bash
python3 run_port_8080.py
```

**또는 간단한 실행:**
```bash
python3 run_simple.py
```

> **macOS 사용자 참고**: macOS에서는 AirPlay가 기본적으로 포트 5000을 사용합니다. `run_port_8080.py`를 사용하여 포트 8080에서 실행하는 것을 권장합니다.

### 4. 브라우저에서 접속
- **포트 8080**: http://localhost:8080
- **포트 5000**: http://localhost:5000 (macOS에서는 AirPlay 비활성화 필요)

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
Final_Ai-image-detector_project/
├── app.py                           # 메인 Flask 웹 애플리케이션
├── ai_image_detector_model_vit.py   # ViT 모델 학습 코드
├── model_evaluation.py              # 모델 평가 스크립트
├── model_retrain.py                 # 모델 재학습 기능
├── quick_evaluation.py              # 빠른 모델 평가
│
├── run_port_8080.py                 # 포트 8080 실행 스크립트 (권장)
├── run_simple.py                    # 간단한 실행 스크립트
├── run_website.py                   # 전체 기능 실행 스크립트
├── start_app.py                     # 앱 시작 스크립트
│
├── requirements.txt                 # Python 패키지 의존성
├── .gitignore                       # Git 무시 파일 목록
├── LICENSE                          # MIT 라이선스
├── README.md                        # 프로젝트 설명서
│
├── static/                          # 정적 파일
│   ├── css/style.css               # 스타일시트
│   ├── js/main.js                  # JavaScript 파일
│   ├── uploads/                    # 업로드된 이미지 (gitignore)
│   └── results/                    # 분석 결과 이미지 (gitignore)
│
├── templates/                       # HTML 템플릿
│   ├── index.html                  # 메인 페이지
│   ├── about.html                  # 소개 페이지
│   ├── stats.html                  # 통계 페이지
│   └── base.html                   # 기본 템플릿
│
├── data/                            # 데이터 저장소
│   └── feedback/                   # 사용자 피드백 (gitignore)
│
├── test image/                      # 테스트용 샘플 이미지
│   ├── fake/                       # AI 생성 이미지 샘플
│   └── real/                       # 실제 사진 샘플
│
└── logs/                            # 로그 파일 (gitignore)
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

## 🧪 모델 학습 및 평가

### 테스트 데이터 준비
모델을 학습하고 평가하려면 테스트 이미지 데이터셋이 필요합니다.

프로젝트 루트에 `test image/` 폴더를 만들고 다음과 같이 구성하세요:
```
test image/
├── fake/    # AI 생성 이미지
└── real/    # 실제 사진
```

**권장 데이터셋:**
- [CIFAKE Dataset](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)
- 또는 직접 수집한 이미지 사용

### 모델 학습
```bash
python3 ai_image_detector_model_vit.py
```

### 모델 평가
```bash
# 전체 평가
python3 model_evaluation.py

# 빠른 평가
python3 quick_evaluation.py
```

## 🔧 추가 기능

### 피드백 기반 재학습
사용자 피드백을 수집하여 모델을 지속적으로 개선할 수 있습니다.

### 통계 대시보드
`/stats` 페이지에서 실시간으로 모델 성능과 사용 통계를 확인할 수 있습니다.

## 📝 문서

- [LOCAL_SETUP_GUIDE.md](LOCAL_SETUP_GUIDE.md) - 로컬 환경 설정 가이드
- [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) - 배포 가이드
- [evaluation_guide.md](evaluation_guide.md) - 모델 평가 가이드

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참고하세요.

## 📞 연락처

프로젝트 링크: [https://github.com/yanggangyiplus/Final_Ai-image-detector_project](https://github.com/yanggangyiplus/Final_Ai-image-detector_project)

## 🙏 감사의 말

- [Hugging Face](https://huggingface.co/) - 사전 훈련된 ViT 모델 제공
- [PyTorch](https://pytorch.org/) - 딥러닝 프레임워크
- [Flask](https://flask.palletsprojects.com/) - 웹 프레임워크
- [Transformers](https://huggingface.co/docs/transformers/) - Vision Transformer 모델

## 🎓 프로젝트 정보

이 프로젝트는 AI 이미지 생성 기술의 발전과 함께 증가하는 딥페이크 및 AI 생성 이미지를 탐지하기 위해 개발되었습니다.

**주요 특징:**
- Vision Transformer (ViT) 기반 분류
- 실시간 웹 인터페이스
- 사용자 피드백 시스템
- 지속적인 모델 개선

---

**⭐ 이 프로젝트가 도움이 되었다면 Star를 눌러주세요!**