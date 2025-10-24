# 깃허브 업로드 가이드

## 🚀 깃허브에 프로젝트 올리기

### 1. 깃허브에서 새 저장소 생성
1. [github.com](https://github.com) 접속 및 로그인
2. 우측 상단 "+" 버튼 → "New repository"
3. 저장소 이름: `ai-image-detector` (또는 원하는 이름)
4. 설명: "AI 이미지 분류기 - AI 생성 이미지와 실제 사진을 구분하는 웹 애플리케이션"
5. Public/Private 선택
6. **중요**: "Add a README file", "Add .gitignore", "Choose a license" 체크 해제
7. "Create repository" 클릭

### 2. 터미널에서 다음 명령어 실행

```bash
# 깃허브 저장소와 연결 (your-username을 실제 사용자명으로 변경)
git remote add origin https://github.com/your-username/ai-image-detector.git

# 메인 브랜치로 이름 변경
git branch -M main

# 깃허브에 업로드
git push -u origin main
```

### 3. 예시 (사용자명이 "john"인 경우)

```bash
git remote add origin https://github.com/john/ai-image-detector.git
git branch -M main
git push -u origin main
```

### 4. 인증 정보 입력
- Username: 깃허브 사용자명
- Password: 깃허브 Personal Access Token (비밀번호 아님!)

**Personal Access Token 생성 방법:**
1. 깃허브 → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. "Generate new token" 클릭
3. Note: "AI Image Detector"
4. Expiration: 원하는 기간 선택
5. Scopes: "repo" 체크
6. "Generate token" 클릭
7. 생성된 토큰을 복사하여 비밀번호로 사용

### 5. 업로드 완료 확인
- 깃허브 저장소 페이지에서 파일들이 업로드되었는지 확인
- README.md가 자동으로 표시되는지 확인

## 🔄 이후 업데이트 방법

코드를 수정한 후:

```bash
# 변경사항 추가
git add .

# 커밋
git commit -m "업데이트 내용 설명"

# 깃허브에 푸시
git push
```

## 📋 체크리스트

- [ ] 깃허브 계정 생성
- [ ] 새 저장소 생성
- [ ] 로컬 Git 초기화 완료
- [ ] 첫 커밋 완료
- [ ] 깃허브 저장소와 연결
- [ ] Personal Access Token 생성
- [ ] 첫 업로드 완료

## 🎉 완료!

이제 다른 사람들이 당신의 프로젝트를 볼 수 있고, 클론하여 사용할 수 있습니다!

**프로젝트 URL**: `https://github.com/your-username/ai-image-detector`
