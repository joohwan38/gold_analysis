# 🚀 Hugging Face Spaces 배포 가이드

이 문서는 WoW 골드 분석기를 Hugging Face Spaces에 배포하는 방법을 안내합니다.

## 📋 필요한 파일 체크리스트

배포를 위해 다음 파일들이 준비되었습니다:

- ✅ `app.py` - Hugging Face Spaces용 메인 애플리케이션
- ✅ `requirements.txt` - Python 의존성 목록
- ✅ `README_HF.md` - Hugging Face용 README (메타데이터 포함)
- ✅ `.gitignore` - Git 제외 파일 목록

## 🎯 배포 단계

### 1단계: Hugging Face 계정 생성 및 설정

#### A. 계정 생성
1. [Hugging Face](https://huggingface.co/) 접속
2. "Sign Up" 클릭하여 계정 생성 (무료)
3. 이메일 인증 완료

#### B. 액세스 토큰 생성
1. 로그인 후 우측 상단 프로필 클릭 → Settings
2. 좌측 메뉴에서 "Access Tokens" 선택
3. "New token" 클릭
4. 토큰 이름 입력 (예: "gold-analyzer-deploy")
5. Role: "Write" 선택
6. "Generate a token" 클릭
7. **토큰을 안전한 곳에 복사해두기** (다시 볼 수 없습니다!)

---

### 2단계: Space 생성

#### A. 새 Space 만들기
1. Hugging Face 메인 페이지에서 "Spaces" 탭 클릭
2. "Create new Space" 클릭
3. 다음 정보 입력:
   - **Owner**: 본인 계정 선택
   - **Space name**: `gold-analyzer` (또는 원하는 이름)
   - **License**: MIT
   - **Select the Space SDK**: **Gradio** 선택 ⭐
   - **Space hardware**: CPU basic (무료)
   - **Visibility**: Public 또는 Private 선택

4. "Create Space" 클릭

---

### 3단계: 로컬에서 Git 설정

#### A. Git 저장소 준비

터미널(PowerShell)을 열고 프로젝트 디렉토리로 이동:

```powershell
cd C:\Users\mrson\Documents\gold_analysis
```

#### B. Git 초기화 (아직 안했다면)

```bash
# Git 초기화
git init

# 사용자 정보 설정 (처음 한번만)
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

#### C. Hugging Face CLI 설치 (선택사항 - 더 쉬운 방법)

```bash
pip install huggingface_hub

# 로그인
huggingface-cli login
# 위에서 생성한 토큰 입력
```

---

### 4단계: 파일 준비 및 업로드

#### A. README_HF.md를 README.md로 복사

```bash
# 기존 README 백업
copy README.md README_OLD.md

# Hugging Face용 README로 교체
copy README_HF.md README.md
```

또는 직접 `README_HF.md` 파일의 내용을 `README.md`에 복사-붙여넣기 해주세요.

**중요**: README.md 파일 맨 위에 반드시 다음 메타데이터가 있어야 합니다:

```yaml
---
title: WoW Gold Analyzer
emoji: 💰
colorFrom: yellow
colorTo: orange
sdk: gradio
sdk_version: 5.7.1
app_file: app.py
pinned: false
license: mit
---
```

#### B. Git에 파일 추가

```bash
# 변경사항 확인
git status

# 필요한 파일만 추가
git add app.py
git add requirements.txt
git add README.md
git add .gitignore

# 커밋
git commit -m "Initial commit: WoW Gold Analyzer for Hugging Face Spaces"
```

---

### 5단계: Hugging Face에 푸시

#### 방법 1: HTTPS (추천)

```bash
# Hugging Face Space를 원격 저장소로 추가
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/gold-analyzer

# 푸시 (토큰을 비밀번호로 사용)
git push space main
```

푸시 시 사용자명과 비밀번호를 요청하면:
- **Username**: Hugging Face 사용자명
- **Password**: 위에서 생성한 액세스 토큰 (비밀번호 아님!)

#### 방법 2: SSH (고급 사용자)

```bash
# SSH 키 설정 후
git remote add space git@hf.co:spaces/YOUR_USERNAME/gold-analyzer
git push space main
```

---

### 6단계: 배포 확인

1. Hugging Face Space 페이지로 이동: `https://huggingface.co/spaces/YOUR_USERNAME/gold-analyzer`
2. "Files" 탭에서 파일들이 업로드되었는지 확인
3. "App" 탭에서 앱이 빌드되고 실행되는지 확인
   - 처음 빌드는 1~2분 소요됩니다
   - 로그를 확인하여 에러가 없는지 체크
4. 빌드가 완료되면 앱이 자동으로 실행됩니다!

---

## 🔄 업데이트 방법

나중에 코드를 수정하고 다시 배포하려면:

```bash
# 파일 수정 후
git add .
git commit -m "Update: 변경 내용 설명"
git push space main
```

Hugging Face가 자동으로 변경사항을 감지하고 재배포합니다.

---

## ⚙️ Space 설정 (선택사항)

Space 페이지의 "Settings" 탭에서 추가 설정 가능:

### A. 하드웨어 업그레이드 (유료)
- CPU basic (무료) → CPU upgrade ($0.03/시간)
- T4 GPU ($0.60/시간) - 머신러닝 앱용

### B. 슬립 모드 설정
- "Sleep time" 기본값: 30분
- 유료 플랜에서는 조정 가능

### C. 환경 변수 설정
- "Repository secrets"에서 비밀 정보 저장
- 예: API 키, 비밀번호 등

### D. 커스텀 도메인 (Pro 플랜)
- 본인 도메인 연결 가능

---

## 🐛 문제 해결

### 1. 빌드 실패

**증상**: "Build failed" 에러 표시

**해결**:
1. "Files" → "Logs" 탭에서 에러 로그 확인
2. 주로 `requirements.txt` 문제:
   ```bash
   # 버전 충돌 시 버전 번호 제거
   gradio
   pandas
   matplotlib
   ```
3. 파일 수정 후 다시 푸시

### 2. 앱이 시작되지 않음

**증상**: 빌드는 성공했지만 앱이 실행 안됨

**해결**:
1. `app.py` 마지막 줄 확인:
   ```python
   if __name__ == "__main__":
       app.launch()  # 이렇게만 있어야 함
   ```
2. README.md의 `app_file: app.py` 확인

### 3. 한글이 깨짐

**증상**: 차트나 텍스트에서 한글이 □로 표시

**해결**:
- Matplotlib 폰트 설정이 이미 되어 있습니다
- Hugging Face 서버에 한글 폰트가 없어서 발생할 수 있음
- 차트 텍스트를 영어로 변경하거나 폰트 파일을 포함시켜야 함

### 4. 슬립 모드에서 깨어나지 않음

**증상**: 30분 후 접속 시 "Building..." 상태에 멈춤

**해결**:
- 페이지 새로고침 (Ctrl+F5)
- 1~2분 기다리기
- 계속 안되면 Space 재시작: Settings → Restart this Space

---

## 📊 모니터링

### 사용량 확인
1. Space 페이지 → "Settings" 탭
2. "Usage and Billing" 섹션
3. CPU/메모리 사용량, 트래픽 확인

### 로그 확인
1. Space 페이지 → "Files" 탭
2. "Logs" 버튼 클릭
3. 실시간 로그 스트리밍 확인

---

## 🔒 보안 고려사항

### 1. 비공개 설정
Space를 Private으로 설정하면:
- 본인만 접근 가능
- 무료 플랜에서도 사용 가능
- 공유하려면 특정 사용자 초대

### 2. 인증 추가 (선택)
`app.py`에 인증 추가:

```python
app.launch(
    auth=("username", "password")  # 기본 인증
)
```

### 3. Rate Limiting
많은 사용자가 접속할 경우 Cloudflare와 연동하여 DDoS 방지 가능 (고급)

---

## 🎉 완료!

축하합니다! 이제 당신의 WoW 골드 분석기가 전 세계에서 접속 가능합니다!

**Space URL**: `https://YOUR_USERNAME-gold-analyzer.hf.space`

이 URL을 친구들과 공유하세요! 🚀

---

## 📚 추가 자료

- [Gradio Spaces 공식 가이드](https://huggingface.co/docs/hub/spaces-sdks-gradio)
- [Hugging Face Hub 문서](https://huggingface.co/docs/hub/)
- [Gradio 공식 문서](https://gradio.app/docs/)

---

## 💡 팁

### 1. 예시 데이터 추가
`app.py`에 "예시 로드" 버튼을 추가하면 사용자가 쉽게 테스트 가능:

```python
def load_example():
    return """1000 gold
Long TestPlayer Horde 50 coins

2000 gold
Medium TestPlayer2 Alliance 80 coins"""

# 버튼 추가
example_btn = gr.Button("📝 예시 데이터 로드")
example_btn.click(fn=load_example, outputs=[text_input])
```

### 2. Gradio Analytics
Space Settings에서 Analytics를 활성화하면 방문자 통계 확인 가능

### 3. 배지 추가
GitHub README에 Hugging Face 배지 추가:

```markdown
[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/YOUR_USERNAME/gold-analyzer)
```

---

**문서 버전**: 1.0
**작성일**: 2026-01-09
**다음 업데이트**: 배포 완료 후
