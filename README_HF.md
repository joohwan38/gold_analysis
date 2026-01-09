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

# 💰 WoW 골드 거래 분석기

World of Warcraft의 골드 거래 데이터를 분석하여 최고의 거래를 찾아주는 웹 애플리케이션입니다.

## 🚀 주요 기능

- 📊 **거래 데이터 파싱**: 텍스트 형식의 거래 데이터를 자동으로 분석
- 🏆 **가성비 순위**: 코인당 골드(GPC) 기준으로 거래를 정렬
- 📈 **가격 추이 분석**: 골드 수량별 최고 가성비 시각화
- 👥 **판매자 분석**: 평균 GPC 기준 상위 10명의 판매자 분석
- 💾 **데이터 저장**: 분석한 데이터를 파일로 저장

## 📖 사용 방법

### 1. 데이터 입력
WoW 게임 내 골드 거래소에서 거래 데이터를 복사하여 붙여넣습니다.

**예시 형식:**
```
1000 gold
Long PlayerName Horde 50 coins

2000 gold
Medium PlayerName2 Alliance 80 coins
```

### 2. 데이터 처리
"🔄 데이터 처리" 버튼을 클릭하여 데이터를 분석합니다.

### 3. 결과 확인
"📊 분석 결과" 탭에서 다음을 확인할 수 있습니다:
- 가성비 순위 (코인당 골드 기준)
- 가격 추이 차트
- 판매자 분석 차트

## ⚠️ 중요 안내

이 앱은 **Hugging Face Spaces 무료 플랜**으로 호스팅됩니다.

- **30분 동안 사용하지 않으면 슬립 모드로 전환됩니다**
- 처음 접속 시 또는 슬립 모드 후 재접속 시 **앱 로딩에 30초~1분 정도 소요**될 수 있습니다
- 잠시만 기다려주세요! 🙏

## 🛠️ 기술 스택

- **Python 3.11+**
- **Gradio**: 웹 인터페이스
- **Pandas**: 데이터 처리
- **Matplotlib**: 차트 시각화

## 📊 분석 지표

### GPC (Gold Per Coin)
- 1 코인당 얻을 수 있는 골드 양
- 높을수록 가성비가 좋습니다
- 공식: `GPC = 총 골드 / 가격(코인)`

## 🔧 로컬 실행

로컬 환경에서 실행하려면:

```bash
# 저장소 클론
git clone https://huggingface.co/spaces/YOUR_USERNAME/gold-analyzer
cd gold-analyzer

# 의존성 설치
pip install -r requirements.txt

# 실행
python app.py
```

## 📝 라이선스

MIT License

## 🤝 기여

버그 리포트나 기능 제안은 언제든 환영합니다!

---

Made with ❤️ for WoW players
