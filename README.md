# DiNuri AI — FastAPI 백엔드
> **AI Document Assistant for Seniors**  
> “복잡한 서류, 이제 쉽게 이해하세요.”  

---

![Grand Prize](https://img.shields.io/badge/🏆_Grand_Prize-2025_AI_Hackathon-blue?style=for-the-badge)
![Status](https://img.shields.io/badge/🚧_Coming_Soon-in_progress-orange?style=for-the-badge)

---

## 🏆 수상 이력

**🎖️ 2025 캡스톤 디자인 및 AI 해커톤 경진대회 — 생성형 AI 활용 분야 대상**  
주최: **사단법인 한국컴퓨터교육학회**  
후원: **SK Broadband · Kakao · HANCOM InSpace · AhnLab · NEXV · 영림원소프트랩 · ValiantData 외**

---

## 프로젝트 개요

**디누리(DiNuri)** 는 시니어 사용자를 위해  
공공서류, 금융문서 등 복잡한 문서를 **읽고, 요약하고, 들려주는** AI 기반 서비스입니다.  

**DiNuri AI**는 시니어를 위한 문서 해설 서비스입니다.  
이미지 문서 → 텍스트 추출 → 요약 → 음성 변환을 한 번에 수행합니다.

---

## 주요 기능

| Endpoint | 설명 |
|-----------|------|
| `POST /api/ocr-summarize` | 이미지 → OCR + 요약 + 태그 |
| `POST /api/ocr-tags` | 이미지 → 태그만 추출 |
| `POST /api/tts-summary` | 요약 JSON → MP3 음성 |
| `POST /api/tts` | 텍스트 → MP3 음성 |
| `GET /api/health` | 서버 상태 확인 |

---

## 환경 설정

`.env.example`을 복사해 `.env`로 변경한 뒤,  
다음 항목을 본인 환경에 맞게 수정하세요.

```bash
GOOGLE_APPLICATION_CREDENTIALS=/Users/yourname/Desktop/dinuri-ai/service-account.json
OPENAI_API_KEY=sk-proj-당신의_키
```

---

## 실행 방법 

```bash
# 1. 가상환경 생성
python3 -m venv venv

# 2. 가상환경 활성화
source venv/bin/activate   # macOS / Linux
# venv\Scripts\activate    # Windows

# 3. 패키지 설치
pip install -r requirements.txt

# 4. 서버 실행
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

---

## 테스트
- 서버 상태 확인: http://localhost:8000/api/health

- Swagger 문서: http://localhost:8000/docs

---

## 주요 파일 구조
```bash
DINURI_AI/
├── app.py              # FastAPI 서버 엔트리포인트
├── ai_doc_helper2.py   # OCR + 요약 + TTS 핵심 로직
├── requirements.txt    # 패키지 목록
├── .env.example        # 환경변수 예시 파일
└── README.md
```
