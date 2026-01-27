# ai_doc_helper2.py
# Google Cloud Vision OCR + OpenAI GPT-4o-mini + gTTS

from dotenv import load_dotenv
import io
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image, ImageDraw
from google.cloud import vision
from google.oauth2 import service_account
from google.api_core.client_options import ClientOptions

# OpenAI SDK version check
try:
    from openai import OpenAI
    _OPENAI_SDK_V1 = True
except Exception:
    import openai
    _OPENAI_SDK_V1 = False

from gtts import gTTS

# DNS resolver
os.environ.setdefault("GRPC_DNS_RESOLVER", "native")

# env
load_dotenv()
DEFAULT_GCP_KEY = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
DEFAULT_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def build_vision_client(key_path: Optional[str] = None) -> vision.ImageAnnotatorClient:
    if key_path is None:
        key_path = DEFAULT_GCP_KEY
    creds = service_account.Credentials.from_service_account_file(str(Path(key_path)))
    return vision.ImageAnnotatorClient(
        credentials=creds,
        client_options=ClientOptions(api_endpoint="https://vision.googleapis.com"),
        transport="rest",
    )


def build_openai_client(api_key: Optional[str] = None):
    if api_key is None:
        api_key = DEFAULT_OPENAI_API_KEY
    if _OPENAI_SDK_V1:
        return OpenAI(api_key=api_key)
    else:
        openai.api_key = api_key
        return openai


# OCR
def pil_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def gcv_ocr(image_bytes: bytes, client_v: vision.ImageAnnotatorClient) -> Tuple[str, List[List[Tuple[int, int]]]]:
    image = vision.Image(content=image_bytes)
    ctx = vision.ImageContext(language_hints=["ko"])
    resp = client_v.document_text_detection(image=image, image_context=ctx)
    fta = resp.full_text_annotation
    full_text = fta.text if fta else ""
    boxes: List[List[Tuple[int, int]]] = []
    if fta:
        for page in fta.pages:
            for block in page.blocks:
                boxes.append([(v.x, v.y) for v in block.bounding_box.vertices])
    return full_text, boxes


def draw_boxes(pil_img: Image.Image, boxes: List[List[Tuple[int, int]]], stroke: int = 3) -> Image.Image:
    out = pil_img.convert("RGB").copy()
    d = ImageDraw.Draw(out)
    for b in boxes:
        if b:
            d.line(b + [b[0]], width=stroke, fill=(255, 0, 0))
    return out


# tag extraction by regex
TAG_REGEX = {
    "날짜": [
        r"\b20\d{2}[./-]\d{1,2}[./-]\d{1,2}\b",
        r"\b\d{1,2}\s*월\s*\d{1,2}\s*일\b",
        r"\b\d{4}\s*년\s*\d{1,2}\s*월\s*\d{1,2}\s*일\b"
    ],
    "금액": [
        r"\b[0-9]{1,3}(?:,[0-9]{3})*(?:\s*원)\b",
        r"\b\d+\s*원\b"
    ],
    "전화번호": [
        r"\b0\d{1,2}-\d{3,4}-\d{4}\b",
        r"\b0\d{9,10}\b"
    ],
    "계좌": [
        r"\b\d{2,3}-\d{2,6}-\d{2,6}-?\d{0,6}\b",
        r"\b계좌(?:번호)?\s*[:：]?\s*\d[\d-]{6,}\b"
    ],
    "기간/마감": [
        r"(접수|신청|납부|마감|기한|유효)[:：]?\s*[~\-]?\s*\d{0,4}[./-]?\d{1,2}[./-]?\d{1,2}",
        r"\b(~|부터|까지)\b"
    ],
    "주소/URL": [
        r"https?://[^\s]+",
        r"\bwww\.[^\s]+"
    ],
}
# 개인정보 관련 태깅을 따로 하였습니다. => AI Module 연결 예정
TAG_KEYWORDS = {
    "기관/부서": ["국민건강보험", "국세청", "행정복지센터", "시청", "구청", "병원", "의원", "은행", "보험사", "카드사", "고용센터", "연금공단"],
    "신분/개인식별": ["성명", "이름", "생년월일", "주민등록번호", "주소", "연락처"],
}


def _compile_patterns():
    return {k: [re.compile(p) for p in ps] for k, ps in TAG_REGEX.items()}


_COMPILED = _compile_patterns()


def extract_tags(text: str) -> Dict[str, List[str]]:
    found: Dict[str, List[str]] = {k: [] for k in list(TAG_REGEX.keys()) + list(TAG_KEYWORDS.keys())}
    for tag, patterns in _COMPILED.items():
        for pat in patterns:
            for m in pat.findall(text or ""):
                s = m if isinstance(m, str) else (m[0] if m else "")
                if s and s not in found[tag]:
                    found[tag].append(s)
    for tag, kws in TAG_KEYWORDS.items():
        for k in kws:
            if (text or "").find(k) != -1 and k not in found[tag]:
                found[tag].append(k)
    return {k: v for k, v in found.items() if v}


# 문서 종류 분류해뒀는데, version 1.0.0에서는 사용 안 하고 있습니다.
DOC_TYPES = {
    "정부·공공": ["지원금", "세금", "건강보험", "환급"],
    "의료": ["진단서", "혈압", "혈당", "검사", "병원"],
    "금융": ["대출", "보험", "약관", "이자", "수수료"],
    "법률": ["계약서", "동의서", "조항"],
    "일상": ["고지서", "공고", "관리비"],
}


def guess_doc_type(text: str) -> str:
    text_l = (text or "").lower()
    best, score = "일상", 0
    for dtype, kws in DOC_TYPES.items():
        s = sum(1 for k in kws if k.lower() in text_l)
        if s > score:
            best, score = dtype, s
    return best


# 할루시네이션 방지
def is_low_info(text: str, tags: Dict[str, List[str]]) -> bool:
    # 길이가 매우 짧거나, 숫자/한글이 거의 없거나, 태그가 비어 있으면 정보부족으로 판단
    t = (text or "").strip()
    if len(t) < 25:
        return True
    # 의미있는 한글/숫자 토큰 개수 기준
    tokens = re.findall(r"[가-힣A-Za-z0-9]{2,}", t)
    return len(tokens) < 10 and len(tags) == 0


# -------------------------------------------------------------------
# Schedule extraction (LEGACY: title/date_range/time_range)
# -------------------------------------------------------------------
TIME_PAT = re.compile(r"(오전|오후)?\s*\d{1,2}[:시]\s*\d{0,2}\s*(분)?\s*(~|-|부터)\s*(오전|오후)?\s*\d{1,2}[:시]\s*\d{0,2}\s*(분)?")
TIME_SIMPLE = re.compile(r"(오전|오후)?\s*\d{1,2}\s*(~|-)\s*(오전|오후)?\s*\d{1,2}\s*")

# ✅ PATCH: 날짜 패턴 확장(붙여쓰기/마침표 포함)
DATE_PATS = [
    # 2025-12-15 / 2025.12.15 / 2025/12/15 / 2025.12.15.
    re.compile(r"20\d{2}[./-]\d{1,2}[./-]\d{1,2}\.?"),

    # 2025년12월15일 / 2025년 12월 15일
    re.compile(r"20\d{2}\s*년\s*\d{1,2}\s*월\s*\d{1,2}\s*일"),

    # 12월15일 / 12월 15일
    re.compile(r"\d{1,2}\s*월\s*\d{1,2}\s*일"),
]


def extract_schedule(text: str) -> Dict[str, Optional[str]]:
    """
    (LEGACY) 팀 요청: 제목/날짜범위/시간범위
    """
    title = None
    date_hits: List[str] = []
    time_hits: List[str] = []
    # 날짜
    for p in DATE_PATS:
        date_hits += p.findall(text or "")
    # 시간
    time_hits += TIME_PAT.findall(text or "")
    if not time_hits:
        time_hits += TIME_SIMPLE.findall(text or "")
    # 타이틀 후보: 회의/세미나/행사/일정/미팅/발표 등 포함된 첫 줄
    for line in (text or "").splitlines():
        if any(k in line for k in ["회의", "세미나", "행사", "일정", "미팅", "발표", "시연", "면접", "수업", "강의", "오리엔테이션"]):
            title = line.strip()
            break
    # 정리
    date_range = " ~ ".join(date_hits[:2]) if date_hits else ""
    # 시간 정규식 튜플을 문자열로 정리
    def _norm_time(hit):
        if isinstance(hit, tuple):
            return "".join([h for h in hit if isinstance(h, str)])
        return hit
    time_range = " ~ ".join([_norm_time(h) for h in time_hits[:2]]) if time_hits else ""
    return {
        "title": (title or "")[:120],
        "date_range": date_range,
        "time_range": time_range,
    }


# -------------------------------------------------------------------
# (26-01-27) NEW: ISO 8601 규격 schedule_suggestion + LLM title
# + (PATCH) 기간/등록/마감 키워드 기반 범위 우선 추출
# + (PATCH) 날짜 없으면 schedule_suggestion을 "빈 일정"으로 반환
# + (PATCH) 캘린더 제목 프롬프트: 문서 내용을 객관적으로 분석하도록 개선
# -------------------------------------------------------------------
from datetime import datetime

KST_TZ = "Asia/Seoul"

ISO_TIME_PAT = re.compile(
    r"(오전|오후)?\s*(\d{1,2})(?:[:시]\s*(\d{1,2}))?\s*(?:분)?\s*(~|-|부터|까지)\s*"
    r"(오전|오후)?\s*(\d{1,2})(?:[:시]\s*(\d{1,2}))?\s*(?:분)?"
)
ISO_TIME_SIMPLE = re.compile(
    r"(오전|오후)?\s*(\d{1,2})\s*(~|-)\s*(오전|오후)?\s*(\d{1,2})\s*"
)


def _pad2(n: int) -> str:
    return f"{n:02d}"


def _to_iso_date(y: int, m: int, d: int) -> str:
    return f"{y}-{_pad2(m)}-{_pad2(d)}"


def _parse_date_token(s: str, default_year: int) -> Optional[str]:
    s = (s or "").strip()

    # ✅ PATCH: 끝에 '.' 같은 문장부호 제거
    s = re.sub(r"[^\w가-힣./-]+$", "", s)

    # 2026-01-27 / 2026.01.27 / 2026/01/27 / 2026.01.27.
    m1 = re.match(r"^(20\d{2})[./-](\d{1,2})[./-](\d{1,2})\.?$", s)
    if m1:
        y, mo, da = map(int, m1.groups())
        return _to_iso_date(y, mo, da)

    # 2026년1월27일 / 2026년 1월 27일
    m2 = re.match(r"^(20\d{2})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일$", s)
    if m2:
        y, mo, da = map(int, m2.groups())
        return _to_iso_date(y, mo, da)

    # 1월27일 / 1월 27일 (연도 없음 → default_year)
    m3 = re.match(r"^(\d{1,2})\s*월\s*(\d{1,2})\s*일$", s)
    if m3:
        mo, da = map(int, m3.groups())
        return _to_iso_date(default_year, mo, da)

    return None


def _parse_ko_time(ampm: Optional[str], hour: int, minute: int) -> Tuple[int, int]:
    h = hour
    if ampm == "오전":
        if h == 12:
            h = 0
    elif ampm == "오후":
        if h != 12:
            h += 12
    return h, minute


def _to_hhmm(h: int, m: int) -> str:
    return f"{_pad2(h)}:{_pad2(m)}"


def _fallback_title_rule(text: str) -> str:
    title = ""
    for line in (text or "").splitlines():
        if any(k in line for k in ["회의", "세미나", "행사", "일정", "미팅", "발표", "시연", "면접", "수업", "강의", "오리엔테이션"]):
            title = line.strip()[:120]
            break
    return title


def llm_schedule_title(doc_text: str, client) -> str:
    """
    캘린더 제목용: 짧고 자연스럽게
    - 문서 내용을 "객관적으로 분석"해서 일정의 성격을 판별 후 제목 생성
    - 날짜/시간/요약문장/조항문구/긴 문장 방지
    - 애매하면 빈 문자열
    - JSON only
    """
    prompt = f"""
역할: 당신은 문서(OCR 텍스트)를 객관적으로 분석하여, 캘린더에 추가할 "일정 제목"만 생성하는 도우미다.

목표:
- 문서의 핵심이 "어떤 일정 이벤트인지"를 객관적으로 파악한 뒤, 캘린더 제목 1개를 생성한다.

규칙:
- 5~20자 내외의 짧은 제목
- 날짜/시간/요일/장소/번호/금액/괄호 설명/긴 문장을 포함하지 않는다
- 문서에 실제로 존재하는 이벤트(예: 등록, 신청, 접수, 제출, 납부, 마감, 발표, 안내 등)만 제목으로 만든다
- 문장 조각(조항/안내문 문장)을 그대로 제목으로 쓰지 않는다
- 일정 근거가 불명확하면 빈 문자열

출력은 반드시 JSON만:
{{"title": "..."}} 또는 {{"title": ""}}

OCR 텍스트:
\"\"\"{(doc_text or "")[:1500]}\"\"\"
""".strip()

    if _OPENAI_SDK_V1:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[
                {"role": "system", "content": "JSON만 출력하세요."},
                {"role": "user", "content": prompt},
            ],
        )
        content = resp.choices[0].message.content
    else:
        resp = client.ChatCompletion.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[
                {"role": "system", "content": "JSON만 출력하세요."},
                {"role": "user", "content": prompt},
            ],
        )
        content = resp["choices"][0]["message"]["content"]

    content = (content or "").strip().strip("`")
    if content.lower().startswith("json"):
        content = content[4:].strip()

    data = json.loads(content)
    title = (data.get("title") or "").strip()

    # ✅ 방어: 너무 길거나 문장부호 많으면 무효 처리
    if len(title) > 40:
        return ""
    if sum(1 for ch in title if ch in [",", ".", "，", "。", ":", ";"]) >= 2:
        return ""
    return title[:40]


# -----------------------------
# (PATCH) 기간 범위 우선 추출
# -----------------------------
RANGE_KEYWORDS = ["등록", "문서 등록", "등록기간", "등록 기간", "신청", "접수", "제출", "납부", "마감", "기한", "기간"]

# 예: 2025년 12월 15일 09:00부터 12월 17일 14:00까지
KO_DATE_TIME_RANGE_PAT = re.compile(
    r"(?:\b(20\d{2})\s*년\s*)?(\d{1,2})\s*월\s*(\d{1,2})\s*일\s*"
    r"(오전|오후)?\s*(\d{1,2})\s*(?:[:시]\s*(\d{1,2}))?\s*(?:분)?\s*"
    r"(?:부터|~|-)\s*"
    r"(?:\b(20\d{2})\s*년\s*)?(\d{1,2})\s*월\s*(\d{1,2})\s*일\s*"
    r"(오전|오후)?\s*(\d{1,2})\s*(?:[:시]\s*(\d{1,2}))?\s*(?:분)?\s*"
    r"(?:까지)?"
)

# 예: 2025.12.15 09:00 ~ 2025.12.17 14:00 (또는 2025-12-15 09:00 ~ 2025-12-17 14:00)
NUM_DATE_TIME_RANGE_PAT = re.compile(
    r"\b(20\d{2})[./-](\d{1,2})[./-](\d{1,2})\s+(\d{1,2}):(\d{2})\s*(?:~|-)\s*"
    r"(20\d{2})[./-](\d{1,2})[./-](\d{1,2})\s+(\d{1,2}):(\d{2})\b"
)


def _score_range_context(text: str, span_start: int, span_end: int) -> int:
    win = (text or "")[max(0, span_start - 60): min(len(text or ""), span_end + 60)]
    score = 0
    for kw in RANGE_KEYWORDS:
        if kw in win:
            score += 1
    return score


def _pick_best_match(text: str, matches: List[re.Match]) -> Optional[re.Match]:
    if not matches:
        return None
    best = matches[0]
    best_score = _score_range_context(text, best.start(), best.end())
    for m in matches[1:]:
        sc = _score_range_context(text, m.start(), m.end())
        if sc > best_score:
            best = m
            best_score = sc
    return best


def _parse_ko_date_time_range_match(m: re.Match, default_year: int):
    y1, mo1, d1, ampm1, h1, mi1, y2, mo2, d2, ampm2, h2, mi2 = m.groups()
    y1 = int(y1) if y1 else default_year
    y2 = int(y2) if y2 else y1

    mo1, d1 = int(mo1), int(d1)
    mo2, d2 = int(mo2), int(d2)

    h1 = int(h1)
    h2 = int(h2)
    mi1 = int(mi1) if mi1 is not None else 0
    mi2 = int(mi2) if mi2 is not None else 0

    H1, M1 = _parse_ko_time(ampm1, h1, mi1)
    H2, M2 = _parse_ko_time(ampm2, h2, mi2)

    start_date = _to_iso_date(y1, mo1, d1)
    end_date = _to_iso_date(y2, mo2, d2)
    start_time = _to_hhmm(H1, M1)
    end_time = _to_hhmm(H2, M2)
    return start_date, end_date, start_time, end_time


def _parse_num_date_time_range_match(m: re.Match):
    y1, mo1, d1, h1, mi1, y2, mo2, d2, h2, mi2 = m.groups()
    start_date = _to_iso_date(int(y1), int(mo1), int(d1))
    end_date = _to_iso_date(int(y2), int(mo2), int(d2))
    start_time = _to_hhmm(int(h1), int(mi1))
    end_time = _to_hhmm(int(h2), int(mi2))
    return start_date, end_date, start_time, end_time


def extract_schedule_iso(text: str, client=None) -> Dict[str, Optional[object]]:
    """
    ISO 8601 스펙:
    {
      title: str,
      start_date: "YYYY-MM-DD" | None,
      end_date: "YYYY-MM-DD" | None,
      start_time: "HH:MM" | None,
      end_time: "HH:MM" | None,
      timezone: "Asia/Seoul",
      all_day: bool
    }

    정책:
    - '등록/신청/접수/마감/기간/기한' 키워드가 붙은 날짜+시간 "범위"를 최우선으로 캘린더 일정으로 선택
    - 범위가 없으면 기존 방식(첫 날짜/시간) fallback
    - 날짜만 있고 시간이 없으면 all_day=True
    - 날짜가 하나도 없으면 일정이 아니라고 보고, title 포함 전부 빈 값으로 반환(프론트 혼란 방지)
    - title은 LLM로 생성(캘린더용), 실패/빈값이면 규칙 기반 fallback
    """
    now_year = datetime.now().year

    # 1) 기간(범위) 우선 파싱
    start_date = end_date = start_time = end_time = None

    ko_matches = list(KO_DATE_TIME_RANGE_PAT.finditer(text or ""))
    num_matches = list(NUM_DATE_TIME_RANGE_PAT.finditer(text or ""))

    picked = None
    picked_kind = None

    if ko_matches:
        picked = _pick_best_match(text, ko_matches)
        picked_kind = "ko"
    elif num_matches:
        picked = _pick_best_match(text, num_matches)
        picked_kind = "num"

    if picked is not None:
        try:
            if picked_kind == "ko":
                start_date, end_date, start_time, end_time = _parse_ko_date_time_range_match(picked, default_year=now_year)
            else:
                start_date, end_date, start_time, end_time = _parse_num_date_time_range_match(picked)
        except Exception:
            start_date = end_date = start_time = end_time = None

    # 2) fallback: 기존 방식
    if start_date is None:
        # dates 수집
        date_hits: List[str] = []
        for p in DATE_PATS:
            date_hits += p.findall(text or "")

        iso_dates: List[str] = []
        for dh in date_hits:
            iso = _parse_date_token(dh, default_year=now_year)
            if iso and iso not in iso_dates:
                iso_dates.append(iso)

        start_date = iso_dates[0] if len(iso_dates) >= 1 else None
        end_date = iso_dates[1] if len(iso_dates) >= 2 else start_date

        # time 파싱
        m = ISO_TIME_PAT.search(text or "")
        if m:
            ampm1, h1, mi1, _, ampm2, h2, mi2 = m.groups()
            h1 = int(h1)
            h2 = int(h2)
            mi1 = int(mi1) if mi1 is not None else 0
            mi2 = int(mi2) if mi2 is not None else 0
            H1, M1 = _parse_ko_time(ampm1, h1, mi1)
            H2, M2 = _parse_ko_time(ampm2, h2, mi2)
            start_time = _to_hhmm(H1, M1)
            end_time = _to_hhmm(H2, M2)
        else:
            ms = ISO_TIME_SIMPLE.search(text or "")
            if ms:
                ampm1, h1, _, ampm2, h2 = ms.groups()
                h1 = int(h1)
                h2 = int(h2)
                H1, M1 = _parse_ko_time(ampm1, h1, 0)
                H2, M2 = _parse_ko_time(ampm2, h2, 0)
                start_time = _to_hhmm(H1, M1)
                end_time = _to_hhmm(H2, M2)

    # ✅ PATCH: 날짜가 없으면 일정이 아닌 것으로 처리 (title도 비움)
    if not start_date:
        return {
            "title": "",
            "start_date": None,
            "end_date": None,
            "start_time": None,
            "end_time": None,
            "timezone": KST_TZ,
            "all_day": False,
        }

    # all_day 정책
    all_day = bool(start_date and (start_time is None and end_time is None))

    # title (fallback 먼저)
    fallback_title = _fallback_title_rule(text)
    title = fallback_title

    # 날짜가 있는 경우(여기까지 왔으면 있음) LLM로 제목 생성 시도
    if client is not None:
        try:
            llm_title = llm_schedule_title(text, client)
            if llm_title:
                title = llm_title
        except Exception:
            title = fallback_title

    return {
        "title": (title or "")[:120],
        "start_date": start_date,
        "end_date": end_date,
        "start_time": start_time,
        "end_time": end_time,
        "timezone": KST_TZ,
        "all_day": all_day,
    }


# LLM Prompting
PROMPT_TEMPLATE = """
당신은 한국어 'AI 문서 해설사'입니다.
입력 문서를 어르신이 이해하기 쉽게 요약(핵심,행동안내)으로 변환하세요.
정보가 부족하면 요약을 만들지 말고, 사용자에게 물어볼 질문을 제안하세요.
귀여운 손주가 존댓말로 친근하게 설명합니다.

문서 유형: {doc_type}
본문:
{doc_text}

출력(JSON 형식):
{{
  "bullets": ["핵심 3~5가지 요약"],
  "next_actions": ["지금 해야 할 행동 1~2개"]
  "need_more_info": false,
  "ask_back": ["사용자에게 추가로 물어볼 질문 1~3개"]
}}
"""


def llm_summarize(doc_text: str, doc_type: str, client) -> Dict:
    prompt = PROMPT_TEMPLATE.format(doc_type=doc_type, doc_text=(doc_text or "")[:6000])
    if _OPENAI_SDK_V1:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[
                {"role": "system", "content": "JSON만 출력하세요."},
                {"role": "user", "content": prompt},
            ],
        )
        content = resp.choices[0].message.content
    else:
        resp = client.ChatCompletion.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[
                {"role": "system", "content": "JSON만 출력하세요."},
                {"role": "user", "content": prompt},
            ],
        )
        content = resp["choices"][0]["message"]["content"]

    content = (content or "").strip().strip("`")
    if content.lower().startswith("json"):
        content = content[4:].strip()
    try:
        data = json.loads(content)
    except Exception as e:
        data = {"bullets": [f"요약 오류: {e}"], "next_actions": []}
    data.setdefault("bullets", [])
    data.setdefault("next_actions", [])
    return data


# TTS
def tts_bytes_ko(text: str) -> bytes:
    txt = (text or "").strip() or "읽을 내용이 없습니다. 다시 시도해 주세요."
    tts = gTTS(text=txt, lang="ko")
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    return buf.getvalue()


# Pipeline
def process_image_bytes(
    image_bytes: bytes,
    gcp_key: Optional[str] = None,
    openai_key: Optional[str] = None
) -> Dict:
    v_client = build_vision_client(gcp_key)
    o_client = build_openai_client(openai_key)
    full_text, boxes = gcv_ocr(image_bytes, v_client)
    doc_type = guess_doc_type(full_text)

    tags = extract_tags(full_text)

    # 일정 추출 (ISO + LLM title)
    schedule_legacy = extract_schedule(full_text)
    schedule_iso = extract_schedule_iso(full_text, client=o_client)

    # 정보부족 처리
    low = is_low_info(full_text, tags)
    if low:
        summary = {
            "bullets": [],
            "next_actions": [],
            "need_more_info": True,
            "ask_back": [
                "이 문서가 어떤 내용인지 한 줄로 알려주실 수 있을까요?",
                "일정이라면 제목, 날짜 범위, 시간 범위를 알려주세요.",
            ],
        }
    else:
        summary = llm_summarize(full_text, doc_type, o_client)
        # 모델이 과하게 말했을 가능성 최소화: 본문에 없는 숫자 URL이 다수인 경우 제약
        if len(summary.get("bullets", [])) == 0 and len(summary.get("next_actions", [])) == 0:
            summary["need_more_info"] = True
            if not summary.get("ask_back"):
                summary["ask_back"] = ["이 문서의 핵심이 무엇인지 설명해주시면 더 정확히 도와드릴게요."]

    return {
        "doc_type": doc_type,
        "full_text": full_text,
        "boxes": boxes,
        "tags": tags,
        "summary": summary,

        # (ISO 8601 + 캘린더용 title)
        "schedule_suggestion": schedule_iso,
        "schedule_suggestion_legacy": schedule_legacy,
    }


def build_tts_from_summary(summary: Dict, full_text: Optional[str] = None, mode: str = "summary") -> bytes:
    if mode == "summary":
        bullets = (summary or {}).get("bullets", [])
        acts = (summary or {}).get("next_actions", [])
        if (summary or {}).get("need_more_info"):
            speak_text = "사진만으로는 정보가 부족해요. 문서 내용이나 일정의 제목, 날짜 범위, 시간 범위를 알려주시면 정확히 도와드릴게요."
        else:
            speak_text = " / ".join([*bullets, *acts]) or "요약이 비어 있습니다."
        return tts_bytes_ko(speak_text)
    else:
        txt = (full_text or "").strip() or "읽을 내용이 없습니다. 다시 시도해 주세요."
        return tts_bytes_ko(txt[:4000])
