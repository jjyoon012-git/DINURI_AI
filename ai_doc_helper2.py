#0204
# ai_doc_helper2.py
# Google Cloud Vision OCR + OpenAI GPT-4o-mini + gTTS

from dotenv import load_dotenv
import io
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
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
from datetime import datetime

# DNS resolver
os.environ.setdefault("GRPC_DNS_RESOLVER", "native")

# env
# NOTE: tmux/서버 환경에서 "예전 키가 남는 문제" 방지 위해 override=True 권장
#       (기존 동작 유지 원하면 override=False로 변경 가능)
load_dotenv(override=True)

DEFAULT_GCP_KEY = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
DEFAULT_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

KST_TZ = "Asia/Seoul"


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


# -----------------------------
# Tag extraction (regex)
# -----------------------------
TAG_REGEX = {
    "날짜": [
        r"\b20\d{2}[./-]\d{1,2}[./-]\d{1,2}(?:\.)?(?:\([월화수목금토일]\))?\b",
        r"\b\d{1,2}\s*월\s*\d{1,2}\s*일(?:\([월화수목금토일]\))?\b",
        r"\b20\d{2}\s*년\s*\d{1,2}\s*월\s*\d{1,2}\s*일(?:\([월화수목금토일]\))?\b"
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


# -----------------------------
# Doc type
# -----------------------------
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


def is_low_info(text: str, tags: Dict[str, List[str]]) -> bool:
    t = (text or "").strip()
    if len(t) < 25:
        return True
    tokens = re.findall(r"[가-힣A-Za-z0-9]{2,}", t)
    return len(tokens) < 10 and len(tags) == 0


# -------------------------------------------------------------------
# Schedule extraction (LEGACY: title/date_range/time_range)
# -------------------------------------------------------------------
TIME_PAT = re.compile(
    r"(오전|오후)?\s*\d{1,2}[:시]\s*\d{0,2}\s*(분)?\s*(~|-|부터)\s*(오전|오후)?\s*\d{1,2}[:시]\s*\d{0,2}\s*(분)?"
)
TIME_SIMPLE = re.compile(r"(오전|오후)?\s*\d{1,2}\s*(~|-)\s*(오전|오후)?\s*\d{1,2}\s*")

DATE_PATS = [
    re.compile(r"20\d{2}[./-]\d{1,2}[./-]\d{1,2}(?:\.)?(?:\([월화수목금토일]\))?"),
    re.compile(r"20\d{2}\s*년\s*\d{1,2}\s*월\s*\d{1,2}\s*일(?:\([월화수목금토일]\))?"),
    re.compile(r"\d{1,2}\s*월\s*\d{1,2}\s*일(?:\([월화수목금토일]\))?"),
]


def extract_schedule(text: str) -> Dict[str, Optional[str]]:
    """
    (LEGACY) 팀 요청: 제목/날짜범위/시간범위
    """
    title = None
    date_hits: List[str] = []
    time_hits: List[str] = []

    for p in DATE_PATS:
        date_hits += p.findall(text or "")

    time_hits += TIME_PAT.findall(text or "")
    if not time_hits:
        time_hits += TIME_SIMPLE.findall(text or "")

    for line in (text or "").splitlines():
        if any(k in line for k in ["회의", "세미나", "행사", "일정", "미팅", "발표", "시연", "면접", "수업", "강의", "오리엔테이션"]):
            title = line.strip()
            break

    date_range = " ~ ".join(date_hits[:2]) if date_hits else ""

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
# ISO 8601 schedule_suggestion + LLM title
# -------------------------------------------------------------------

def _pad2(n: int) -> str:
    return f"{n:02d}"


def _to_iso_date(y: int, m: int, d: int) -> str:
    return f"{y}-{_pad2(m)}-{_pad2(d)}"


def _to_hhmm(h: int, m: int) -> str:
    return f"{_pad2(h)}:{_pad2(m)}"


def _parse_ko_time(ampm: Optional[str], hour: int, minute: int) -> Tuple[int, int]:
    h = hour
    if ampm == "오전":
        if h == 12:
            h = 0
    elif ampm == "오후":
        if h != 12:
            h += 12
    return h, minute


def _strip_date_suffix(token: str) -> str:
    s = (token or "").strip()
    s = re.sub(r"[.,;:]+$", "", s)
    s = re.sub(r"\([월화수목금토일]\)$", "", s).strip()
    return s


def _parse_date_token(s: str, default_year: int) -> Optional[str]:
    s = _strip_date_suffix(s)

    m1 = re.match(r"^(20\d{2})[./-](\d{1,2})[./-](\d{1,2})$", s)
    if m1:
        y, mo, da = map(int, m1.groups())
        return _to_iso_date(y, mo, da)

    m2 = re.match(r"^(20\d{2})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일$", s)
    if m2:
        y, mo, da = map(int, m2.groups())
        return _to_iso_date(y, mo, da)

    m3 = re.match(r"^(\d{1,2})\s*월\s*(\d{1,2})\s*일$", s)
    if m3:
        mo, da = map(int, m3.groups())
        return _to_iso_date(default_year, mo, da)

    return None


RANGE_KEYWORDS = ["등록", "등록기간", "등록 기간", "문서 등록", "신청", "신청기간", "접수", "접수기간", "제출", "납부", "마감", "기한", "기간", "안내", "발표"]

def _score_context(text: str, start: int, end: int) -> int:
    win = (text or "")[max(0, start - 80): min(len(text or ""), end + 80)]
    score = 0
    for kw in RANGE_KEYWORDS:
        if kw in win:
            score += 1
    return score


def _pick_best_match(text: str, matches: List[re.Match]) -> Optional[re.Match]:
    if not matches:
        return None
    best = matches[0]
    best_score = _score_context(text, best.start(), best.end())
    for m in matches[1:]:
        sc = _score_context(text, m.start(), m.end())
        if sc > best_score:
            best = m
            best_score = sc
    return best


KO_DATE_TIME_RANGE_PAT = re.compile(
    r"(?:\b(20\d{2})\s*년\s*)?(\d{1,2})\s*월\s*(\d{1,2})\s*일(?:\([월화수목금토일]\))?\s*"
    r"(오전|오후)?\s*(\d{1,2})\s*[:시]\s*(\d{2})\s*(?:분)?\s*"
    r"(?:부터|~|-)\s*"
    r"(?:\b(20\d{2})\s*년\s*)?(\d{1,2})\s*월\s*(\d{1,2})\s*일(?:\([월화수목금토일]\))?\s*"
    r"(오전|오후)?\s*(\d{1,2})\s*[:시]\s*(\d{2})\s*(?:분)?\s*(?:까지)?"
)

NUM_DATE_TIME_RANGE_PAT = re.compile(
    r"\b(20\d{2})[./-](\d{1,2})[./-](\d{1,2})(?:\.)?(?:\([월화수목금토일]\))?\s+"
    r"(\d{1,2})\s*:\s*(\d{2})\s*(?:~|-|부터)\s*"
    r"(20\d{2})[./-](\d{1,2})[./-](\d{1,2})(?:\.)?(?:\([월화수목금토일]\))?\s+"
    r"(\d{1,2})\s*:\s*(\d{2})\b"
)

ONE_DATE_24H_TIME_RANGE_PAT = re.compile(
    r"\b(20\d{2}[./-]\d{1,2}[./-]\d{1,2}(?:\.)?(?:\([월화수목금토일]\))?)\s+"
    r"(\d{1,2})\s*:\s*(\d{2})\s*(?:~|-|부터)\s*(\d{1,2})\s*:\s*(\d{2})\b"
)

TIME_24H_RANGE_PAT = re.compile(r"\b(\d{1,2})\s*:\s*(\d{2})\s*(?:~|-|부터)\s*(\d{1,2})\s*:\s*(\d{2})\b")

KO_TIME_RANGE_PAT = re.compile(
    r"(오전|오후)?\s*(\d{1,2})(?:[:시]\s*(\d{1,2}))?\s*(?:분)?\s*(~|-|부터|까지)\s*"
    r"(오전|오후)?\s*(\d{1,2})(?:[:시]\s*(\d{1,2}))?\s*(?:분)?"
)


def _parse_ko_date_time_range_match(m: re.Match, default_year: int):
    y1, mo1, d1, ampm1, h1, mi1, y2, mo2, d2, ampm2, h2, mi2 = m.groups()
    y1 = int(y1) if y1 else default_year
    y2 = int(y2) if y2 else y1

    mo1, d1 = int(mo1), int(d1)
    mo2, d2 = int(mo2), int(d2)

    h1, mi1 = int(h1), int(mi1)
    h2, mi2 = int(h2), int(mi2)

    H1, M1 = _parse_ko_time(ampm1, h1, mi1)
    H2, M2 = _parse_ko_time(ampm2, h2, mi2)

    return (
        _to_iso_date(y1, mo1, d1),
        _to_iso_date(y2, mo2, d2),
        _to_hhmm(H1, M1),
        _to_hhmm(H2, M2),
    )


def _parse_num_date_time_range_match(m: re.Match):
    y1, mo1, d1, h1, mi1, y2, mo2, d2, h2, mi2 = m.groups()
    return (
        _to_iso_date(int(y1), int(mo1), int(d1)),
        _to_iso_date(int(y2), int(mo2), int(d2)),
        _to_hhmm(int(h1), int(mi1)),
        _to_hhmm(int(h2), int(mi2)),
    )


def _parse_one_date_24h_time_range_match(m: re.Match, default_year: int):
    date_token, h1, mi1, h2, mi2 = m.groups()
    sd = _parse_date_token(date_token, default_year=default_year)
    if not sd:
        return None, None, None, None
    return sd, sd, _to_hhmm(int(h1), int(mi1)), _to_hhmm(int(h2), int(mi2))


def _fallback_title_rule(text: str) -> str:
    for line in (text or "").splitlines():
        if any(k in line for k in ["등록", "신청", "접수", "제출", "납부", "마감", "발표", "안내", "면접", "오리엔테이션", "설명회", "회의", "세미나", "행사"]):
            s = line.strip()
            return s[:80]
    return ""


def llm_schedule_title(doc_text: str, client) -> str:
    prompt = f"""
역할: 문서(OCR 텍스트)를 객관적으로 분석해서, 캘린더에 추가할 일정 제목 1개만 만든다.

원칙:
- 6~20자 내외
- 날짜/시간/요일/장소/번호/금액/괄호 설명은 포함하지 않는다
- 문서에 실제로 존재하는 이벤트(등록/신청/접수/제출/납부/마감/발표/안내 등)만 제목으로 만든다
- 조항 문장 일부를 그대로 제목으로 쓰지 않는다
- 일정 근거가 불명확하면 빈 문자열

출력은 반드시 JSON:
{{"title":"..."}} 또는 {{"title":""}}

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

    try:
        data = json.loads(content)
    except Exception:
        return ""

    title = (data.get("title") or "").strip()

    if len(title) > 40:
        return ""
    if sum(1 for ch in title if ch in [",", ".", "，", "。", ":", ";"]) >= 2:
        return ""
    if "경우" in title and len(title) >= 16:
        return ""
    return title[:40]


def extract_schedule_iso(text: str, client=None) -> Dict[str, Optional[object]]:
    now_year = datetime.now().year
    src = text or ""

    start_date = end_date = start_time = end_time = None

    ko_matches = list(KO_DATE_TIME_RANGE_PAT.finditer(src))
    num_matches = list(NUM_DATE_TIME_RANGE_PAT.finditer(src))

    picked = None
    picked_kind = None
    if ko_matches:
        picked = _pick_best_match(src, ko_matches)
        picked_kind = "ko"
    elif num_matches:
        picked = _pick_best_match(src, num_matches)
        picked_kind = "num"

    if picked is not None:
        try:
            if picked_kind == "ko":
                start_date, end_date, start_time, end_time = _parse_ko_date_time_range_match(picked, default_year=now_year)
            else:
                start_date, end_date, start_time, end_time = _parse_num_date_time_range_match(picked)
        except Exception:
            start_date = end_date = start_time = end_time = None

    if start_date is None:
        m = ONE_DATE_24H_TIME_RANGE_PAT.search(src)
        if m:
            sd, ed, st, et = _parse_one_date_24h_time_range_match(m, default_year=now_year)
            if sd:
                start_date, end_date, start_time, end_time = sd, ed, st, et

    if start_date is None:
        date_hits: List[str] = []
        for p in DATE_PATS:
            date_hits += p.findall(src)

        iso_dates: List[str] = []
        for dh in date_hits:
            iso = _parse_date_token(dh, default_year=now_year)
            if iso and iso not in iso_dates:
                iso_dates.append(iso)

        start_date = iso_dates[0] if len(iso_dates) >= 1 else None
        end_date = iso_dates[1] if len(iso_dates) >= 2 else start_date

        mt24 = TIME_24H_RANGE_PAT.search(src)
        if mt24:
            h1, mi1, h2, mi2 = mt24.groups()
            start_time = _to_hhmm(int(h1), int(mi1))
            end_time = _to_hhmm(int(h2), int(mi2))
        else:
            mk = KO_TIME_RANGE_PAT.search(src)
            if mk:
                ampm1, h1, mi1, _, ampm2, h2, mi2 = mk.groups()
                h1 = int(h1); h2 = int(h2)
                mi1 = int(mi1) if mi1 is not None else 0
                mi2 = int(mi2) if mi2 is not None else 0
                H1, M1 = _parse_ko_time(ampm1, h1, mi1)
                H2, M2 = _parse_ko_time(ampm2, h2, mi2)
                start_time = _to_hhmm(H1, M1)
                end_time = _to_hhmm(H2, M2)

    if not start_date:
        return {
            "title": "",
            "start_date": "",
            "end_date": "",
            "start_time": "",
            "end_time": "",
            "timezone": KST_TZ,
            "all_day": False,
        }

    all_day = bool(start_date and (not start_time) and (not end_time))

    title = _fallback_title_rule(src)

    if client is not None:
        try:
            llm_title = llm_schedule_title(src, client)
            if llm_title:
                title = llm_title
        except Exception:
            pass

    def _nz(x: Optional[str]) -> str:
        return x if isinstance(x, str) and x is not None else ""

    return {
        "title": (title or "")[:120],
        "start_date": _nz(start_date),
        "end_date": _nz(end_date),
        "start_time": _nz(start_time),
        "end_time": _nz(end_time),
        "timezone": KST_TZ,
        "all_day": bool(all_day),
    }


# -----------------------------
# LLM summarization
# -----------------------------
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


# -----------------------------
# TTS
# -----------------------------
def tts_bytes_ko(text: str) -> bytes:
    txt = (text or "").strip() or "읽을 내용이 없습니다. 다시 시도해 주세요."
    tts = gTTS(text=txt, lang="ko")
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    return buf.getvalue()


# ============================================================
# (B) Action Planning: LLM 선택 + 강제 검증(Guardrails)
# ============================================================

ACTION_TYPES = ["call", "copy_account", "create_calendar"]

# 문맥 점수 키워드(규칙 기반 보조)
CALL_HINTS = ["문의", "연락", "전화", "고객센터", "대표번호", "콜센터"]
ACCOUNT_HINTS = ["입금", "송금", "납부", "계좌", "계좌번호", "무통장", "가상계좌", "은행"]
CAL_HINTS = ["등록", "신청", "접수", "제출", "납부", "마감", "기한", "기간", "설명회", "면접", "오리엔테이션", "예약", "검사"]


def _window(text: str, token: str, radius: int = 60) -> str:
    if not text or not token:
        return ""
    idx = text.find(token)
    if idx == -1:
        return ""
    return text[max(0, idx - radius): min(len(text), idx + len(token) + radius)]


def _ctx_score(text: str, token: str, hints: List[str]) -> float:
    win = _window(text or "", token or "", radius=80)
    if not win:
        return 0.0
    score = 0
    for h in hints:
        if h in win:
            score += 1
    # 0~1로 스케일(느슨하게)
    return min(1.0, score / 4.0)


def _safe_json_load(s: str) -> Optional[Dict[str, Any]]:
    if not s:
        return None
    t = (s or "").strip().strip("`")
    if t.lower().startswith("json"):
        t = t[4:].strip()
    # 일부 모델이 앞뒤에 텍스트를 붙일 수 있어 첫 { ... } 블록만 추출
    m = re.search(r"\{.*\}", t, flags=re.DOTALL)
    if m:
        t = m.group(0)
    try:
        return json.loads(t)
    except Exception:
        return None


def llm_action_plan(
    doc_text: str,
    tags: Dict[str, List[str]],
    schedule_iso: Dict[str, Any],
    client
) -> Dict[str, Any]:
    """
    LLM은 '후보 리스트' 중에서 선택만 하게 만든다.
    - phone_candidates: tags["전화번호"]
    - account_candidates: tags["계좌"]
    - calendar_candidate: schedule_iso (start_date가 있을 때만 의미)
    출력(JSON only):
    {
      "actions": [
        {"type":"call","pick":0,"confidence":0.0~1.0},
        {"type":"copy_account","pick":1,"confidence":...},
        {"type":"create_calendar","pick":"schedule","confidence":...}
      ],
      "need_more_info": false,
      "ask_back": []
    }
    """
    phone_cands = tags.get("전화번호", []) if tags else []
    acct_cands = tags.get("계좌", []) if tags else []

    has_sched = bool((schedule_iso or {}).get("start_date"))

    # 후보만 제공(모델이 값 생성 불가)
    payload = {
        "phone_candidates": phone_cands,
        "account_candidates": acct_cands,
        "schedule_candidate": schedule_iso if has_sched else None,
        "doc_text_snippet": (doc_text or "")[:1200],  # 너무 길게 주지 않음
    }

    prompt = f"""
역할: OCR 문서에서 "실행 버튼"을 안전하게 제안한다. (통화, 계좌복사, 캘린더)

중요 규칙(반드시 준수):
1) 절대로 새로운 전화번호/계좌/날짜/시간을 만들어내지 마라.
2) 아래 후보 리스트에 있는 항목만 선택할 수 있다. 선택은 index(pick)로만 한다.
3) 근거가 약하면 actions를 비우고 need_more_info=true로 둔 뒤, ask_back 질문을 만든다.
4) actions는 최대 3개. confidence는 0.0~1.0.
5) 캘린더는 schedule_candidate가 있을 때만 가능하며 pick은 문자열 "schedule"만 허용한다.

가능한 action type:
- "call" (phone_candidates에서 pick=index)
- "copy_account" (account_candidates에서 pick=index)
- "create_calendar" (pick="schedule")

출력은 반드시 JSON만:
{{
  "actions":[{{"type":"call","pick":0,"confidence":0.9}}],
  "need_more_info": false,
  "ask_back":[]
}}

입력:
{json.dumps(payload, ensure_ascii=False)}
""".strip()

    if _OPENAI_SDK_V1:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.0,
            messages=[
                {"role": "system", "content": "JSON만 출력하세요. 다른 텍스트 금지."},
                {"role": "user", "content": prompt},
            ],
        )
        content = resp.choices[0].message.content
    else:
        resp = client.ChatCompletion.create(
            model="gpt-4o-mini",
            temperature=0.0,
            messages=[
                {"role": "system", "content": "JSON만 출력하세요. 다른 텍스트 금지."},
                {"role": "user", "content": prompt},
            ],
        )
        content = resp["choices"][0]["message"]["content"]

    data = _safe_json_load(content) or {}
    data.setdefault("actions", [])
    data.setdefault("need_more_info", False)
    data.setdefault("ask_back", [])
    return data


def validate_and_build_actions(
    doc_text: str,
    tags: Dict[str, List[str]],
    schedule_iso: Dict[str, Any],
    llm_plan: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Guardrails:
    - LLM이 고른 pick이 후보 범위를 벗어나면 폐기
    - 최종 actions는 evidence를 반드시 포함
    - 문맥 점수 + LLM confidence를 조합해 최종 confidence를 낮춤/상향(보수적)
    - 증거가 없으면 버튼 생성 금지
    """
    phone_cands = tags.get("전화번호", []) if tags else []
    acct_cands = tags.get("계좌", []) if tags else []
    has_sched = bool((schedule_iso or {}).get("start_date"))

    actions_out: List[Dict[str, Any]] = []

    raw_actions = (llm_plan or {}).get("actions", []) or []
    for a in raw_actions:
        atype = (a.get("type") or "").strip()
        pick = a.get("pick")
        conf = a.get("confidence", 0.0)
        try:
            conf = float(conf)
        except Exception:
            conf = 0.0
        conf = max(0.0, min(1.0, conf))

        if atype == "call":
            if not phone_cands:
                continue
            if not isinstance(pick, int):
                continue
            if pick < 0 or pick >= len(phone_cands):
                continue
            phone = phone_cands[pick]
            # 문맥 점수(규칙 기반)로 보수적으로 조정
            ctx = _ctx_score(doc_text, phone, CALL_HINTS)
            final_conf = min(conf, 0.95) * (0.6 + 0.4 * ctx)  # ctx 없으면 0.6배
            # 너무 낮으면 버튼 생성 X
            if final_conf < 0.35:
                continue
            actions_out.append({
                "type": "call",
                "label": "전화하기",
                "evidence": {
                    "phone": phone,
                    "source": "tags.전화번호",
                    "context_score": round(ctx, 3),
                },
                "confidence": round(final_conf, 3),
            })

        elif atype == "copy_account":
            if not acct_cands:
                continue
            if not isinstance(pick, int):
                continue
            if pick < 0 or pick >= len(acct_cands):
                continue
            acct = acct_cands[pick]
            ctx = _ctx_score(doc_text, acct, ACCOUNT_HINTS)
            final_conf = min(conf, 0.95) * (0.55 + 0.45 * ctx)
            if final_conf < 0.35:
                continue
            actions_out.append({
                "type": "copy_account",
                "label": "계좌 복사",
                "evidence": {
                    "account": acct,
                    "source": "tags.계좌",
                    "context_score": round(ctx, 3),
                },
                "confidence": round(final_conf, 3),
            })

        elif atype == "create_calendar":
            if not has_sched:
                continue
            if pick != "schedule":
                continue
            # 일정은 schedule_iso에 증거가 있으므로 문맥은 title/키워드 기반 보수 조정
            title = (schedule_iso or {}).get("title") or ""
            # start_date가 있다는 것 자체가 강한 근거지만,
            # title이 비어있으면 사용자 입력 유도 필요하므로 confidence 낮춤
            base = 0.75 if title else 0.55
            # 문서 전체 키워드 기반 점수(아주 거친)
            kw_hit = 0
            for k in CAL_HINTS:
                if k in (doc_text or ""):
                    kw_hit += 1
            ctx = min(1.0, kw_hit / 6.0)
            final_conf = min(conf if conf > 0 else 0.8, 0.95) * (base + 0.2 * ctx)
            if final_conf < 0.35:
                continue
            actions_out.append({
                "type": "create_calendar",
                "label": "캘린더 추가",
                "evidence": {
                    "schedule": {
                        "title": (schedule_iso or {}).get("title", ""),
                        "start_date": (schedule_iso or {}).get("start_date", ""),
                        "end_date": (schedule_iso or {}).get("end_date", ""),
                        "start_time": (schedule_iso or {}).get("start_time", ""),
                        "end_time": (schedule_iso or {}).get("end_time", ""),
                        "timezone": (schedule_iso or {}).get("timezone", KST_TZ),
                        "all_day": bool((schedule_iso or {}).get("all_day", False)),
                    },
                    "source": "extract_schedule_iso",
                    "context_score": round(ctx, 3),
                },
                "confidence": round(final_conf, 3),
            })

    # 중복 타입이 여러 개 들어오면 confidence 높은 것만 유지(안전)
    best_by_type: Dict[str, Dict[str, Any]] = {}
    for a in actions_out:
        t = a["type"]
        if t not in best_by_type or a.get("confidence", 0) > best_by_type[t].get("confidence", 0):
            best_by_type[t] = a

    actions_final = list(best_by_type.values())
    # confidence 내림차순 정렬
    actions_final.sort(key=lambda x: x.get("confidence", 0), reverse=True)

    # need_more_info 결정(액션이 하나도 없고 정보는 있으나 애매한 경우)
    need_more_info = False
    ask_back: List[str] = []

    if len(actions_final) == 0:
        # 태그가 거의 없고 스케줄도 없으면 질문 유도
        if (not phone_cands) and (not acct_cands) and (not has_sched):
            need_more_info = True
            ask_back = [
                "이 문서가 '전화', '입금/송금', '일정(등록/마감/예약)' 중 어떤 목적의 문서인지 알려주실 수 있을까요?",
                "통화/계좌/일정 중 무엇을 하려는지 말씀해 주시면 정확히 버튼을 만들어드릴게요.",
            ]
        else:
            # 후보는 있는데 문맥/신뢰도가 낮아서 걸러진 케이스
            need_more_info = True
            ask_back = [
                "이 문서에서 '전화하기' 또는 '계좌로 납부/송금'이 필요한 상황인가요?",
                "캘린더에 추가해야 하는 일정(신청/등록/마감/예약)이 맞나요? 맞다면 제목도 알려주세요.",
            ]

    return {
        "actions": actions_final,
        "need_more_info": bool(need_more_info),
        "ask_back": ask_back,
    }


# -----------------------------
# Pipeline
# -----------------------------
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

    schedule_legacy = extract_schedule(full_text)
    schedule_iso = extract_schedule_iso(full_text, client=o_client)

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
        # low-info에서는 액션도 최소화
        action_pack = {
            "actions": [],
            "need_more_info": True,
            "ask_back": [
                "통화/계좌복사/캘린더 중 무엇을 하려는지 알려주실 수 있을까요?",
            ],
        }
    else:
        summary = llm_summarize(full_text, doc_type, o_client)
        if len(summary.get("bullets", [])) == 0 and len(summary.get("next_actions", [])) == 0:
            summary["need_more_info"] = True
            if not summary.get("ask_back"):
                summary["ask_back"] = ["이 문서의 핵심이 무엇인지 설명해주시면 더 정확히 도와드릴게요."]

        # (B) Action planning (LLM 선택 + 백엔드 강제 검증)
        llm_plan = llm_action_plan(full_text, tags, schedule_iso, o_client)
        action_pack = validate_and_build_actions(full_text, tags, schedule_iso, llm_plan)

        # 액션이 need_more_info면 summary에도 질문을 보강(사용자 경험 통일)
        if action_pack.get("need_more_info") and not summary.get("need_more_info"):
            summary["need_more_info"] = True
            # ask_back 합치기(중복 제거)
            merged = []
            for q in (summary.get("ask_back", []) or []) + (action_pack.get("ask_back", []) or []):
                q = (q or "").strip()
                if q and q not in merged:
                    merged.append(q)
            summary["ask_back"] = merged[:3]

    return {
        "doc_type": doc_type,
        "full_text": full_text,
        "boxes": boxes,
        "tags": tags,
        "summary": summary,

        # 기존 유지
        "schedule_suggestion": schedule_iso,
        "schedule_suggestion_legacy": schedule_legacy,

        # 신규: UI/실행용 액션(정확도 중요하니 LLM next_actions와 분리)
        "ui_actions": action_pack.get("actions", []),
        "ui_actions_need_more_info": bool(action_pack.get("need_more_info", False)),
        "ui_actions_ask_back": action_pack.get("ask_back", []),
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
