# 0204: 계좌 엄격 3트 (ULTRA STRICT)
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

os.environ.setdefault("GRPC_DNS_RESOLVER", "native")

# tmux/서버 환경에서 예전 export 키가 .env를 덮는 문제를 줄이려면 override=True 권장
load_dotenv(override=True)

DEFAULT_GCP_KEY = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
DEFAULT_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

KST_TZ = "Asia/Seoul"


# ============================================================
# Clients
# ============================================================

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


# ============================================================
# OCR
# ============================================================

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


# ============================================================
# Tag extraction (regex)
# ============================================================

PHONE_PAT_STRICT = re.compile(r"\b0\d{1,2}-\d{3,4}-\d{4}\b")

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
    # 계좌는 후보 추출만: 최종 버튼은 서버의 strict recheck + bank name mandatory를 통과해야 생성
    "계좌": [
        # 하이픈 숫자열 후보 (후속 필터에서 2-4-4/2-3-4/3-4-4, phone-like 강차단 + bank+context 강제)
        r"\b\d{2,4}-\d{2,6}-\d{2,8}(?:-\d{1,6})?\b",
        # '계좌번호:' 같은 명시 후보 (그래도 은행명 강제)
        r"\b계좌(?:번호)?\s*[:：]?\s*\d[\d-]{9,}\b"
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


# ============================================================
# Account/Phone strict helpers (핵심: bank mandatory + shape exclusion)
# ============================================================

ACCOUNT_POS_KWS = ["계좌", "계좌번호", "가상계좌", "입금", "송금", "납부", "이체", "은행", "예금주", "수취"]
ACCOUNT_NEG_KWS = ["문서번호", "접수번호", "승인번호", "관리번호", "고객번호", "주문번호", "운송장", "송장", "청구번호"]

# 전화 문맥이 근처면 계좌 차단(“전화번호가 계좌로 추출” 방지)
ACCOUNT_PHONE_NEG_KWS = ["전화", "문의", "연락", "고객센터", "대표번호", "상담", "ARS", "내선", "FAX", "팩스"]

BANK_NAME_KWS = [
    "국민", "신한", "우리", "하나", "농협", "기업", "카카오뱅크", "토스뱅크",
    "새마을", "우체국", "수협", "대구", "부산", "광주", "전북", "경남", "SC", "씨티"
]

# 계좌로 절대 허용하지 않을 하이픈 형태
EXCLUDE_HYPHEN_SHAPES = {(2, 4, 4), (2, 3, 4), (3, 4, 4)}


def normalize_digits_hyphen(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"[^\d-]", "", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s


def _window(text: str, token: str, radius: int = 160) -> str:
    if not text or not token:
        return ""
    idx = (text or "").find(token)
    if idx == -1:
        return ""
    return (text or "")[max(0, idx - radius): min(len(text or ""), idx + len(token) + radius)]


def _has_any(text: str, kws: List[str]) -> bool:
    t = text or ""
    return any(k in t for k in kws)


def is_phone_like(raw: str) -> bool:
    if not raw:
        return False

    s = raw.strip()

    # 하이픈 전화
    if PHONE_PAT_STRICT.search(s):
        return True

    # 숫자만 전화(0으로 시작 10~11자리)
    digits = re.sub(r"\D", "", s)
    if len(digits) in (10, 11) and digits.startswith("0"):
        return True

    return False


def is_excluded_hyphen_shape(raw: str) -> bool:
    s = normalize_digits_hyphen(raw)
    parts = s.split("-") if s else []
    if len(parts) == 3:
        shape = (len(parts[0]), len(parts[1]), len(parts[2]))
        return shape in EXCLUDE_HYPHEN_SHAPES
    return False


def _account_context_gate(text: str, token: str) -> bool:
    """
    계좌 후보가 '계좌'로 인정되려면:
    - 은행명 1개 이상 + 계좌 문맥 1개 이상이 같은 window에 존재
    - 문서번호류/전화문맥 있으면 즉시 차단
    """
    win = _window(text, token, radius=220)
    if not win:
        return False

    if _has_any(win, ACCOUNT_NEG_KWS):
        return False
    if _has_any(win, ACCOUNT_PHONE_NEG_KWS):
        return False

    has_bank = _has_any(win, BANK_NAME_KWS)
    has_pos = _has_any(win, ACCOUNT_POS_KWS)

    return bool(has_bank and has_pos)


def is_plausible_korean_account(raw: str) -> bool:
    """
    Ultra strict format filter:
    - phone-like 차단
    - 2-4-4 / 2-3-4 / 3-4-4 차단
    - 하이픈 포함
    - digits 길이 9~16
    - 파트 최소 길이 2
    """
    if is_phone_like(raw):
        return False
    if is_excluded_hyphen_shape(raw):
        return False

    s = normalize_digits_hyphen(raw)
    if "-" not in s:
        return False

    digits = re.sub(r"-", "", s)
    if not (9 <= len(digits) <= 16):
        return False

    parts = s.split("-")
    if any(len(p) < 2 for p in parts):
        return False

    return True


def ctx_score_account(text: str, token: str) -> float:
    """
    점수는 사실상 gate로 쓴다:
    - 은행명+계좌문맥이 있으면 0.90 이상
    - 그 외는 0.0
    """
    win = _window(text, token, radius=240)
    if not win:
        return 0.0

    if _has_any(win, ACCOUNT_NEG_KWS):
        return 0.0
    if _has_any(win, ACCOUNT_PHONE_NEG_KWS):
        return 0.0

    pos = sum(1 for k in ACCOUNT_POS_KWS if k in win)
    bank = sum(1 for k in BANK_NAME_KWS if k in win)

    if pos < 1 or bank < 1:
        return 0.0

    # 최소 요건만 만족해도 0.90 (THRESH_ACCOUNT 통과)
    score = 0.0
    score += 0.45  # pos 존재
    score += 0.45  # bank 존재
    if pos >= 2:
        score += 0.10
    return max(0.0, min(1.0, score))


def _filter_account_candidates_in_tags(full_text: str, cands: List[str]) -> List[str]:
    """
    tags 단계에서도 계좌 후보를 최대한 줄여서 LLM 오염 방지.
    (최종 UI는 validate 단계에서 다시 strict recheck)
    """
    out: List[str] = []
    for c in cands or []:
        if is_phone_like(c):
            continue
        if is_excluded_hyphen_shape(c):
            continue
        if not is_plausible_korean_account(c):
            continue
        if not _account_context_gate(full_text, c):
            continue
        norm = normalize_digits_hyphen(c)
        if norm and norm not in out:
            out.append(norm)
    return out


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

    # 계좌 태그는 여기서 1차로 강하게 줄인다(은행명+문맥 필수)
    if found.get("계좌"):
        found["계좌"] = _filter_account_candidates_in_tags(text or "", found["계좌"])

    return {k: v for k, v in found.items() if v}


# ============================================================
# Doc type
# ============================================================

DOC_TYPES = {
    "정부·공공": ["지원금", "세금", "건강보험", "환급"],
    "의료": ["진단서", "혈압", "혈당", "검사", "병원"],
    "금융": ["대출", "보험", "약관", "이자", "수수료", "납부", "입금", "가상계좌", "계좌번호"],
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


# ============================================================
# Schedule extraction (LEGACY)
# ============================================================

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


# ============================================================
# ISO 8601 schedule_suggestion + LLM title
# ============================================================

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
- 일정 근거가 불명확하면 빈 문자열

출력 JSON:
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
                h1 = int(h1)
                h2 = int(h2)
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


# ============================================================
# LLM summarization
# ============================================================

PROMPT_TEMPLATE = """
당신은 한국어 'AI 문서 해설사'입니다.
입력 문서를 어르신이 이해하기 쉽게 요약(핵심,행동안내)으로 변환하세요.
정보가 부족하면 요약을 만들지 말고, 사용자에게 물어볼 질문을 제안하세요.
친근한 존댓말로 설명합니다.

문서 유형: {doc_type}
본문:
{doc_text}

출력(JSON 형식):
{{
  "bullets": ["핵심 3~5가지 요약"],
  "next_actions": ["지금 해야 할 행동 1~2개"],
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
        data = {"bullets": [f"요약 오류: {e}"], "next_actions": [], "need_more_info": True, "ask_back": []}
    data.setdefault("bullets", [])
    data.setdefault("next_actions", [])
    data.setdefault("need_more_info", False)
    data.setdefault("ask_back", [])
    return data


# ============================================================
# TTS
# ============================================================

def tts_bytes_ko(text: str) -> bytes:
    txt = (text or "").strip() or "읽을 내용이 없습니다. 다시 시도해 주세요."
    tts = gTTS(text=txt, lang="ko")
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    return buf.getvalue()


# ============================================================
# Action Planning (ultra-strict account)
# ============================================================

CALL_HINTS = ["문의", "연락", "전화", "고객센터", "대표번호", "콜센터", "상담", "ARS", "내선", "FAX", "팩스"]
CAL_EVENT_HINTS = ["등록", "신청", "접수", "제출", "납부", "마감", "기한", "기간", "예약", "검사", "면접", "설명회", "오리엔테이션"]
CAL_NEG_HINTS = ["발행", "발급", "작성", "출력", "고지", "청구"]

ACTION_TYPES = ["call", "copy_account", "create_calendar"]

THRESH_CALL = 0.60
THRESH_ACCOUNT = 0.90
THRESH_CAL = 0.70

ENABLE_SINGLETON_EXCEPTION_CALL = True
ENABLE_SINGLETON_EXCEPTION_ACCOUNT = False

SINGLETON_MAX_LEN = 180
SINGLETON_CONF_MAX = 0.55
SINGLETON_MIN = 0.45


def _ctx_score(text: str, token: str, hints: List[str]) -> float:
    win = _window(text or "", token or "", radius=90)
    if not win:
        return 0.0
    score = 0
    for h in hints:
        if h in win:
            score += 1
    return min(1.0, score / 4.0)


def extract_account_candidates_strict(doc_text: str, raw_candidates: List[str]) -> List[str]:
    """
    서버 최종 strict 후보:
    - phone-like/2-4-4/2-3-4/3-4-4 제외
    - format plausibility
    - 은행명+계좌문맥 반드시 존재
    - neg/전화문맥 있으면 차단
    - ctx_score_account >= THRESH_ACCOUNT
    """
    out: List[str] = []
    for c in raw_candidates or []:
        if is_phone_like(c):
            continue
        if is_excluded_hyphen_shape(c):
            continue
        if not is_plausible_korean_account(c):
            continue

        sc = ctx_score_account(doc_text, c)
        if sc < THRESH_ACCOUNT:
            continue

        s = normalize_digits_hyphen(c)
        if s and s not in out:
            out.append(s)
    return out


def _safe_json_load(s: str) -> Optional[Dict[str, Any]]:
    if not s:
        return None
    t = (s or "").strip().strip("`")
    if t.lower().startswith("json"):
        t = t[4:].strip()
    m = re.search(r"\{.*\}", t, flags=re.DOTALL)
    if m:
        t = m.group(0)
    try:
        return json.loads(t)
    except Exception:
        return None


def _singleton_exception_ok_call(doc_text: str, phone_cands: List[str], has_sched: bool) -> bool:
    if not ENABLE_SINGLETON_EXCEPTION_CALL:
        return False
    txt = (doc_text or "").strip()
    if len(txt) > SINGLETON_MAX_LEN:
        return False
    num_tokens = re.findall(r"\d{3,}", txt)
    if len(num_tokens) >= 6:
        return False
    return (len(phone_cands) == 1) and (not has_sched)


def llm_action_plan(
    doc_text: str,
    tags: Dict[str, List[str]],
    schedule_iso: Dict[str, Any],
    client
) -> Dict[str, Any]:
    """
    LLM은 후보를 고르는 역할만.
    계좌는 후보 리스트에 있어도 서버 strict에서 다시 걸러진다.
    """
    phone_cands = tags.get("전화번호", []) if tags else []
    acct_raw = tags.get("계좌", []) if tags else []
    has_sched = bool((schedule_iso or {}).get("start_date"))

    phone_payload = [{"value": p, "context": _window(doc_text, p, 80)} for p in phone_cands]
    acct_payload = [{"value": a, "context": _window(doc_text, a, 220)} for a in acct_raw]

    payload = {
        "doc_len": len((doc_text or "").strip()),
        "phone_candidates": phone_payload,
        "account_candidates": acct_payload,
        "schedule_candidate": schedule_iso if has_sched else None,
        "doc_text_snippet": (doc_text or "")[:900],
    }

    prompt = f"""
역할: OCR 문서에서 실행 버튼(통화/계좌복사/캘린더)을 매우 보수적으로 제안한다.

규칙:
1) 새로운 전화번호/계좌/날짜/시간/금액을 절대로 만들어내지 마라.
2) 아래 후보 리스트에 있는 항목만 선택할 수 있다. 선택은 index(pick)로만 한다.
3) 확실하지 않으면 actions를 비우고 need_more_info=true + 질문을 만든다.
4) actions는 최대 2개.
5) confidence는 0.0~1.0. 애매하면 낮게.

허용 action type:
- "call" (phone_candidates에서 pick=int)
- "copy_account" (account_candidates에서 pick=int)
- "create_calendar" (pick="schedule"; schedule_candidate가 있을 때만)

copy_account(초엄격):
- 후보 주변에 은행명이 반드시 있어야 한다.
- 후보 주변에 계좌/가상계좌/입금/납부/이체/예금주 같은 계좌 문맥이 있어야 한다.
- 후보 주변에 전화/ARS/FAX/문의/고객센터 단서가 있으면 금지.
- 문서번호/접수번호/승인번호/관리번호 주변이면 금지.
- 단독 계좌 예외는 금지.

출력(JSON만):
{{
  "actions":[
    {{"type":"call","pick":0,"confidence":0.8,"reason":"문의"}},
    {{"type":"copy_account","pick":0,"confidence":0.6,"reason":"은행명"}}
  ],
  "need_more_info": false,
  "ask_back": []
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
    서버 최종 게이트:
    - 계좌: strict 후보 리스트만 인정 (은행명+문맥 필수 + 2-4-4/2-3-4/3-4-4/phone-like 차단)
    - 전화/캘린더는 기존 정책 유지
    """
    phone_cands = tags.get("전화번호", []) if tags else []
    acct_raw = tags.get("계좌", []) if tags else []
    has_sched = bool((schedule_iso or {}).get("start_date"))

    acct_cands_strict = extract_account_candidates_strict(doc_text, acct_raw)

    actions_out: List[Dict[str, Any]] = []
    raw_actions = (llm_plan or {}).get("actions", []) or []

    for a in raw_actions[:3]:
        atype = (a.get("type") or "").strip()
        pick = a.get("pick")
        reason = (a.get("reason") or "").strip()
        conf = a.get("confidence", 0.0)
        try:
            conf = float(conf)
        except Exception:
            conf = 0.0
        conf = max(0.0, min(1.0, conf))

        if atype == "call":
            if not phone_cands or not isinstance(pick, int) or pick < 0 or pick >= len(phone_cands):
                continue
            phone = phone_cands[pick]
            ctx = _ctx_score(doc_text, phone, CALL_HINTS)

            win = _window(doc_text, phone, 100)
            only_faxish = ("FAX" in win or "팩스" in win) and (not _has_any(win, ["문의", "연락", "고객센터", "상담", "대표번호", "전화"]))
            if only_faxish:
                continue

            final_conf = min(conf, 0.95) * (0.55 + 0.45 * ctx)

            if final_conf < THRESH_CALL:
                if _singleton_exception_ok_call(doc_text, phone_cands, has_sched):
                    final_conf = max(SINGLETON_MIN, min(SINGLETON_CONF_MAX, max(final_conf, 0.50)))
                    reason = reason or "단독 전화"
                else:
                    continue

            actions_out.append({
                "type": "call",
                "label": "전화하기",
                "evidence": {
                    "phone": phone,
                    "source": "tags.전화번호",
                    "context_score": round(ctx, 3),
                    "reason": reason[:20],
                },
                "confidence": round(final_conf, 3),
            })

        elif atype == "copy_account":
            # strict 후보만 인정
            if not acct_cands_strict:
                continue
            if not isinstance(pick, int) or pick < 0 or pick >= len(acct_raw):
                continue

            raw = acct_raw[pick]

            if is_phone_like(raw):
                continue
            if is_excluded_hyphen_shape(raw):
                continue

            norm = normalize_digits_hyphen(raw)
            if not norm or norm not in acct_cands_strict:
                continue

            ctx = ctx_score_account(doc_text, raw)
            if ctx < THRESH_ACCOUNT:
                continue

            final_conf = min(conf if conf > 0 else 0.90, 0.95) * (0.80 + 0.20 * ctx)
            if final_conf < THRESH_ACCOUNT:
                continue

            actions_out.append({
                "type": "copy_account",
                "label": "계좌 복사",
                "evidence": {
                    "account": norm,
                    "source": "tags.계좌 (ultra_strict_recheck)",
                    "context_score": round(ctx, 3),
                    "reason": (reason or "은행명+문맥")[:20],
                },
                "confidence": round(final_conf, 3),
            })

        elif atype == "create_calendar":
            if not has_sched:
                continue
            if pick != "schedule":
                continue

            txt = doc_text or ""
            has_pos = _has_any(txt, CAL_EVENT_HINTS)
            has_neg = _has_any(txt, CAL_NEG_HINTS)

            if not has_pos:
                continue
            if has_neg and not _has_any(txt, ["마감", "기한", "까지", "신청", "접수", "등록", "예약"]):
                continue

            title = (schedule_iso or {}).get("title") or ""
            base = 0.80 if title else 0.72
            hit = sum(1 for k in CAL_EVENT_HINTS if k in txt)
            ctx = min(1.0, hit / 6.0)

            final_conf = min(conf if conf > 0 else 0.85, 0.95) * (base + 0.2 * ctx)
            if final_conf < THRESH_CAL:
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
                    "reason": (reason or "이벤트")[:20],
                },
                "confidence": round(final_conf, 3),
            })

    # 동일 타입 중복 제거(최고 confidence만)
    best_by_type: Dict[str, Dict[str, Any]] = {}
    for x in actions_out:
        t = x["type"]
        if t not in best_by_type or x.get("confidence", 0) > best_by_type[t].get("confidence", 0):
            best_by_type[t] = x

    actions_final = list(best_by_type.values())
    actions_final.sort(key=lambda x: x.get("confidence", 0), reverse=True)

    need_more_info = False
    ask_back: List[str] = []

    if len(actions_final) == 0:
        if (not phone_cands) and (not acct_raw) and (not has_sched):
            need_more_info = True
            ask_back = [
                "이 문서에서 하려는 일이 전화, 입금/납부, 일정 추가 중 무엇인가요?",
                "가능하면 문서의 핵심 문장을 한 줄만 알려주세요.",
            ]
        else:
            need_more_info = True
            ask_back = [
                "이 정보가 실제로 입금/납부용 계좌가 맞나요?",
                "은행명과 납부/입금 문구가 함께 있는 줄을 알려주시면 더 정확합니다.",
            ]

    return {
        "actions": actions_final,
        "need_more_info": bool(need_more_info),
        "ask_back": ask_back[:3],
    }


# ============================================================
# Pipeline
# ============================================================

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
        action_pack = {
            "actions": [],
            "need_more_info": True,
            "ask_back": [
                "통화, 계좌복사, 캘린더 중 무엇을 하려는지 알려주실 수 있을까요?",
            ],
        }
    else:
        summary = llm_summarize(full_text, doc_type, o_client)
        if len(summary.get("bullets", [])) == 0 and len(summary.get("next_actions", [])) == 0:
            summary["need_more_info"] = True
            if not summary.get("ask_back"):
                summary["ask_back"] = ["이 문서의 핵심이 무엇인지 설명해주시면 더 정확히 도와드릴게요."]

        llm_plan = llm_action_plan(full_text, tags, schedule_iso, o_client)
        action_pack = validate_and_build_actions(full_text, tags, schedule_iso, llm_plan)

        if action_pack.get("need_more_info") and not summary.get("need_more_info"):
            summary["need_more_info"] = True

        merged: List[str] = []
        for q in (summary.get("ask_back", []) or []) + (action_pack.get("ask_back", []) or []):
            q = (q or "").strip()
            if q and q not in merged:
                merged.append(q)
        if merged:
            summary["ask_back"] = merged[:3]

    return {
        "doc_type": doc_type,
        "full_text": full_text,
        "boxes": boxes,
        "tags": tags,
        "summary": summary,

        "schedule_suggestion": schedule_iso,
        "schedule_suggestion_legacy": schedule_legacy,

        "ui_actions": action_pack.get("actions", []),
        "ui_actions_need_more_info": bool(action_pack.get("need_more_info", False)),
        "ui_actions_ask_back": action_pack.get("ask_back", []),
    }


def build_tts_from_summary(summary: Dict, full_text: Optional[str] = None, mode: str = "summary") -> bytes:
    if mode == "summary":
        bullets = (summary or {}).get("bullets", [])
        acts = (summary or {}).get("next_actions", [])
        if (summary or {}).get("need_more_info"):
            speak_text = "사진만으로는 정보가 부족합니다. 문서 내용이나 일정의 제목, 날짜 범위, 시간 범위를 알려주시면 더 정확히 도와드릴게요."
        else:
            speak_text = " / ".join([*bullets, *acts]) or "요약이 비어 있습니다."
        return tts_bytes_ko(speak_text)
    else:
        txt = (full_text or "").strip() or "읽을 내용이 없습니다. 다시 시도해 주세요."
        return tts_bytes_ko(txt[:4000])
