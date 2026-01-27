# app.py
# DiNuri REST API
# version 1.0.0

# POST /api/ocr-summarize : 이미지 → OCR+요약(JSON)
# POST /api/tts-summary   : 요약 JSON/full_text → MP3
# POST /api/tts           : 텍스트 → MP3
# POST /api/ocr-tags      : 이미지 → OCR → 태그 JSON
# GET  /api/health        : 헬스체크

import os
import json
from typing import Optional
from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from ai_doc_helper2 import (
    process_image_bytes,
    build_tts_from_summary,
    tts_bytes_ko,
)

# env
load_dotenv()
GCP_KEY = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

app = FastAPI(
    title="DiNuri API",
    description="OCR → 요약 → 음성 변환 REST API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/ocr-tags")
async def ocr_tags(
    image: UploadFile = File(...),
    gcp_key_path: Optional[str] = Form(None),
    openai_api_key: Optional[str] = Form(None),
):
    """
    이미지 → OCR → 태그 JSON만 반환
    (프론트에서 태그 기반 기능 제어 가능)
    """
    try:
        img_bytes = await image.read()

        effective_gcp_key = gcp_key_path or GCP_KEY
        effective_openai_key = openai_api_key or OPENAI_KEY

        result = process_image_bytes(
            image_bytes=img_bytes,
            gcp_key=effective_gcp_key,
            openai_key=effective_openai_key,
        )

        tags = result.get("tags", {})
        box_count = len(result.get("boxes", []) or [])

        return JSONResponse({
            "tags": tags,
            "doc_type": result.get("doc_type"),
            "box_count": box_count,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"태그 추출 실패: {str(e)}")


@app.post("/api/ocr-summarize")
async def ocr_summarize(
    image: UploadFile = File(...),
    gcp_key_path: Optional[str] = Form(None),
    openai_api_key: Optional[str] = Form(None),
):
    """
    이미지 → OCR → 태그 + 요약 + 문서유형 반환
    (프론트에서 태그 기반 기능 제어 가능)
    """
    try:
        img_bytes = await image.read()

        effective_gcp_key = gcp_key_path or GCP_KEY
        effective_openai_key = openai_api_key or OPENAI_KEY

        result = process_image_bytes(
            image_bytes=img_bytes,
            gcp_key=effective_gcp_key,
            openai_key=effective_openai_key,
        )

        tags = result.get("tags", {})
        doc_type = result.get("doc_type")
        summary = result.get("summary", {})
        full_text = result.get("full_text", "")

        schedule_suggestion = result.get("schedule_suggestion")
        schedule_suggestion_legacy = result.get("schedule_suggestion_legacy")

        # ISO 8601로 수정했습니다.
        return JSONResponse({
            "doc_type": doc_type,
            "summary": summary,
            "tags": tags,
            "full_text": full_text,
            "schedule_suggestion": schedule_suggestion,
            "schedule_suggestion_legacy": schedule_suggestion_legacy,
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR/요약 실패: {str(e)}")


@app.post("/api/tts-summary")
async def tts_summary_endpoint(
    summary_json: str = Form(...),
    mode: str = Form("summary"),     # "summary" | "full"
    full_text: Optional[str] = Form(None),
):
    try:
        summary = json.loads(summary_json)
        mp3 = build_tts_from_summary(summary, full_text=full_text, mode=mode)
        return Response(content=mp3, media_type="audio/mpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS 생성 실패: {str(e)}")


@app.post("/api/tts")
async def tts_endpoint(text: str = Form(...)):
    try:
        mp3 = tts_bytes_ko(text)
        return Response(content=mp3, media_type="audio/mpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS 실패: {str(e)}")


@app.get("/api/health")
def health_check():
    return {"status": "ok", "message": "DiNuri API 동작 중"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)