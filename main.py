#접속주소 : http://127.0.0.1:8000/api/v1/

import os
import time
import logging
import asyncio
from pathlib import Path
from tempfile import NamedTemporaryFile
from fastapi import FastAPI, File, UploadFile, Query, HTTPException, status, APIRouter
from fastapi.responses import JSONResponse, FileResponse
from threading import Thread

from fastapi.middleware.cors import CORSMiddleware

# OCR 핵심 로직 파일을 임포트합니다.
# ocr_roboflow_GCV_v1.py에서 상태 확인 함수와 OCR 처리 함수를 가져옵니다.
from ocr_roboflow_GCV_v1 import process_and_extract_info, get_client_status


# --- 환경 변수 설정 (배포 환경에서 사용) ---
# 로깅 레벨, CORS 허용 오리진 등은 필요에 따라 설정하세요.
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
TIMEOUT_SECONDS = int(os.getenv("TIMEOUT_SECONDS", 90))

# --- 로깅 설정 ---
logging.basicConfig(
    level=LOG_LEVEL,
    format="[%(asctime)s] [%(levelname)s] [%(process)d] [%(threadName)s] [%(name)s] %(message)s",
)
log = logging.getLogger("uvicorn")

app = FastAPI(
    title="Car Info OCR API",
    description="A FastAPI service to extract car information (VIN, Year, Maker, Model) from an image.",
    version="1.0.0"
)

# --- CORS 미들웨어 ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/v1/")
async def serve_html_page():
    """루트 URL('/')에 접속하면 index.html 파일을 반환합니다."""
    return FileResponse("index.html")


@app.get("/api/v1/status")
async def get_status():
    """OCR 모듈의 상태를 확인하는 API 엔드포인트."""
    # ocr_roboflow_GCV_v1.py에서 직접 상태를 가져와 반환합니다.
    current_status, error_detail = get_client_status()
    if current_status == "error":
        log.error(f"OCR module reports an error: {error_detail}")
        return {"status": "error", "detail": error_detail}
    return {"status": current_status}


# OCR 작업을 별도의 스레드에서 실행하기 위한 헬퍼 함수
def _run_ocr_blocking(temp_path: str):
    """
    OCR 처리를 동기적으로 실행하고 결과를 반환합니다.
    `process_and_extract_info` 함수가 이미 클라이언트 객체를 내부적으로 사용하도록 설계되어 있습니다.
    """
    try:
        car_info, result_status = process_and_extract_info(temp_path)
        return car_info, result_status
    finally:
        # 임시 파일 정리
        os.remove(temp_path)


@app.post("/api/v1/ocr/car-info")
async def extract_car_info_api(
    image_file: UploadFile = File(...),
    dry_run: bool = Query(False, alias="dryRun")
):
    """
    업로드된 이미지에서 차량 정보를 추출하는 API 엔드포인트.
    """
    # 1. OCR 모듈의 상태를 먼저 확인합니다.
    current_status, error_detail = get_client_status()
    if current_status != "ready":
        # 'loading' 또는 'error' 상태일 경우 적절한 HTTP 오류를 반환합니다.
        log.warning(f"OCR service is not ready. Status: {current_status}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"OCR service is not ready. Status: {current_status}, Error: {error_detail}"
        )
    
    # 2. 업로드된 파일 처리
    data = await image_file.read()
    
    # 2.1. 파일 크기 확인 (10MB 제한)
    if len(data) > 10 * 1024 * 1024:
        raise HTTPException(status_code=status.HTTP_413_PAYLOAD_TOO_LARGE, detail="File size exceeds the 10MB limit.")
        
    # 2.2. 임시 파일로 저장
    try:
        suffix = Path(image_file.filename).suffix
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(data)
            temp_path = tmp.name
        log.info(f"Temporary file saved: {temp_path} ({len(data)} bytes)")
    except Exception as e:
        log.error(f"Failed to save temporary file: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Could not save file on server.")
    
    # 2.5. 드라이런
    if dry_run:
        os.remove(temp_path)
        return {"message": "Dry run successful.", "file_size": len(data)}

    # 3. OCR 실행
    try:
        loop = asyncio.get_running_loop()
        car_info, result_status = await asyncio.wait_for(
            loop.run_in_executor(None, _run_ocr_blocking, temp_path),
            timeout=TIMEOUT_SECONDS
        )
        
        if result_status == "Success":
            return JSONResponse(content={"status": "success", "data": car_info})
        else:
            return JSONResponse(content={"status": "error", "message": result_status}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

    except asyncio.TimeoutError:
        log.error(f"OCR process timed out after {TIMEOUT_SECONDS} seconds.")
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail="OCR process timed out.")
    except Exception as e:
        log.exception("An unexpected error occurred during OCR process.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An unexpected error occurred: {str(e)}")