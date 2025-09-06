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
from ocr_roboflow_GCV_v1 import process_and_extract_info



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


# Vision API 클라이언트 초기화 상태 관리
OCR_READY = False
OCR_LOADING = False
OCR_ERROR = None
client = None

def _init_gcv_client():
    """Google Cloud Vision API 클라이언트를 초기화하는 백그라운드 작업"""
    global OCR_READY, OCR_LOADING, OCR_ERROR, client
    
    if OCR_READY or OCR_LOADING:
        return
    
    OCR_LOADING = True
    OCR_ERROR = None
    log.info("Starting Google Cloud Vision API client initialization...")
    try:
        from google.cloud import vision
        api_key = os.getenv("GOOGLE_API_KEY", "AIzaSyC_XrPJ4TwUypY7nHc8FouI3lRzIQIqgO8")
        client = vision.ImageAnnotatorClient(client_options={"api_key": api_key})
        log.info("Google Cloud Vision API client initialized successfully.")
        OCR_READY = True
    except Exception as e:
        OCR_ERROR = repr(e)
        log.exception("Failed to initialize Google Cloud Vision API client.")
    finally:
        OCR_LOADING = False
        
        
@app.on_event("startup")
def start_init_thread():
    """서버 시작 시 클라이언트 초기화 작업을 백그라운드 스레드에서 시작"""
    thread = Thread(target=_init_gcv_client, daemon=True, name="GCV-Init")
    thread.start()

# --- 라우트 정의 (API v1) ---

@app.get("/api/v1/")
async def serve_html_page():
    """루트 URL('/api/v1/')에 접속하면 index.html 파일을 반환합니다."""
    # index.html 파일이 main.py와 같은 디렉토리에 있어야 합니다.
    return FileResponse("index.html")


@app.get("/api/v1/healthz")
def healthz():
    """FastAPI 서버 상태 확인"""
    return {"status": "ok"}


@app.get("/api/v1/status")
def get_status():
    """OCR 모델 (Vision API 클라이언트) 상태 확인"""
    return {
        "status": "ready" if OCR_READY else "loading" if OCR_LOADING else "error",
        "detail": OCR_ERROR,
    }


def _run_ocr_blocking(temp_path: str):
    """OCR 로직을 실행하는 동기 함수. 별도의 스레드에서 호출됩니다."""
    try:
        return process_and_extract_info(temp_path)
    finally:
        # 작업 완료 후 임시 파일 삭제
        os.remove(temp_path)
        log.info(f"Temporary file removed: {temp_path}")


@app.post("/api/v1/ocr/car-info", summary="Extract car information from an image")
async def extract_car_info(
    image_file: UploadFile = File(...),
    dry_run: bool = Query(False, description="If true, only check file upload without running OCR.")
):
    ...


async def extract_car_info(
    image_file: UploadFile = File(...),
    dry_run: bool = Query(False, description="If true, only check file upload without running OCR.")
):
    """
    자동차 이미지를 받아 차량 정보(차대번호, 제조사, 연식, 모델명)를 추출합니다.

    - **dry_run**: OCR 모델을 실행하지 않고 파일 업로드만 테스트합니다.
    - **503 Service Unavailable**: Vision API 클라이언트가 로딩 중일 때 반환됩니다.
    - **504 Gateway Timeout**: OCR 처리 시간이 제한 시간을 초과할 경우 반환됩니다.
    """
    log.info(f"OCR request received: dry_run={dry_run}, filename={image_file.filename}, content_type={image_file.content_type}")

    if not OCR_READY and not dry_run:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Google Cloud Vision client is not ready. Please try again in a moment."
        )

    # 1. 파일 처리 및 검증
    try:
        data = await image_file.read()
    except Exception as e:
        log.error(f"Failed to read uploaded file: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Failed to read file.")

    if not data:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty file uploaded.")

    # 2. 임시 파일 저장
    suffix = Path(image_file.filename).suffix if image_file.filename else ".bin"
    try:
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
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail="OCR processing timed out.")
    except Exception as e:
        log.error(f"An error occurred during OCR: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An error occurred during OCR processing: {e}")