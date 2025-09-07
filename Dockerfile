# 사용할 Python 기본 이미지를 지정합니다.
FROM python:3.11-slim

# 필요한 시스템 라이브러리 설치 (OpenCV를 위해 필요)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 컨테이너 내 작업 디렉토리를 설정합니다.
WORKDIR /app

# 애플리케이션 소스 파일들을 컨테이너로 복사합니다.
# requirements.txt를 먼저 복사하여 종속성 레이어를 캐싱합니다.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# YOLOv8 모델 파일도 컨테이너로 복사합니다.
COPY best_v2_250830.pt .

# 나머지 앱 소스 코드를 복사합니다.
COPY main.py .
COPY ocr_roboflow_GCV_v1.py .
COPY index.html .

# FastAPI 애플리케이션이 사용할 포트를 외부에 노출합니다.
EXPOSE 8080

# 컨테이너 실행 시 실행될 명령어를 정의합니다.
# HOST를 0.0.0.0으로 설정하여 외부 접속을 허용하고, 포트는 8080으로 변경합니다.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]