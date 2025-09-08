# 차량 VIN 정보 OCR API 시스템

## 📋 프로젝트 개요

이 프로젝트는 차량 이미지에서 VIN(차대번호), 제조사, 제작년도, 모델명을 자동으로 추출하는 OCR API 서비스입니다. YOLOv8 객체 탐지와 Google Cloud Vision API를 활용하여 정확한 차량 정보를 추출합니다.

## 🏗️ 시스템 아키텍처

### 핵심 구성 요소
- **FastAPI**: REST API 서버 프레임워크
- **YOLOv8**: 차량 이미지에서 VIN 영역과 스티커 영역 탐지
- **Google Cloud Vision API**: OCR 텍스트 인식
- **Docker**: 컨테이너화된 배포 환경

### 파일 구조
```
2_CAR_VIN_OCR_AWS1/
├── main.py                    # FastAPI 메인 서버 파일
├── ocr_roboflow_GCV_v1.py    # OCR 핵심 로직 (YOLO + GCV)
├── index.html                # 웹 인터페이스
├── best_v2_250830.pt         # YOLOv8 학습된 모델 파일
├── requirements.txt          # Python 의존성 패키지
├── Dockerfile               # Docker 컨테이너 설정
├── dockerignore            # Docker 빌드 시 제외 파일
└── README.md               # 이 파일
```

## 🚀 설치 및 실행 방법

### 1. 로컬 개발 환경 실행

#### 필수 요구사항
- Python 3.11 이상
- Google Cloud Vision API 키

#### 설치 단계
```bash
# 1. 의존성 패키지 설치
pip install -r requirements.txt

# 2. Google Cloud Vision API 키 설정
# ocr_roboflow_GCV_v1.py 파일의 20번째 줄에서 API 키 수정
api_key = "YOUR_GOOGLE_CLOUD_VISION_API_KEY"

# 3. 로컬 테스트용 서버 실행
uvicorn main:app --host 0.0.0.0 --port 8000
```

#### 접속 방법
- 웹 인터페이스: http://16.184.13.168:8080/api/v1/
- API 문서: http://16.184.13.168:8080/docs

### 2. Docker 컨테이너 실행

#### Docker 이미지 빌드
```bash
# Docker 이미지 빌드
docker build -t vin-ocr .

# 컨테이너 실행 (백그라운드, 자동 재시작)
docker run -d --name vin-ocr --restart=always -p 8080:8080 vin-ocr
```

#### 컨테이너 관리 명령어
```bash
# 실행 중인 컨테이너 확인
docker ps

# 컨테이너 로그 확인
docker logs -f vin-ocr

# 컨테이너 중지
docker stop vin-ocr

# 컨테이너 삭제
docker rm vin-ocr
```

## 🔧 API 사용법

### 주요 엔드포인트

#### 1. 상태 확인
```http
GET /api/v1/status
```
**응답 예시:**
```json
{
  "status": "ready"
}
```

#### 2. 차량 정보 추출
```http
POST /api/v1/ocr/car-info
Content-Type: multipart/form-data

image_file: [이미지 파일]
dryRun: false (선택사항)
```

**성공 응답 예시:**
```json
{
  "status": "success",
  "data": {
    "VIN": "KMHXX00XXXX0000000",
    "Maker": "현대",
    "Year": "2023.05",
    "Model": "아반떼"
  }
}
```

**오류 응답 예시:**
```json
{
  "status": "error",
  "message": "No Text Detected"
}
```

### 웹 인터페이스 사용법
1. 브라우저에서 http://16.184.13.168:8080/api/v1/ 접속
2. "이미지 선택" 버튼 클릭하거나 이미지를 드래그 앤 드롭
3. 자동으로 OCR 처리 후 결과 확인

## 🧠 OCR 처리 로직

### 1. 이미지 전처리
- 이미지 크기 최적화 (최대 2000px)
- 필요시 비율 유지하며 리사이징

### 2. YOLOv8 객체 탐지
- **sticker_area**: 차량 스티커 영역 탐지 (우선순위 높음)
- **vin_area**: VIN 번호 영역 탐지
- 신뢰도 임계값: sticker_area 0.5, vin_area 0.8

### 3. 영역별 OCR 처리
1. **1차**: sticker_area에서 전체 정보 추출 시도
2. **2차**: VIN이 없으면 vin_area에서 VIN만 별도 추출
3. **폴백**: 전체 이미지에서 정보 추출

### 4. 정보 추출 규칙

#### VIN (차대번호)
- 17자리 영숫자 (O→0, I→1 자동 보정)
- 허용 문자: A-H, J-N, P-R, T-Z, 0-9

#### 제조사 (Maker)
- 텍스트 패턴 매칭 우선
- VIN WMI 코드로 보정 (앞 3자리)
- 지원 제조사: 현대, 기아, GM, BMW, 메르세데스벤츠 등

#### 제작년도 (Year)
- 형식: YYYY.MM 또는 YYYY
- 미래 날짜 자동 필터링

#### 모델명 (Model)
- 키워드 기반 추출: "차명", "모델명"
- 히유리스틱 알고리즘으로 보완

## ⚙️ 환경 설정

### 환경 변수
```bash
LOG_LEVEL=INFO                    # 로그 레벨
TIMEOUT_SECONDS=90               # OCR 처리 타임아웃
```

### Google Cloud Vision API 설정
1. Google Cloud Console에서 Vision API 활성화
2. API 키 생성
3. `ocr_roboflow_GCV_v1.py` 파일의 20번째 줄에서 API 키 설정

## 🔍 문제 해결

### 자주 발생하는 문제

#### 1. 모델 로딩 실패
```
YOLO 모델 로드 실패: [오류 메시지]
```
**해결방법:**
- `best_v2_250830.pt` 파일이 올바른 위치에 있는지 확인
- 파일 권한 확인

#### 2. Google Cloud Vision API 오류
```
OCR service is not ready. Status: error
```
**해결방법:**
- API 키가 올바른지 확인
- Google Cloud Console에서 Vision API 활성화 상태 확인
- API 할당량 확인

#### 3. 컨테이너 실행 오류
**해결방법:**
- 포트 8080이 이미 사용 중인지 확인
- Docker 데몬이 실행 중인지 확인
- 컨테이너 로그 확인: `docker logs vin-ocr`

### 로그 확인 방법
```bash
# 로컬 실행 시
# 터미널에서 직접 확인

# Docker 실행 시
docker logs -f vin-ocr
```

## 📊 성능 및 제한사항

### 성능 지표
- **처리 시간**: 평균 3-5초 (이미지 크기에 따라 변동)
- **파일 크기 제한**: 10MB
- **지원 형식**: JPG, JPEG, PNG
- **타임아웃**: 90초

### 정확도
- **VIN 인식률**: 약 85-90%
- **제조사 인식률**: 약 90-95%
- **제작년도 인식률**: 약 80-85%
- **모델명 인식률**: 약 70-80%

### 제한사항
- 이미지 품질이 낮으면 인식률 저하
- 손상되거나 가려진 VIN은 인식 어려움
- 특수 제조사나 구형 차량은 인식률 낮을 수 있음

## 🔄 업데이트 및 유지보수

### 모델 업데이트
1. 새로운 YOLO 모델 파일로 `best_v2_250830.pt` 교체
2. 컨테이너 재빌드 및 재배포

### 코드 수정 시
1. 로컬에서 테스트 완료
2. Docker 이미지 재빌드
3. 컨테이너 재시작

### 로그 모니터링
- 정기적으로 컨테이너 로그 확인
- 오류 발생 시 즉시 대응
- 성능 지표 모니터링

## 📞 연락처 및 지원

### 기술 지원
- 시스템 관련 문의: 아우라웍스 이중재 대표
- API 사용 문의: 아우라웍스 이중재 대표

### 문서 버전
- 작성일: 2025년 9월 7일
- 최종 수정일: 2025년 9월 7일
- 버전: 1.0.0

---

**⚠️ 주의사항**
- Google Cloud Vision API 사용량에 따라 비용이 발생할 수 있습니다.
- 프로덕션 환경에서는 API 키를 환경 변수로 관리하는 것을 권장합니다.
- 정기적인 모델 성능 평가 및 업데이트가 필요합니다.

