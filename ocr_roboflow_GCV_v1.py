
#ocr_roboflow_GCV_v1: 파일명 ocr_roboflow_5_GCP.PY 동일 파일임. 배포용으로 이름만 변경 실시.


import os
import cv2
from ultralytics import YOLO
from google.cloud import vision
import io

# 1. API 키 설정 (YOUR_API_KEY 부분을 본인의 API 키로 교체하세요)
# --- Vision API 클라이언트 초기화 및 상태 관리 ---
client = None
status = "loading"
error_message = None

def initialize_client_on_import():
    global client, status, error_message
    
    api_key = "AIzaSyC_XrPJ4TwUypY7nHc8FouI3lRzIQIqgO8" # 여기에 실제 키 하드코딩

    try:
        client = vision.ImageAnnotatorClient(client_options={"api_key": api_key})
        status = "ready"
    except Exception as e:
        status = "error"
        error_message = str(e)

# 파일이 임포트될 때 클라이언트 초기화 함수를 바로 실행합니다.
initialize_client_on_import()
# main.py에서 호출할 상태 확인 함수
def get_client_status():
    return status, error_message

# YOLOv8 모델 로드
MODEL_PATH = r"C:/01_Coding/250801_CAR_OCR_PHOTO/2_CAR_VIN_OCR_AWS1/best_v2_250830.pt"
yolo_model = YOLO(MODEL_PATH)



def resize_image_if_needed(image, max_dim=2000):
    """
    이미지 픽셀 크기가 max_dim보다 크면 비율을 유지하며 축소합니다.
    """
    h, w = image.shape[:2]
    if h > max_dim or w > max_dim:
        scale = max_dim / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image

def get_best_detection(results, class_name, min_conf=0.5):
    """
    특정 클래스에서 가장 높은 신뢰도의 탐지 결과를 반환합니다.
    """
    preds = results[0].boxes
    best_pred = None
    max_conf = -1
    for pred in preds:
        cls = int(pred.cls)
        conf = float(pred.conf)
        
        # 클래스 이름 매핑
        if yolo_model.names[cls] == class_name and conf > max_conf and conf >= min_conf:
            max_conf = conf
            best_pred = pred
            
    return best_pred




import re



def extract_car_info_from_text(ocr_text):
    """
    OCR 텍스트에서 Maker, VIN, Year, Model 추출 (사전 없이 모델 회수 강화).
    - Year: YYYY.MM 또는 YYYY 로만 반환
    - VIN: 허용외 문자 제거 + O->0, I->1 보정 후 17자만 채택
    - Model: '차명/모델명/차 명/모델 명' 라인 + 주변 2~3줄 + 전역 단독 토큰 히유리스틱으로 회수(사전 사용 안 함)
    - Maker: 텍스트에서 찾고, 없으면 VIN WMI로 보정 : wmi_to_maker VIN의 앞자리 3글자를 따와서 제조사 추정함 : 제조사 패턴은 maker_patterns에 있음  
    """
    import re

    lines = [line.strip() for line in ocr_text.split('\n') if line.strip()]
    extracted = {"VIN": "", "Maker": "", "Year": "", "Model": ""}

    # ---------- Maker 패턴 & WMI ----------
    maker_patterns = [
        (r'KIA|기아', '기아'),
        (r'HYUNDAI|현대', '현대'),
        (r'CHEVROLET|쉐보레|GM\s*KOREA|지엠|GENERAL', 'GM'),
        (r'MERCEDES|BENZ|메르세데스|벤츠|다임러', '메르세데스벤츠'),
        (r'\bBMW\b|BMW\s*AG|비엠더블유코리아', 'BMW'),
        (r'RENAULT|르노|삼성', '르노'),
        (r'SSANGYONG|쌍용|쌍용자?', '쌍용'),
        (r'GENESIS|제네시스', '제네시스'),
        (r'TOYOTA|토요타|도요타', 'TOYOTA'),
        (r'\bLEXUS\b|렉서스', 'LEXUS'),
        (r'랜드로버|재규어', '랜드로버'),
        (r'ROLLS|ROYCE|롤스로이스', '롤스로이스'),
        (r'MASERATI', '마세라티'),
        (r'볼보', '볼보'),
        (r'Audi|아우디|BENTLEY', '아우디'),
        (r'Automobili|Volkswagen', '폭스바겐'),
        (r'FCA|Chrysler', '크라이슬러'),
        (r'테슬라', '테슬라'),
        (r'FERRARI', '페라리'),
        (r'포드|FORD', '포드'),
        (r'포르쉐|Porsche', '포르쉐'),
        (r'CITROEN', '푸조'),
        (r'혼다', '혼다'),
        
    ]

    wmi_to_maker = {
        'KNA':'기아','KNB':'기아','KNC':'기아','KND':'기아','KNE':'기아','KNR':'기아','KNT':'기아',
        'KMH':'현대','KMF':'현대','KMX':'현대','5NP':'현대','KMJ':'현대', 'KMT':'현대', 'KNC':'현대',
        'KLY':'GM','KL1':'GM','KL2':'GM','KLA':'GM','KL3':'GM','KL4':'GM','KL5':'GM','1GN':'GM','1GY':'GM','1GC':'GM',
        'WDD':'메르세데스벤츠','W1K':'메르세데스벤츠','W1N':'메르세데스벤츠', 'WDC':'메르세데스벤츠', 'WDB':'메르세데스벤츠', 
        'WBA':'BMW','WBX':'BMW', 'WBS':'BMW',
        'VF1':'르노','KNM':'르노',
        'KPT':'쌍용',
        'JTJ':'TOYOTA','JTH':'TOYOTA', 'JTM':'TOYOTA',  #렉서스는 코드 동일하여 TOYOTA 로 통합.
        'SCA':'롤스로이스',
        'SAL':'랜드로버',
        'ZAM':'마세라티','ZN6':'마세라티',
        '1C4':'크라이슬러',
        'LVY':'볼보', 'YV1':'볼보',
        'WAU':'아우디', 'ZHW':'아우디','SJA':'아우디',
        '5YJ':'테슬라',
        'ZFF':'페라리',
        'WF0':'포드', '1FA':'포드', '1FM':'포드', '2FA':'포드', '2FM':'포드',
        'WP0':'포르쉐',
        'ZPB':'폭스바겐', 'WVW':'폭스바겐', '1VW':'폭스바겐', 'WVG':'폭스바겐', 'W0L':'폭스바겐', '3VW':'폭스바겐', 'W0L':'폭스바겐', 
        'VF7':'푸조',
        '1HG':'혼다', '2HG':'혼다','3HG':'혼다',
        'JNK':'닛산','JNN':'닛산',
        'JA3':'미쓰비시','JAL':'미쓰비시',
        'JM1':'마쓰다',
        
    }

    def maker_from_text(text):
        for pat, label in maker_patterns:
            if re.search(pat, text, flags=re.IGNORECASE):
                return label
        return ""

    # ---------- 유틸: VIN/Year/Model 정규화 ----------
    def clean_vin_candidate(s: str) -> str:
        s = re.sub(r'[^A-Za-z0-9]', '', s).upper()
        s = s.replace('O', '0').replace('I', '1')
        return s if re.fullmatch(r'[A-HJ-NPR-Z0-9]{17}', s) else ""
    
    def normalize_year(val: str, full_text: str) -> str:
        import re
        from datetime import datetime

        now = datetime.now()
        curr_year, curr_month = now.year, now.month

        # 1) 값 자체에서 "YYYY.MM / YYYY-MM / YYYY/MM / YYYY MM / YYYY년 MM월" 모두 허용
        m = re.search(
            r'(20\d{2})'                 # 연도
            r'(?:\s*년)?'                # '년' (옵션)
            r'(?:\s*[.\-/]\s*|\s+)?'     # 구분자(./- 또는 공백) (옵션)
            r'(1[0-2]|0?[1-9])'          # 월 (두 자리 우선)
            r'(?:\s*월)?',               # '월' (옵션)
            val
        )
        if m:
            y, mo = int(m.group(1)), int(m.group(2))
            if y > curr_year or (y == curr_year and mo > curr_month):
                return ""  # 미래값 가드
            return f"{y}.{mo:02d}"

        # 2) 값 자체에서 "YYYY년" 또는 단독 "YYYY"
        m = re.search(r'\b((?:19|20)\d{2})\s*년\b', val)
        if m:
            y = int(m.group(1))
            return "" if y > curr_year else str(y)

        m = re.search(r'\b(19|20)\d{2}\b', val)
        if m:
            y = int(m.group(0))
            return "" if y > curr_year else str(y)

        # 3) 전체 텍스트 폴백: 키워드 주변/일반 패턴 모두 탐색
        patterns = [
            # 키워드 근처에서 연-월 (년/월 표기 포함) 캡처
            r'(?:제작\s*(?:연월|년월|월|연도|년도)|MFD|MFG\.?|Manufactured(?:\s*Date)?)\s*[:\-]?\s*'
            r'(20\d{2})(?:\s*년)?(?:\s*[.\-/]\s*|\s+)?(1[0-2]|0?[1-9])(?:\s*월)?',

            # 구분자/공백 기반 연-월
            r'(20\d{2})(?:\s*년)?(?:\s*[.\-/]\s*|\s+)?(1[0-2]|0?[1-9])(?:\s*월)?',

            # "YYYY년" 또는 단독 "YYYY"
            r'\b((?:19|20)\d{2})\s*년\b',
            r'\b(19|20)\d{2}\b',
        ]
        for pat in patterns:
            m = re.search(pat, full_text, flags=re.IGNORECASE)
            if m:
                # 연-월 패턴
                if len(m.groups()) >= 2 and m.group(2):
                    y, mo = int(m.group(1)), int(m.group(2))
                    if y > curr_year or (y == curr_year and mo > curr_month):
                        return ""  # 미래값 가드
                    return f"{y}.{mo:02d}"
                # 연도만 패턴
                year_str = m.group(1) if (len(m.groups()) >= 1 and len(m.group(1)) == 4) else m.group(0)
                y = int(year_str)
                return "" if y > curr_year else str(y)

        return ""





    def clean_model_value(val: str) -> str:
        # '차종: 승용', '승용', 선행 '차명/모델명' 토큰 제거
        val = re.sub(r'(차종\s*[:：]?\s*\S+)|\b승용\b|(?:차\s*명|모델\s*명)\s*[:：]?', '', val, flags=re.IGNORECASE).strip()
        return val[:40].strip()

    # ---------- 0) Maker 전역 탐색 ----------
    extracted["Maker"] = maker_from_text(ocr_text)

    # ---------- 키워드 ----------
    vin_keywords   = ['차대번호', 'vin', 'v.i.n']
    year_keywords  = ['제작년도', '제작연도', '제작연월', '연월', '년월', '제작월', '제작년월']
    model_keywords = ['차명', '모델명']  # '차 종'은 제외

    # ---------- 0.5) Model 전역 1차 캡처 (공백 허용: '차 명', '모델 명')
    if not extracted["Model"]:
        m = re.search(r'(차\s*명|모델\s*명)\s*[:：]?\s*([^\n\r]+)', ocr_text, flags=re.IGNORECASE)
        if m:
            extracted["Model"] = clean_model_value(m.group(2))

    # ---------- 1) 라인 기반 파싱 ----------
    for i, line in enumerate(lines):
        low = line.lower()
        nospace = re.sub(r'\s+', '', line).lower()

        # VIN
        if not extracted["VIN"]:
            for kw in vin_keywords:
                if kw in low or kw in nospace:
                    val = line.split(':', 1)[-1].strip() if ':' in line else line.split(kw, 1)[-1].strip()
                    vin = clean_vin_candidate(val) or (clean_vin_candidate(lines[i+1]) if i+1 < len(lines) else "")
                    if vin:
                        extracted["VIN"] = vin
                        break

        # Year
        if not extracted["Year"]:
            for kw in year_keywords:
                if kw in low or re.sub(r'\s+', '', kw) in nospace:
                    val = line.split(':', 1)[-1].strip() if ':' in line else line.split(kw, 1)[-1].strip()
                    if not val and i + 1 < len(lines): val = lines[i+1].strip()
                    norm = normalize_year(val, ocr_text)
                    if norm:
                        extracted["Year"] = norm
                        break

        # Model (공백 허용 키워드: '차 명', '모델 명')
        if not extracted["Model"]:
            if re.search(r'(차\s*명|모델\s*명)', line):
                mline = re.search(r'(차\s*명|모델\s*명)\s*[:：]?\s*([^\n]+)', line, flags=re.IGNORECASE)
                val = mline.group(2) if mline else (line.split(':', 1)[-1].strip() if ':' in line else "")
                if not val and i + 1 < len(lines): val = lines[i+1].strip()
                val = clean_model_value(val)
                if val: extracted["Model"] = val

        # 키워드가 있었는데 같은 줄에서 못 잡으면, 다음 2~3줄 내 단독 토큰 보조 회수
        if not extracted["Model"] and re.search(r'(차\s*명|모델\s*명)', line):
            for la in lines[i+1:i+4]:
                la_clean = clean_model_value(la)
                if la_clean and re.fullmatch(r'[가-힣A-Za-z0-9\- ]{2,20}', la_clean):
                    extracted["Model"] = la_clean
                    break

    # ---------- 2) VIN 전역 폴백 ----------
    if not extracted["VIN"]:
        for cand in re.findall(r'[A-Za-z0-9\-\s]{16,20}', ocr_text):
            vin = clean_vin_candidate(cand)
            if vin:
                extracted["VIN"] = vin
                break

    # ---------- 3) Year 전역 폴백 ----------
    if not extracted["Year"]:
        extracted["Year"] = normalize_year("", ocr_text)

    # ---------- 3.5) Model 전역 히유리스틱 폴백 (사전 없이) ----------
    if not extracted["Model"]:
        # 후보: 한글/영문/숫자 2~15자 단독 토큰 라인 (일반 단위/필드 제외)
        banned = r'(제작|차량총중량|타이어|공기압|림|전축|후축|적차|psi|kg|변속기|차축비|외장|내장|이\s*자동차는|대한민국|자동차\s*관리법령|적합|제작되었습니다)'
        cand = []
        for idx, ln in enumerate(lines):
            # 괄호 등 제거 후 평가
            s = re.sub(r'[\(\)\[\]\{\},:：]', ' ', ln)
            s = re.sub(r'\s+', ' ', s).strip()
            s_strip = re.sub(r'[^가-힣A-Za-z0-9\- ]', '', s)
            if re.search(banned, s_strip, flags=re.IGNORECASE):
                continue
            # 단독 토큰(공백 거의 없음) 위주
            if re.fullmatch(r'[가-힣]{2,15}', s_strip) or re.fullmatch(r'[A-Za-z0-9\-]{2,15}', s_strip):
                cand.append((idx, s_strip))

        if cand:
            # '차명/모델명/차종'이 마지막으로 등장한 라인과의 거리로 우선순위
            last_key = None
            for idx, ln in enumerate(lines):
                if re.search(r'(차\s*명|모델\s*명|차\s*종|차종)', ln):
                    last_key = idx
            if last_key is not None:
                cand.sort(key=lambda t: abs(t[0] - last_key))
            extracted["Model"] = cand[0][1]

    # ---------- 4) Maker WMI 보정 ----------
    if not extracted["Maker"] and extracted["VIN"]:
        extracted["Maker"] = wmi_to_maker.get(extracted["VIN"][:3], "")

    return extracted


# 박스 그리기 및 이미지 저장 헬퍼 함수
def save_image_with_boxes(image, detections, file_path, save_dir, filename):
    """
    탐지된 객체에 박스를 그리고 이미지를 새로운 폴더에 저장합니다.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"📦 저장 폴더 생성: {save_dir}")
        
    for det in detections:
        x1, y1, x2, y2 = [int(val) for val in det.xyxy[0]]
        label = yolo_model.names[int(det.cls)]
        
        color = (255, 0, 255) if label == 'sticker_area' else (0, 0, 255)  # 보라색: (255, 0, 255), 빨간색: (0, 0, 255)
        
        # 박스 그리기
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        # 라벨 텍스트 추가
        text = f"{label}: {det.conf[0]*100:.2f}%"
        cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 파일명 변경 (예: 'image.jpg' -> 'image_box.jpg')
    base, ext = os.path.splitext(filename)
    new_filename = f"{base}_box{ext}"
    save_path = os.path.join(save_dir, new_filename)
    
    cv2.imwrite(save_path, image)
    print(f"🖼️ 박스가 그려진 이미지를 저장했습니다: {new_filename}")


def process_and_extract_info(image_path, min_conf_vin=0.8, min_conf_sticker=0.5):
    """
    메인 워크플로우를 실행하고 최종 정보를 반환합니다.
    """
    print(f"\n--- 이미지 처리 시작: {os.path.basename(image_path)} ---")
    
    # 1. 이미지 로드 및 전처리
    image_np = cv2.imread(image_path)
    if image_np is None:
        print(f"❌ 이미지 파일을 읽을 수 없습니다: {image_path}")
        return None, "Image Read Error"

    image_resized = resize_image_if_needed(image_np)
    
    # 2. YOLOv8 탐지
    results = yolo_model.predict(source=image_resized, save=False, verbose=False)
    
    vin_pred = get_best_detection(results, 'vin_area', min_conf_vin)
    sticker_pred = get_best_detection(results, 'sticker_area', min_conf_sticker)
    
    cropped_image = None
    crop_info = ""

    # 3. 신뢰도에 따른 이미지 영역 선택 (Sticker Area 우선)
    if sticker_pred:
        print(f"✅ 'sticker_area' 탐지 성공 (신뢰도: {sticker_pred.conf[0]*100:.2f}%)")
        x1, y1, x2, y2 = [int(val) for val in sticker_pred.xyxy[0]]
        cropped_image = image_resized[y1:y2, x1:x2]
        crop_info = "Sticker Area"

        # VIN 영역이 Sticker 영역 안에 있는지 추가 확인
        if vin_pred:
            print(f"  > 'vin_area'도 성공적으로 탐지됨 (신뢰도: {vin_pred.conf[0]*100:.2f}%)")
        else:
            print("  > 'vin_area'는 탐지되지 않았습니다.")
    
    else:
        print("❌ 'sticker_area' 탐지 실패. 전체 이미지 사용")
        cropped_image = image_resized
        crop_info = "Full Image"
        
        # 전체 이미지에서 VIN 영역 탐지 여부 확인
        if vin_pred:
            print(f"  > 전체 이미지에서 'vin_area'가 탐지됨 (신뢰도: {vin_pred.conf[0]*100:.2f}%)")

    if cropped_image.size == 0:
        print("❌ 잘라낸 이미지의 크기가 0입니다. 전체 이미지를 사용합니다.")
        cropped_image = image_resized
        
    #== 박스 결과 시각화 검토용 코드==
    # 4. 박스쳐진 이미지 다른이름저장.. 탐지된 모든 박스를 리스트로 모읍니다.
    # all_detections = []
    # if sticker_pred:
    #     all_detections.append(sticker_pred)
    # if vin_pred:
    #     all_detections.append(vin_pred)
    # 박스가 탐지되었을 경우에만 이미지 저장 함수를 호출합니다.
    # if all_detections:
    #     # 파일명과 저장 경로를 준비합니다.
    #     filename = os.path.basename(image_path)
    #     image_dir_save = f"{image_dir}/box"
    #     # 박스 그리기 및 저장 함수 호출
    #     save_image_with_boxes(image_resized.copy(), all_detections, image_path, image_dir_save, filename)
    #==================================================================================================
    
    # '''
    # 4. GCV API로 텍스트 인식
    try:
        # 4-1) sticker_area 먼저 OCR
        _, encoded_image = cv2.imencode('.jpg', cropped_image)
        image_for_gcv = vision.Image(content=encoded_image.tobytes())
        response = client.text_detection(image=image_for_gcv)
        texts = response.text_annotations

        car_info = None
        all_text = ""

        if texts:
            all_text = texts[0].description
            # print(f"✅ GCV OCR(sticker) 텍스트:\n{all_text}")  # 디버깅용
            car_info = extract_car_info_from_text(all_text)
        else:
            print("⚠️ GCV가 sticker 영역에서 텍스트를 인식하지 못했습니다.")

        # 4-2) VIN 폴백: sticker에서 VIN이 비면 vin_area만 따로 재시도
        if (car_info is None) or (not car_info.get("VIN")):
            if vin_pred is not None:
                x1v, y1v, x2v, y2v = [int(v) for v in vin_pred.xyxy[0]]
                vin_crop = image_resized[y1v:y2v, x1v:x2v]

                # 글자 키우기(안전한 업샘플)
                try:
                    vin_crop = cv2.resize(vin_crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
                except:
                    pass

                _, vin_enc = cv2.imencode('.jpg', vin_crop)
                vin_image_for_gcv = vision.Image(content=vin_enc.tobytes())
                vin_resp = client.text_detection(image=vin_image_for_gcv)
                if vin_resp.text_annotations:
                    vin_text = vin_resp.text_annotations[0].description
                    # print(f"🔁 GCV OCR(vin_area) 텍스트:\n{vin_text}")  # 디버깅용
                    vin_only = extract_car_info_from_text(vin_text)
                    if car_info is None:
                        car_info = {"VIN": "", "Maker": "", "Year": "", "Model": ""}
                    if vin_only.get("VIN"):
                        car_info["VIN"] = vin_only["VIN"]
                        # Maker/Year/Model은 sticker 쪽 결과가 있다면 그대로 두고,
                        # 없을 때만 vin_only 결과로 보완
                        if not car_info.get("Maker"):
                            car_info["Maker"] = vin_only.get("Maker", "")
                        if not car_info.get("Year"):
                            car_info["Year"] = vin_only.get("Year", "")
                        if not car_info.get("Model"):
                            car_info["Model"] = vin_only.get("Model", "")
                        print("🔁 VIN 폴백: vin_area OCR 결과로 VIN 보완 완료.")
                else:
                    print("⚠️ vin_area 폴백에서도 텍스트를 인식하지 못했습니다.")
            else:
                print("ℹ️ vin_area 탐지가 없어 폴백을 수행하지 못했습니다.")

        if car_info:
            print("--- 추출 결과 ---")
            print(f"사용된 영역: {crop_info}")
            for key, value in car_info.items():
                print(f"  - {key}: {value}")
            return car_info, "Success"
        else:
            return None, "No Text Detected"

    except Exception as e:
        print(f"❌ GCV API 호출 중 오류 발생: {e}")
        return None, str(e)
    # '''
    

# --- 메인 실행 코드 ---
if __name__ == "__main__":
    # 사용자 환경에 맞게 경로를 설정하세요.
    image_dir = r"C:\01_Coding\250801_CAR_OCR_PHOTO\2_VIN_OCR\RQMT\250825_VIN_0825\기아_VIN사진"
    # test_images = ['1 2009 BMW.jpg',]
    # test_images = ['BENZ1.jpg','BENZ2.jpg','BMW1.jpg','FCA1.jpg','GM1.jpg','RENAULT1.jpg','TOYOTA1.jpg']

    print("--- 통합 워크플로우 시작 ---")
    # 폴더 내 모든 이미지 파일 목록 가져오기
    # .jpg, .jpeg, .png 파일만 처리하도록 설정
    all_files = os.listdir(image_dir)
    test_images = []
    i = 0
    while i < len(all_files):
        filename = all_files[i]
        # 파일 확장자 검사
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            test_images.append(filename)
        # 5개 이미지만 처리하고 종료
        # 테스트 이미지 개수를 제한하고 싶을 때 사용
        if len(test_images) > 2:
            print("설정한 개수 만큼 이미지 리스트 추가")
            break
        i += 1
    if not test_images:
        print(f"❌ '{image_dir}' 폴더에 처리할 이미지 파일이 없습니다.")
    for filename in test_images:
        test_path = os.path.join(image_dir, filename)
        process_and_extract_info(test_path)
    
    print("\n--- 통합 워크플로우 종료 ---")