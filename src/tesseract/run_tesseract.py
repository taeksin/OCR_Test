"""OCR 처리를 위한 모듈"""
import os
import json
import cv2  # type: ignore
import pytesseract  # type: ignore
from PIL import Image  # type: ignore
import numpy as np

# Set tesseract path if needed (common Windows paths)
if os.name == 'nt':  # Windows
    possible_paths = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
    ]
    for path in possible_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            break

# LANGUAGE = "kor+eng"
LANGUAGE = "eng"

def enhance_image_quality(image):
    """이미지 품질을 향상시키는 함수"""
    # PIL Image를 numpy array로 변환
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # 그레이스케일 변환
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # type: ignore
    else:
        gray = image
    
    # 노이즈 제거
    denoised = cv2.medianBlur(gray, 3)  # type: ignore
    
    # 대비 향상 (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # type: ignore
    enhanced = clahe.apply(denoised)
    
    # 샤프닝
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)  # type: ignore
    
    return sharpened


def ocr_with_string_mode(image, lang=LANGUAGE):
    """image_to_string 모드로 OCR 수행"""
    return pytesseract.image_to_string(image, lang=lang)


def ocr_with_data_mode(image, lang=LANGUAGE):
    """image_to_data 모드로 OCR 수행"""
    data = pytesseract.image_to_data(image, lang=lang, output_type=pytesseract.Output.DICT)
    return data


def save_string_result(text, output_file, page_info=""):
    """문자열 결과를 파일로 저장"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"=== {page_info} OCR 결과 (String Mode) ===\n\n")
        f.write(text)


def save_data_result(data, output_file, page_info=""):
    """데이터 결과를 파일로 저장"""
    # JSON 형태로 저장
    json_file = output_file.replace('.txt', '_data.json')
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    # 텍스트 형태로도 저장 (가독성을 위해)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"=== {page_info} OCR 결과 (Data Mode) ===\n\n")
        
        # 신뢰도가 높은 텍스트만 추출
        texts = []
        confidences = data['conf']
        words = data['text']
        
        for i, (conf, word) in enumerate(zip(confidences, words)):
            if conf > 0 and word.strip():  # 신뢰도 30 이상인 단어만
                texts.append(f"[신뢰도: {conf}] {word}")
        
        f.write("=== 추출된 텍스트 (신뢰도 30 이상) ===\n")
        f.write('\n'.join(texts))
        
        f.write(f"\n\n=== 상세 데이터 ===\n")
        f.write(f"총 감지된 요소 수: {len(data['text'])}\n")
        f.write(f"평균 신뢰도: {sum(confidences)/len(confidences):.2f}\n")
        f.write(f"상세 데이터는 {json_file} 파일을 참조하세요.\n")


def process_single_image(image_input, output_dir, ocr_mode, page_info="", file_prefix="image"):
    """단일 이미지에 대해 OCR을 수행하는 함수"""
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 이미지 처리
        if isinstance(image_input, str):
            # 이미지 파일 경로인 경우
            image = cv2.imread(image_input)  # type: ignore
            if image is None:
                raise ValueError(f"이미지를 로드할 수 없습니다: {image_input}")
        else:
            # PIL Image 객체인 경우 (PDF 페이지)
            image = image_input
        
        # 이미지 품질 향상
        enhanced_image = enhance_image_quality(image)
        
        # OCR 모드에 따라 처리
        if ocr_mode == "image_to_string":
            result = ocr_with_string_mode(enhanced_image)
            output_file = os.path.join(output_dir, f"{file_prefix}_string.txt")
            save_string_result(result, output_file, page_info)
            
        elif ocr_mode == "image_to_data":
            result = ocr_with_data_mode(enhanced_image)
            output_file = os.path.join(output_dir, f"{file_prefix}_data.txt")
            save_data_result(result, output_file, page_info)
            
        else:
            raise ValueError(f"지원하지 않는 OCR 모드: {ocr_mode}")
        
        # 처리된 이미지도 저장
        processed_image_file = os.path.join(output_dir, f"{file_prefix}_processed.png")
        cv2.imwrite(processed_image_file, enhanced_image)  # type: ignore
        
        return {
            'success': True,
            'output_file': output_file,
            'processed_image': processed_image_file,
            'page_info': page_info
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'page_info': page_info
        }