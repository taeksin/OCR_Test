import cv2  # type: ignore
import pytesseract  # type: ignore
import os
from datetime import datetime
from pdf2image import convert_from_path  # type: ignore
from PIL import Image  # type: ignore
import numpy as np

def enhance_image_quality(image):
    """이미지 품질을 향상시키는 함수"""
    # PIL Image를 numpy array로 변환
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # 그레이스케일 변환
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # 노이즈 제거
    denoised = cv2.medianBlur(gray, 3)
    
    # 대비 향상 (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    
    # 샤프닝
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    
    return sharpened

def ocr_pdf_pages(pdf_path, output_base_dir="test_result"):
    """PDF 페이지별로 OCR을 수행하는 함수"""
    
    # 실행 시간 기반 폴더명 생성
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    filename = os.path.splitext(os.path.basename(pdf_path))[0]
    output_dir = os.path.join(output_base_dir, f"{timestamp}_{filename}")
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # PDF를 고해상도 이미지로 변환 (300 DPI)
        print(f"PDF 변환 중: {pdf_path}")
        pages = convert_from_path(pdf_path, dpi=300, fmt='RGB')
        
        print(f"총 {len(pages)} 페이지 발견")
        
        # 각 페이지별로 OCR 수행
        for page_num, page_image in enumerate(pages, 1):
            print(f"페이지 {page_num} 처리 중...")
            
            # 이미지 품질 향상
            enhanced_image = enhance_image_quality(page_image)
            
            # OCR 수행 (한국어 + 영어)
            text = pytesseract.image_to_string(enhanced_image, lang='kor+eng')
            
            # 결과를 텍스트 파일로 저장
            output_file = os.path.join(output_dir, f"page_{page_num:03d}.txt")
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"=== 페이지 {page_num} OCR 결과 ===\n\n")
                f.write(text)
            
            # 처리된 이미지도 저장 (선택사항)
            image_file = os.path.join(output_dir, f"page_{page_num:03d}_processed.png")
            cv2.imwrite(image_file, enhanced_image)
            
            print(f"페이지 {page_num} 완료: {output_file}")
        
        print(f"\n모든 페이지 처리 완료!")
        print(f"결과 저장 위치: {output_dir}")
        
    except Exception as e:
        print(f"오류 발생: {e}")
        return None
    
    return output_dir

if __name__ == "__main__":
    # Set tesseract path if needed (common Windows paths)
    if os.name == 'nt':  # Windows
        possible_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            r'C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'.format(os.getenv('USERNAME'))
        ]
        for path in possible_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                break
    
    # PDF 파일 경로
    pdf_path = './assets/인보이스.pdf'
    
    # PDF OCR 실행
    result_dir = ocr_pdf_pages(pdf_path)
    
    if result_dir:
        print(f"\n✅ OCR 완료! 결과는 '{result_dir}' 폴더에 저장되었습니다.")
    else:
        print("❌ OCR 처리 중 오류가 발생했습니다.")