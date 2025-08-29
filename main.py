"""OCR 메인 실행 파일"""
import os
from datetime import datetime
from pdf2image import convert_from_path  # type: ignore
from src.tesseract.run_tesseract import ocr_images

#################
# 상수
#################
TESSERACT_OCR_MODE_LIST = ["image_to_string", "image_to_data"]
TESSERACT_OCR_MODE = TESSERACT_OCR_MODE_LIST[1]

FILES_LIST = [
    ('./assets/인보이스.pdf'),
    ('./assets/car_numberpad.png'),  # 이미지 파일 예시
]

# --------------------------------------------------


def get_output_directory(file_path, base_dir="test_result"):
    """실행 시간과 파일명을 기반으로 출력 디렉토리 생성"""
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    filename = os.path.splitext(os.path.basename(file_path))[0]
    return os.path.join(base_dir, f"{timestamp}_{filename}")


def is_pdf_file(file_path):
    """PDF 파일인지 확인"""
    return file_path.lower().endswith('.pdf')


def is_image_file(file_path):
    """이미지 파일인지 확인"""
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
    return any(file_path.lower().endswith(ext) for ext in image_extensions)


def convert_pdf_to_images(pdf_path, dpi=1200):
    """PDF를 이미지 리스트로 변환"""
    try:
        print(f"PDF 변환 중: {pdf_path}")
        pages = convert_from_path(pdf_path, dpi=dpi, fmt='RGB')
        print(f"총 {len(pages)} 페이지 발견")
        return pages
    except Exception as e:
        print(f"PDF 변환 오류: {e}")
        return None


def process_file(file_path, ocr_mode=TESSERACT_OCR_MODE):
    """파일을 처리하는 함수"""
    print(f"\n📁 처리 대상: {file_path}")
    print(f"🔧 OCR 모드: {ocr_mode}")

    # 파일 존재 확인
    if not os.path.exists(file_path):
        print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
        return None

    # 출력 디렉토리 생성
    output_dir = get_output_directory(file_path)

    # 파일 타입에 따라 처리
    if is_pdf_file(file_path):
        print("📄 PDF 파일로 인식됨")
        # PDF를 이미지로 변환
        images = convert_pdf_to_images(file_path)
        if images is None:
            return None

        # 이미지 리스트와 함께 OCR 처리
        result = ocr_images(images, output_dir, ocr_mode, is_pdf=True)

    elif is_image_file(file_path):
        print("🖼️ 이미지 파일로 인식됨")
        # 단일 이미지를 리스트로 만들어서 처리
        result = ocr_images([file_path], output_dir, ocr_mode, is_pdf=False)

    else:
        print(f"❌ 지원하지 않는 파일 형식: {file_path}")
        return None

    return result


def main():
    """메인 실행 함수"""

    print("=== OCR 처리 시작 ===")

    for file_path in FILES_LIST:
        result = process_file(file_path)

        # 결과 확인
        if result:
            print(f"✅ 처리 완료! 결과 저장 위치: {result}")
        else:
            print(f"❌ 처리 실패: {file_path}")

    print("\n=== 모든 OCR 처리 완료 ===")


if __name__ == "__main__":
    main()
