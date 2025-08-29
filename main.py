"""OCR 메인 실행 파일"""
import os
import multiprocessing
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from pdf2image import convert_from_path  # type: ignore
from src.tesseract.run_tesseract import process_single_image

#################
# 상수
#################
TESSERACT_OCR_MODE_LIST = ["image_to_string", "image_to_data"]
TESSERACT_OCR_MODE = TESSERACT_OCR_MODE_LIST[1]

PDF_TO_IMG_DPI = 1200
MAX_WORKERS = 4  # 병렬 처리 최대 스레드 수

FILES_LIST = [
    ('./assets/인보이스.pdf'),
    # ('./assets/car_numberpad.png'),
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


def convert_pdf_to_images(pdf_path, dpi=PDF_TO_IMG_DPI):
    """PDF를 이미지 리스트로 변환"""
    try:
        print(f"PDF 변환 중: {pdf_path}")
        pages = convert_from_path(pdf_path, dpi=dpi, fmt='RGB')
        print(f"총 {len(pages)} 페이지 발견")
        return pages
    except Exception as e:
        print(f"PDF 변환 오류: {e}")
        return None


def process_pdf_parallel(pdf_path, output_dir, ocr_mode, max_workers=4):
    """PDF 페이지들을 병렬로 처리하는 함수"""
    print("📄 PDF 파일로 인식됨")
    
    # PDF를 이미지로 변환
    images = convert_pdf_to_images(pdf_path)
    if images is None:
        return None
    
    print(f"🔄 {len(images)} 페이지를 병렬 처리 시작 (최대 {max_workers} 스레드)")
    
    # 병렬 처리를 위한 작업 목록 생성
    tasks = []
    for idx, image in enumerate(images, 1):
        page_info = f"페이지 {idx}"
        file_prefix = f"page_{idx:03d}"
        tasks.append((image, output_dir, ocr_mode, page_info, file_prefix))
    
    # 병렬 처리 실행
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 모든 작업 제출
        future_to_page = {
            executor.submit(process_single_image, *task): task[3] 
            for task in tasks
        }
        
        # 완료된 작업들 처리
        for future in as_completed(future_to_page):
            page_info = future_to_page[future]
            try:
                result = future.result()
                if result['success']:
                    print(f"✅ {page_info} 처리 완료:")
                    print(f"   📄 OCR 결과: {result['output_file']}")
                    print(f"   🖼️ 원본 이미지: {result['original_image']}")
                    if result.get('boxed_image'):
                        print(f"   📦 바운딩 박스 이미지: {result['boxed_image']}")
                    print(f"   🔧 처리된 이미지: {result['processed_image']}")
                else:
                    print(f"❌ {page_info} 처리 실패: {result['error']}")
                results.append(result)
            except Exception as e:
                print(f"❌ {page_info} 처리 중 예외 발생: {e}")
                results.append({'success': False, 'error': str(e), 'page_info': page_info})
    
    # 결과 요약
    success_count = sum(1 for r in results if r['success'])
    print(f"🎯 처리 완료: {success_count}/{len(results)} 페이지 성공")
    
    return output_dir if success_count > 0 else None


def process_image_file(image_path, output_dir, ocr_mode):
    """단일 이미지 파일을 처리하는 함수"""
    print("🖼️ 이미지 파일로 인식됨")
    
    filename = os.path.splitext(os.path.basename(image_path))[0]
    page_info = filename
    file_prefix = filename
    
    result = process_single_image(image_path, output_dir, ocr_mode, page_info, file_prefix)
    
    if result['success']:
        print(f"✅ 처리 완료:")
        print(f"   📄 OCR 결과: {result['output_file']}")
        print(f"   🖼️ 원본 이미지: {result['original_image']}")
        if result.get('boxed_image'):
            print(f"   📦 바운딩 박스 이미지: {result['boxed_image']}")
        print(f"   🔧 처리된 이미지: {result['processed_image']}")
        return output_dir
    else:
        print(f"❌ 처리 실패: {result['error']}")
        return None


def process_file(file_path, ocr_mode=TESSERACT_OCR_MODE, max_workers=4):
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
        result = process_pdf_parallel(file_path, output_dir, ocr_mode, max_workers)
    elif is_image_file(file_path):
        result = process_image_file(file_path, output_dir, ocr_mode)
    else:
        print(f"❌ 지원하지 않는 파일 형식: {file_path}")
        return None

    return result


def main():
    """메인 실행 함수"""
    start_time = datetime.now()
    
    # CPU 코어 수의 절반을 워커 수로 설정
    max_workers = max(1, multiprocessing.cpu_count())

    print("=== OCR 처리 시작 ===")
    print(f"🔧 병렬 처리 설정: 최대 {max_workers} 스레드 (CPU 코어 수: {multiprocessing.cpu_count()})")
    for file_path in FILES_LIST:
        result = process_file(file_path, max_workers=max_workers)

        # 결과 확인
        if result:
            print(f"✅ 처리 완료! 결과 저장 위치: {result}")
        else:
            print(f"❌ 처리 실패: {file_path}")

    end_time = datetime.now()
    elapsed_time = end_time - start_time
    print(f"\n === 모든 OCR 처리 완료 === ")
    print(f"⏱️ 총 소요시간: {elapsed_time.total_seconds():.2f}초")


if __name__ == "__main__":
    main()
