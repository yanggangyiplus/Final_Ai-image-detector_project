# -*- coding: utf-8 -*-
"""
간단한 모델 평가 스크립트
웹 애플리케이션을 통해 개별 이미지를 테스트할 수 있습니다.
"""

import os
import json
from pathlib import Path
from datetime import datetime
from transformers import pipeline
from PIL import Image

def evaluate_image(image_path, model_path=None):
    """단일 이미지 평가"""
    try:
        # 모델 로드
        if model_path and os.path.exists(model_path):
            classifier = pipeline('image-classification', model=model_path)
        else:
            classifier = pipeline('image-classification', model='dima806/ai_vs_real_image_detection')
        
        # 이미지 전처리 (32x32로 리사이즈)
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        resized_image = image.resize((32, 32), Image.Resampling.LANCZOS)
        
        # 예측 수행
        result = classifier(resized_image)
        predicted_label = result[0]['label']
        confidence = result[0]['score']
        
        return {
            'image_path': str(image_path),
            'predicted_label': predicted_label,
            'confidence': confidence,
            'original_size': original_size,
            'processed_size': (32, 32),
            'success': True
        }
        
    except Exception as e:
        return {
            'image_path': str(image_path),
            'error': str(e),
            'success': False
        }

def batch_evaluate(folder_path, true_label, model_path=None):
    """폴더 내 모든 이미지 배치 평가"""
    folder_path = Path(folder_path)
    results = []
    
    print(f"폴더 평가 시작: {folder_path}")
    print(f"예상 라벨: {true_label}")
    
    # 지원되는 이미지 확장자
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    # 폴더 내 모든 이미지 파일 찾기
    image_files = []
    for ext in image_extensions:
        image_files.extend(folder_path.glob(f'*{ext}'))
        image_files.extend(folder_path.glob(f'*{ext.upper()}'))
    
    print(f"발견된 이미지 수: {len(image_files)}")
    
    # 각 이미지 평가
    for i, image_file in enumerate(image_files, 1):
        print(f"평가 중... ({i}/{len(image_files)}) {image_file.name}")
        result = evaluate_image(image_file, model_path)
        result['true_label'] = true_label
        result['correct'] = result.get('predicted_label') == true_label if result.get('success') else False
        results.append(result)
    
    return results

def analyze_results(results):
    """결과 분석"""
    if not results:
        print("분석할 결과가 없습니다.")
        return
    
    total = len(results)
    successful = sum(1 for r in results if r.get('success', False))
    correct = sum(1 for r in results if r.get('correct', False))
    
    print(f"\n=== 평가 결과 ===")
    print(f"총 이미지 수: {total}")
    print(f"성공적으로 처리된 이미지: {successful}")
    print(f"정확한 예측: {correct}")
    print(f"정확도: {correct/successful*100:.2f}%" if successful > 0 else "정확도: 0%")
    
    # 신뢰도 통계
    confidences = [r.get('confidence', 0) for r in results if r.get('success', False)]
    if confidences:
        print(f"평균 신뢰도: {sum(confidences)/len(confidences):.3f}")
        print(f"최고 신뢰도: {max(confidences):.3f}")
        print(f"최저 신뢰도: {min(confidences):.3f}")
    
    # 오분류 사례
    incorrect = [r for r in results if r.get('success', False) and not r.get('correct', False)]
    if incorrect:
        print(f"\n오분류 사례 ({len(incorrect)}개):")
        for result in incorrect[:5]:  # 상위 5개만 표시
            print(f"  {Path(result['image_path']).name}: {result['true_label']} -> {result['predicted_label']} (신뢰도: {result['confidence']:.3f})")

def main():
    print("AI 이미지 분류기 간단 평가 도구")
    print("=" * 40)
    
    while True:
        print("\n선택하세요:")
        print("1. 단일 이미지 평가")
        print("2. 폴더 배치 평가")
        print("3. 종료")
        
        choice = input("선택 (1-3): ").strip()
        
        if choice == '1':
            image_path = input("이미지 경로를 입력하세요: ").strip()
            if os.path.exists(image_path):
                result = evaluate_image(image_path)
                if result['success']:
                    print(f"\n예측 결과: {result['predicted_label']}")
                    print(f"신뢰도: {result['confidence']:.3f}")
                    print(f"원본 크기: {result['original_size']}")
                else:
                    print(f"오류: {result['error']}")
            else:
                print("파일을 찾을 수 없습니다.")
        
        elif choice == '2':
            folder_path = input("폴더 경로를 입력하세요: ").strip()
            true_label = input("실제 라벨을 입력하세요 (REAL 또는 FAKE): ").strip().upper()
            
            if os.path.exists(folder_path) and true_label in ['REAL', 'FAKE']:
                results = batch_evaluate(folder_path, true_label)
                analyze_results(results)
                
                # 결과 저장
                save = input("결과를 파일로 저장하시겠습니까? (y/n): ").strip().lower()
                if save == 'y':
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f"evaluation_results_{timestamp}.json"
                    with open(filename, 'w', encoding='utf-8') as f:
                        json.dump(results, f, ensure_ascii=False, indent=2)
                    print(f"결과가 {filename}에 저장되었습니다.")
            else:
                print("폴더가 존재하지 않거나 라벨이 올바르지 않습니다.")
        
        elif choice == '3':
            print("평가를 종료합니다.")
            break
        
        else:
            print("잘못된 선택입니다.")

if __name__ == '__main__':
    main()
