# -*- coding: utf-8 -*-
"""
AI 이미지 분류기 모델 평가 스크립트
다양한 이미지에 대한 모델 성능을 자동으로 평가합니다.
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from transformers import pipeline
import torch

class ModelEvaluator:
    def __init__(self, model_path=None):
        """모델 평가기 초기화"""
        print("모델 평가기 초기화 중...")
        
        # 모델 로드
        if model_path and os.path.exists(model_path):
            print(f"로컬 모델 사용: {model_path}")
            self.classifier = pipeline('image-classification', model=model_path)
        else:
            print("사전 훈련된 모델 사용")
            self.classifier = pipeline('image-classification', model='dima806/ai_vs_real_image_detection')
        
        # 결과 저장용
        self.results = []
        self.predictions = []
        self.true_labels = []
        
    def preprocess_image(self, image_path, target_size=(32, 32)):
        """이미지를 모델 입력 크기로 전처리"""
        try:
            image = Image.open(image_path).convert('RGB')
            original_size = image.size
            resized_image = image.resize(target_size, Image.Resampling.LANCZOS)
            return resized_image, original_size
        except Exception as e:
            print(f"이미지 전처리 오류 {image_path}: {e}")
            return None, None
    
    def evaluate_single_image(self, image_path, true_label):
        """단일 이미지 평가"""
        try:
            # 이미지 전처리
            processed_image, original_size = self.preprocess_image(image_path)
            if processed_image is None:
                return None
            
            # 예측 수행
            result = self.classifier(processed_image)
            predicted_label = result[0]['label']
            confidence = result[0]['score']
            
            # 결과 저장
            result_data = {
                'image_path': str(image_path),
                'true_label': true_label,
                'predicted_label': predicted_label,
                'confidence': confidence,
                'correct': true_label == predicted_label,
                'original_size': original_size,
                'processed_size': target_size
            }
            
            self.results.append(result_data)
            self.predictions.append(predicted_label)
            self.true_labels.append(true_label)
            
            return result_data
            
                    except Exception as e:
            print(f"이미지 평가 오류 {image_path}: {e}")
            return None
    
    def evaluate_dataset(self, data_path):
        """전체 데이터셋 평가"""
        print(f"데이터셋 평가 시작: {data_path}")
        
        data_path = Path(data_path)
        
        # 실제 이미지 평가
        real_path = data_path / 'real_images'
        if real_path.exists():
            print("실제 이미지 평가 중...")
            for img_file in real_path.rglob('*'):
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                    self.evaluate_single_image(img_file, 'REAL')
        
        # AI 생성 이미지 평가
        ai_path = data_path / 'ai_generated'
        if ai_path.exists():
            print("AI 생성 이미지 평가 중...")
            for img_file in ai_path.rglob('*'):
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                    self.evaluate_single_image(img_file, 'FAKE')
        
        print(f"평가 완료: 총 {len(self.results)}개 이미지")
    
    def calculate_metrics(self):
        """성능 지표 계산"""
        if not self.predictions or not self.true_labels:
            print("평가 결과가 없습니다.")
            return None
        
        # 기본 지표 계산
        accuracy = accuracy_score(self.true_labels, self.predictions)
        precision = precision_score(self.true_labels, self.predictions, pos_label='FAKE')
        recall = recall_score(self.true_labels, self.predictions, pos_label='FAKE')
        f1 = f1_score(self.true_labels, self.predictions, pos_label='FAKE')
        
        # 혼동 행렬
        cm = confusion_matrix(self.true_labels, self.predictions, labels=['REAL', 'FAKE'])
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm.tolist(),
            'total_images': len(self.results),
            'correct_predictions': sum(1 for r in self.results if r['correct']),
            'incorrect_predictions': sum(1 for r in self.results if not r['correct'])
        }
        
        return metrics
    
    def generate_report(self, output_path='evaluation_report'):
        """평가 보고서 생성"""
        print("평가 보고서 생성 중...")
        
        # 성능 지표 계산
        metrics = self.calculate_metrics()
        if metrics is None:
            return
        
        # 결과를 DataFrame으로 변환
        df = pd.DataFrame(self.results)
        
        # 보고서 디렉토리 생성
        report_dir = Path(output_path)
        report_dir.mkdir(exist_ok=True)
        
        # 1. 텍스트 보고서
        report_file = report_dir / f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("AI 이미지 분류기 평가 보고서\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"평가 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"총 이미지 수: {metrics['total_images']}\n")
            f.write(f"정확한 예측: {metrics['correct_predictions']}\n")
            f.write(f"잘못된 예측: {metrics['incorrect_predictions']}\n\n")
            
            f.write("성능 지표:\n")
            f.write(f"  정확도 (Accuracy): {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)\n")
            f.write(f"  정밀도 (Precision): {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)\n")
            f.write(f"  재현율 (Recall): {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)\n")
            f.write(f"  F1 점수: {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)\n\n")
            
            f.write("혼동 행렬:\n")
            f.write("        예측\n")
            f.write("실제    REAL  FAKE\n")
            f.write(f"REAL    {metrics['confusion_matrix'][0][0]:4d}  {metrics['confusion_matrix'][0][1]:4d}\n")
            f.write(f"FAKE    {metrics['confusion_matrix'][1][0]:4d}  {metrics['confusion_matrix'][1][1]:4d}\n\n")
            
            # 오분류 사례 분석
            incorrect = df[df['correct'] == False]
            if len(incorrect) > 0:
                f.write("오분류 사례 (상위 10개):\n")
                for idx, row in incorrect.head(10).iterrows():
                    f.write(f"  {row['image_path']}: {row['true_label']} -> {row['predicted_label']} (신뢰도: {row['confidence']:.3f})\n")
        
        # 2. JSON 보고서
        json_file = report_dir / f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump({
                'metrics': metrics,
                'detailed_results': self.results
            }, f, ensure_ascii=False, indent=2)
        
        # 3. CSV 보고서
        csv_file = report_dir / f"detailed_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        
        # 4. 시각화
        self.create_visualizations(report_dir, metrics)
        
        print(f"평가 보고서가 생성되었습니다: {report_dir}")
        return report_dir
    
    def create_visualizations(self, output_dir, metrics):
        """시각화 생성"""
        try:
            # 1. 혼동 행렬 히트맵
            plt.figure(figsize=(8, 6))
            cm = np.array(metrics['confusion_matrix'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['REAL', 'FAKE'], 
                       yticklabels=['REAL', 'FAKE'])
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. 성능 지표 막대 그래프
            plt.figure(figsize=(10, 6))
            metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
            metrics_values = [metrics['accuracy'], metrics['precision'], 
                            metrics['recall'], metrics['f1_score']]
            
            bars = plt.bar(metrics_names, metrics_values, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
            plt.title('Model Performance Metrics')
            plt.ylabel('Score')
            plt.ylim(0, 1)
            
            # 값 표시
            for bar, value in zip(bars, metrics_values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{value:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(output_dir / 'performance_metrics.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 3. 신뢰도 분포
            df = pd.DataFrame(self.results)
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            correct_conf = df[df['correct'] == True]['confidence']
            incorrect_conf = df[df['correct'] == False]['confidence']
            
            plt.hist(correct_conf, bins=20, alpha=0.7, label='Correct', color='green')
            plt.hist(incorrect_conf, bins=20, alpha=0.7, label='Incorrect', color='red')
            plt.xlabel('Confidence Score')
            plt.ylabel('Frequency')
            plt.title('Confidence Distribution')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            df['confidence'].hist(bins=20, alpha=0.7, color='skyblue')
            plt.xlabel('Confidence Score')
            plt.ylabel('Frequency')
            plt.title('Overall Confidence Distribution')
            
            plt.tight_layout()
            plt.savefig(output_dir / 'confidence_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"시각화 생성 오류: {e}")

def main():
    parser = argparse.ArgumentParser(description='AI 이미지 분류기 모델 평가')
    parser.add_argument('--data_path', type=str, default='./evaluation_data',
                       help='평가 데이터 경로')
    parser.add_argument('--model_path', type=str, default=None,
                       help='모델 경로 (없으면 사전 훈련된 모델 사용)')
    parser.add_argument('--output_path', type=str, default='./evaluation_report',
                       help='결과 출력 경로')
    
    args = parser.parse_args()
    
    # 평가기 초기화
    evaluator = ModelEvaluator(args.model_path)
    
    # 데이터셋 평가
    evaluator.evaluate_dataset(args.data_path)
    
    # 보고서 생성
    evaluator.generate_report(args.output_path)
    
    print("평가가 완료되었습니다!")

if __name__ == '__main__':
    main()