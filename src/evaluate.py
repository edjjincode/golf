import os
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import precision_score, recall_score, f1_score
from data_loader import load_data   # src 경로에서 명시적으로 가져오기

def evaluate_model(model_path, data_dir):
    # 테스트 데이터 로드
    test_images, test_labels = load_data(data_dir)

    # 저장된 모델 로드
    model = load_model(model_path)

    # 평가 (Loss와 Accuracy)
    loss, accuracy = model.evaluate(test_images, test_labels, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # 예측값 계산
    predictions = model.predict(test_images)
    predicted_labels = np.argmax(predictions, axis=1)  # 예측 클래스 (argmax 사용)

    # Precision, Recall, F1 Score 계산
    precision = precision_score(test_labels, predicted_labels, average='weighted')
    recall = recall_score(test_labels, predicted_labels, average='weighted')
    f1 = f1_score(test_labels, predicted_labels, average='weighted')

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

if __name__ == "__main__":
    # 데이터 디렉토리 및 모델 경로 설정
    test_data_dir = '/root/robo_db_new/data/test'
    model_path = '/root/robo_db_new/models/mobilenet_golf_model.h5'

    # 평가 함수 호출
    evaluate_model(model_path=model_path, data_dir=test_data_dir)
