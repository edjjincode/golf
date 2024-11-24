import os
from data_loader import load_data
from model import create_model

def train_model(data_dir, save_path, num_classes, epochs=10):
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'valid')

    # 데이터 로드
    train_images, train_labels = load_data(train_dir)
    val_images, val_labels = load_data(val_dir)

    # 모델 생성
    model = create_model(num_classes)

    # 모델 학습
    model.fit(train_images, train_labels, validation_data=(val_images, val_labels), epochs=epochs)

    # 모델 저장
    model.save(save_path)
    print(f"Model saved at {save_path}")

if __name__ == "__main__":
    train_model(data_dir='/root/robo_db_new/data', 
                save_path='/root/robo_db_new/models/mobilenet_golf_model.h5', num_classes=9, epochs=20)
