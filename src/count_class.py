import os
import json
from collections import Counter

def count_classes(data_dir):
    # annotations.coco.json 파일 경로
    annotations_file = os.path.join(data_dir, "_annotations.coco.json")
    
    # JSON 파일 읽기
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)
    
    # 클래스 ID를 카운트
    category_counts = Counter([ann['category_id'] for ann in annotations['annotations']])

    # 클래스 ID와 이름 매핑
    category_mapping = {cat['id']: cat['name'] for cat in annotations['categories']}

    # 클래스별 개수 출력
    for class_id, count in category_counts.items():
        class_name = category_mapping.get(class_id, "Unknown")
        print(f"Class '{class_name}' (ID: {class_id}): {count} instances")
    
    return category_counts

if __name__ == "__main__":
    # 각 데이터셋 경로
    train_dir = '/root/robo_db_new/data/train'
    val_dir = '/root/robo_db_new/data/valid'
    test_dir = '/root/robo_db_new/data/test'

    print("Class counts in Train dataset:")
    count_classes(train_dir)
    print("\nClass counts in Validation dataset:")
    count_classes(val_dir)
    print("\nClass counts in Test dataset:")
    count_classes(test_dir)
