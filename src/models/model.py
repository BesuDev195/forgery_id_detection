import os
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from acqusition.acquisition import load_image
from preprocessing.enhancement import enhance_image
from preprocessing.restoration import restore_image
from preprocessing.morphology import apply_morphology
from segmentation.segmentation import segment_image
from features.features import extract_features


def process_image(path):
    img = load_image(path)
    img = enhance_image(img)
    img = restore_image(img)
    img = apply_morphology(img)
    img = segment_image(img)

    features = extract_features(img)
    return features


def load_dataset(base_path="dataset"):
    data = []
    labels = []

    for label, folder in enumerate(["real", "fake"]):
        path = os.path.join(base_path, folder)

        if not os.path.isdir(path):
            raise FileNotFoundError(f"Dataset path not found: {path}")

        for file in os.listdir(path):
            img_path = os.path.join(path, file)

            if not file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                continue

            try:
                features = process_image(img_path)
                data.append(features)
                labels.append(label)
            except Exception as e:
                print(f"Skipping '{img_path}' ({e})")
                continue

    cnt = Counter(labels)
    print(f"Loaded dataset: total={len(data)}, distribution={cnt}")

    if len(data) == 0:
        raise ValueError("No images were loaded from dataset. Please check dataset folder and image paths.")

    return data, labels


def train_model():
    data, labels = load_dataset()

    label_counts = Counter(labels)
    if len(label_counts) < 2:
        raise ValueError(f"Need at least two classes in dataset to train. Found: {label_counts}")

    model = RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=100)

    if len(data) < 10:
        print("Very small dataset: training on all data without test split.")
        model.fit(data, labels)
        print("Warning: model validation not reliable with very few samples.")
        return model

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            data,
            labels,
            test_size=0.2,
            stratify=labels,
            random_state=42,
        )
    except ValueError as e:
        print(f"train_test_split stratify failed ({e}), retrying without stratify.")
        X_train, X_test, y_train, y_test = train_test_split(
            data,
            labels,
            test_size=0.2,
            random_state=42,
        )

    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    print("Accuracy:", acc)

    return model


def predict(model, image_path):
    features = process_image(image_path)
    result = model.predict([features])[0]

    return "REAL ✅" if result == 0 else "FAKE ❌"