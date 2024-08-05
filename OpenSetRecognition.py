import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import minmax_scale
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, recall_score, precision_score, jaccard_score

dataset_dir = r"C:\Users\melis\OneDrive\Masaüstü\VocDetCode\50_speakers_audio_data"
unknown_samples_dir = r"C:\Users\melis\OneDrive\Masaüstü\VocDetCode\unknown_samples"

#feature extraction
def extract_mfcc_features(file_path, n_mfcc=13):
    audio, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfccs = np.mean(mfccs.T, axis=0)
    return mfccs

def load_data_with_unknowns(dataset_dir, unknown_samples_dir):
    all_mfcc_features = []
    labels = []

    if not os.path.exists(unknown_samples_dir):
        os.makedirs(unknown_samples_dir)

    for subdir, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(subdir, file)
                mfcc_features = extract_mfcc_features(file_path)
                all_mfcc_features.append(mfcc_features)
                label = int(os.path.basename(subdir)[-4:])
                labels.append(label)

    for file in os.listdir(unknown_samples_dir):
        file_path = os.path.join(unknown_samples_dir, file)
        mfcc_features = extract_mfcc_features(file_path)
        all_mfcc_features.append(mfcc_features)
        labels.append(-1)

    return np.array(all_mfcc_features), np.array(labels)


#normalizasyon
all_mfcc_features, labels = load_data_with_unknowns(dataset_dir, unknown_samples_dir)
all_mfcc_features_normalized = minmax_scale(all_mfcc_features, feature_range=(0, 1))

#pca dönüşümü
n_components = 10
pca = PCA(n_components=n_components)
all_mfcc_features_pca = pca.fit_transform(all_mfcc_features_normalized)

# Eğitim ve test setlerine ayırma
X_train_val, X_test, y_train_val, y_test = train_test_split(all_mfcc_features_pca, labels, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

#threshold hesaplama
def calculate_threshold(model, X_val, y_val, threshold_fraction=0.95):
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_val)
        max_probs = np.max(probs, axis=1)
        threshold = np.percentile(max_probs, threshold_fraction * 100)
        return threshold
    else:
        raise ValueError("Modelin `predict_proba` metodu yok.")

#hesaplanan threshold ile open set recognition
def open_set_predict(model, X, threshold):
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        max_probs = np.max(probs, axis=1)
        predictions = model.predict(X)
        predictions[max_probs < threshold] = -1  # Bilinmeyen örnekler
    else:
        raise ValueError("Modelin `predict_proba` metodu yok.")
    return predictions

#modeller ve eğitim
models = {
    "SVC": SVC(C=0.5, probability=True, random_state=42),
    "RandomForestClassifier": RandomForestClassifier(n_estimators=10, n_jobs=-1, random_state=42),
    "KNeighborsClassifier": KNeighborsClassifier(n_jobs=-1, n_neighbors=15),
    "GMM": GaussianMixture(n_components=len(set(labels)), random_state=42)
}

for model_name, model in models.items():
    print(f"{model_name} Classifier for Validation Data:")

    if model_name == "GMM":
        model.fit(X_train)
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)
    else:
        model.fit(X_train, y_train)
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)

    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_recall = recall_score(y_val, y_val_pred, average='macro', zero_division=1)
    val_precision = precision_score(y_val, y_val_pred, average='macro', zero_division=1)
    val_intersection = jaccard_score(y_val, y_val_pred, average='macro', zero_division=1)

    print(f"Validation Accuracy: {val_accuracy:.6f}")
    print(f"Validation Recall: {val_recall:.6f}")
    print(f"Validation Precision: {val_precision:.6f}")
    print(f"Validation Intersection: {val_intersection:.6f}")

    print(f"\n{model_name} Classifier for Test Data:")

    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred, average='macro', zero_division=1)
    test_precision = precision_score(y_test, y_test_pred, average='macro', zero_division=1)
    test_intersection = jaccard_score(y_test, y_test_pred, average='macro', zero_division=1)

    print(f"Test Accuracy: {test_accuracy:.6f}")
    print(f"Test Recall: {test_recall:.6f}")
    print(f"Test Precision: {test_precision:.6f}")
    print(f"Test Intersection: {test_intersection:.6f}")
    print()


thresholds = {}
for model_name, model in models.items():
    print(f"{model_name} Classifier with Open Set Recognition:")

    if model_name in ["SVC", "RandomForestClassifier"]:
        model.fit(X_train, y_train)
        threshold = calculate_threshold(model, X_val, y_val)
        thresholds[model_name] = threshold
        y_val_pred = open_set_predict(model, X_val, threshold)
        y_test_pred = open_set_predict(model, X_test, threshold)
    elif model_name == "KNeighborsClassifier":
        model.fit(X_train, y_train)
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)
    else:
        model.fit(X_train)
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)

    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_recall = recall_score(y_val, y_val_pred, average='macro', zero_division=1)
    val_precision = precision_score(y_val, y_val_pred, average='macro', zero_division=1)
    val_intersection = jaccard_score(y_val, y_val_pred, average='macro', zero_division=1)

    print(f"Validation Accuracy: {val_accuracy:.6f}")
    print(f"Validation Recall: {val_recall:.6f}")
    print(f"Validation Precision: {val_precision:.6f}")
    print(f"Validation Intersection: {val_intersection:.6f}")

    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred, average='macro', zero_division=1)
    test_precision = precision_score(y_test, y_test_pred, average='macro', zero_division=1)
    test_intersection = jaccard_score(y_test, y_test_pred, average='macro', zero_division=1)

    print(f"Test Accuracy: {test_accuracy:.6f}")
    print(f"Test Recall: {test_recall:.6f}")
    print(f"Test Precision: {test_precision:.6f}")
    print(f"Test Intersection: {test_intersection:.6f}")
    print()
