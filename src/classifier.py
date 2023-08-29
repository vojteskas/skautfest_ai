from math import floor
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.model_selection import train_test_split
import librosa
import pathlib

SAMPLE_RATE = 16000


def load_data(files_path):
    """Loads .wav sound files for processing"""
    waveforms = [] # Where the loaded sounds will be stored

    p = pathlib.Path(files_path)
    for file_path in p.glob("*.wav"):
        waveform, _ = librosa.load(file_path, sr=SAMPLE_RATE)

        # Trim silence
        non_silent_intervals = librosa.effects.split(waveform, top_db=10)
        y_non_silent = []
        for interval in non_silent_intervals:
            y_non_silent.extend(waveform[interval[0] : interval[1]])

        # Add current loaded file to return
        waveforms.append((np.array(y_non_silent)))
    
    return waveforms


def extract_mfcc(waveforms):
    print("Extracting MFCCs")
    mfccs = []

    l = len(waveforms)
    last_thr = 0

    for i, w in enumerate(waveforms):
        mfcc = librosa.feature.mfcc(y=w, sr=SAMPLE_RATE, n_mfcc=30)
        mfccs.append(np.mean(mfcc, axis=1))
        progress_percentage = (i + 1) / l * 100
        if progress_percentage >= last_thr + 10:
            last_thr = int(progress_percentage // 10) * 10
            print(f"MFCC extraction progress: {progress_percentage:.0f}%")
    
    return np.array(mfccs)


def train_bayes(features, labels):
    bayes_classifier = GaussianNB()
    bayes_classifier.fit(features, labels)
    return bayes_classifier


def train_svm(features, labels):
    svm_classifier = svm.SVC(probability=True)
    svm_classifier.fit(features, labels)
    return svm_classifier


if __name__ == "__main__":
    # Data preparation
    dataset_location = "../dataset/"
    females_data = load_data(dataset_location + "females")
    males_data = load_data(dataset_location + "males")
    females_labels = np.full(len(females_data), "female")
    males_labels = np.full(len(males_data), "male")

    data = females_data + males_data # ladies first
    labels = np.concatenate((females_labels, males_labels))

    # Feature extraction
    mfccs = extract_mfcc(data)
    
    # Split training and testing data
    d_train, d_test, l_train, l_test = train_test_split(mfccs, labels)

    # Reduce dimensions using Linear Discriminant Analysis
    lda = LinearDiscriminantAnalysis(n_components=1)
    lda.fit(d_train, l_train)
    lda_d_train = lda.transform(d_train)
    lda_d_test = lda.transform(d_test)

    # Train model
    bayes = train_bayes(lda_d_train, l_train)
    bayes_score = bayes.score(lda_d_test, l_test)
    svmcl = train_svm(lda_d_train, l_train)
    svmcl_score = svmcl.score(lda_d_test, l_test)
    print(f"Bayes: {bayes_score}, SVM: {svmcl_score}")
    # Dodělat pickle, ať pak můžem blbnout s vlastníma nahrávkama


    # Classify user recordings
    user_data = load_data(dataset_location)
    print("User recodings num:", len(user_data))
    user_mfccs = extract_mfcc(user_data)
    lda_user_mfccs = lda.transform(user_mfccs)
    bp = bayes.predict_proba(lda_user_mfccs)
    sp = svmcl.predict_proba(lda_user_mfccs)
    print(f"Prediction for user recordings:")
    print(f"Bayes: {bayes.classes_} - {bp}")
    print(f"SVM: {svmcl.classes_} - {sp}")
