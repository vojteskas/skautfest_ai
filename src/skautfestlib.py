import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
import librosa
import pathlib
import sys

SAMPLE_RATE = 16000


def load_wav(file_path: pathlib.Path, print_file_name=True):
    """Loads waveform from a single wav file"""

    if print_file_name:
        sys.stderr.write(f"Loading {file_path.name}\n")
    
    waveform, _ = librosa.load(file_path, sr=SAMPLE_RATE)

    # Trim silence
    non_silent_intervals = librosa.effects.split(waveform, top_db=10)
    y_non_silent = []
    for interval in non_silent_intervals:
        y_non_silent.extend(waveform[interval[0] : interval[1]])

    return [(np.array(y_non_silent))]


def load_data(dir_path):
    """Loads .wav sound files for processing"""

    sys.stderr.write(f"Loading wav files in {dir_path}, this may take a while...\n")

    waveforms = [] # Where the loaded sounds will be stored

    p = pathlib.Path(dir_path)
    for file_path in p.glob("*.wav"):
        # Add current loaded file to return
        waveforms = waveforms + load_wav(file_path, False)
    
    return waveforms


def extract_mfcc(waveforms, n=30):
    """Extracts n MFCCs from provided waveforms"""

    sys.stderr.write("Extracting MFCCs\n")

    mfccs = [] # List of extracted mfccs to return

    l = len(waveforms) # For progress printout
    last_thr = 0

    for i, w in enumerate(waveforms):
        mfcc = librosa.feature.mfcc(y=w, sr=SAMPLE_RATE, n_mfcc=n)
        mfccs.append(np.mean(mfcc, axis=1))
        
        # Progress printout
        progress_percentage = (i + 1) / l * 100
        if progress_percentage >= last_thr + 10:
            last_thr = int(progress_percentage // 10) * 10
            sys.stderr.write(f"MFCC extraction progress: {progress_percentage:.0f}%\n")
    
    return np.array(mfccs)


def train_bayes(features, labels) -> GaussianNB:
    """Trains Naive Bayes classifier on features using labels"""
    bayes_classifier = GaussianNB()
    bayes_classifier.fit(features, labels)
    return bayes_classifier


def train_svm(features, labels) -> svm.SVC:
    """Trains SVM clasifier on features using labels"""
    svm_classifier = svm.SVC(probability=True)
    svm_classifier.fit(features, labels)
    return svm_classifier