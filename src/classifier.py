import pathlib
import pickle
from sys import argv

from skautfestlib import load_data, extract_mfcc, load_wav


def load_classifier_or_lda(name: str):
    """Loads sklearn classifier or LDA saved by pickle.dump in wb mode"""
    with pathlib.Path(".").joinpath(f"{name}.pk").open("rb") as f:
        classifier = pickle.load(f)
    return classifier


def test_user_wav(file: str):
    assert file.endswith(".wav")

    user_data = load_wav(pathlib.Path(file))
    user_mfccs = extract_mfcc(user_data)

    lda = load_classifier_or_lda("lda")
    lda_user_mfccs = lda.transform(user_mfccs)

    bayes = load_classifier_or_lda("bayes_classifier")
    svm = load_classifier_or_lda("svm_classifier")

    bp = bayes.predict_proba(lda_user_mfccs)
    sp = svm.predict_proba(lda_user_mfccs)

    # Prettify the output to percentages with 4 decimal numbers
    pretty_bp = [round(value * 100, 4) for sublist in bp for value in sublist]
    pretty_sp = [round(value * 100, 4) for sublist in sp for value in sublist]

    print(f"{file} - {bayes.classes_} - Bayes: {pretty_bp}, SVM: {pretty_sp}\n")


def test_user_dir(dir: str):
    for file in pathlib.Path(dir).glob("*.wav"):
        test_user_wav(dir + file.absolute().name) # Pass the whole path


if __name__ == "__main__":
    if len(argv) > 1: # Files passed as arguments
        for arg in argv[1:]:
            p = pathlib.Path(arg)
            if p.is_dir(): test_user_dir(arg)
            if p.is_file(): test_user_wav(arg)
    else:
        test_user_wav("../dataset/user/vojta.wav")