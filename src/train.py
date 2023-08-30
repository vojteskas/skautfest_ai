import numpy as np
import pathlib
import pickle
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
import sys

from skautfestlib import load_data, extract_mfcc, train_bayes, train_svm

def train():
    """Main training function for both bayessian and svm classifier using skautfestlib"""
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
    lda = LinearDiscriminantAnalysis(n_components=1) # Only two classes, so reducing to one dimension
    lda.fit(d_train, l_train)
    lda_d_train = lda.transform(d_train)
    lda_d_test = lda.transform(d_test)

    # Train models
    bayes = train_bayes(lda_d_train, l_train)
    bayes_score = bayes.score(lda_d_test, l_test)
    svmcl = train_svm(lda_d_train, l_train)
    svmcl_score = svmcl.score(lda_d_test, l_test)
    sys.stderr.write(f"ACCURACCIES - Bayes: {bayes_score:.4f}, SVM: {svmcl_score:.4f}\n")
    
    # Save models and lda
    with pathlib.Path(".").joinpath("lda.pk").open("wb") as f:
        pickle.dump(lda, f)
    with pathlib.Path(".").joinpath("bayes_classifier.pk").open("wb") as f:
        pickle.dump(bayes, f)
    with pathlib.Path(".").joinpath("svm_classifier.pk").open("wb") as f:
        pickle.dump(svmcl, f)

if __name__ == "__main__":
    train()