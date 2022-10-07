#!/usr/bin/env python
"""WSD disambiguation for the Senseval-3 English lexical sample task."""

import csv
import glob
import logging
import statistics

from typing import List, Tuple

from sklearn.feature_extraction import text  # type: ignore
from sklearn import linear_model  # type: ignore
from sklearn import preprocessing


TRAIN_TSV = "data/train/*.tsv"


def extract_features(path: str) -> Tuple[List[str], List[str]]:
    labels: List[str] = []
    features: List[str] = []
    with open(path, "r") as source:
        tsv_reader = csv.reader(source, delimiter="\t")
        for label, context in tsv_reader:
            context = context.replace("_ ", "")
            features.append(context)
            labels.append(label)
    return features, labels


def main() -> None:
    logging.basicConfig(format="%(levelname)s %(message)s", level="INFO")
    correct: List[int] = []
    size: List[int] = []
    for train_path in glob.iglob(TRAIN_TSV):
        feature_vectors, y = extract_features(train_path)
        vectorizer = text.CountVectorizer()
        encoder = preprocessing.LabelEncoder()
        x = vectorizer.fit_transform(feature_vectors)
        model = linear_model.LogisticRegression(penalty="l1", C=10, solver="liblinear")
        model.fit(x, y)
        test_path = train_path.replace("/train/", "/test/")
        feature_vectors, y = extract_features(test_path)
        x = vectorizer.transform(feature_vectors)
        yhat = model.predict(x)
        assert len(y) == len(yhat), "mismatched lengths"
        correct.append(sum(y == yhat))
        size.append(len(y))
    # Accuracies.
    logging.info("Micro-average accuracy:\t%.4f", sum(correct) / sum(size))
    accuracies = [c / s for (c, s) in zip(correct, size)]
    logging.info("Macro-average accuracy:\t%.4f", statistics.mean(accuracies))


if __name__ == "__main__":
    main()
