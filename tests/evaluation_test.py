import pandas as pd
import jsonlines
import pytest
import os
from src.train_with_simpletransformers import get_confusion_matrix, get_span_recall


@pytest.fixture(scope="session")
def bad_sample():
    return os.path.join(os.path.dirname(__file__), "fixtures", "bad_model_gold_and_predictions.tsv")


@pytest.fixture(scope="session")
def good_sample():
    return os.path.join(os.path.dirname(__file__), "fixtures", "good_model_gold_and_predictions.tsv")


@pytest.fixture(scope="session")
def labels():
    return [f"B-MUS", "B-PER", f"I-MUS", f"I-PER", "O"]


def convert_dev_from_json_to_conll(dev_path, conll_path):
    with jsonlines.open(dev_path, "r") as fin:
        with open(conll_path, "w") as fout:
            for line in fin:
                tokens = line["sent_items"]
                print(tokens)
                print([f"{t[0]} {t[1]}\n" for t in tokens])
                for t in tokens:
                    fout.write(f"{t[0]} {t[1]}\n")
                    
                    
def get_set_labels(set_path):
    set_labels = []
    with open(set_path, "r") as f:
        for t in f.readlines():
            if t.strip():
                label = t.split()[-1]
                set_labels.append(label)
            else:
                print(t)
    return set_labels


def get_golds_and_predictions(filepath):
    df = pd.read_csv(filepath, sep="\t", names=["gold", "pred"])
    golds = df["gold"].tolist()
    preds = df["pred"].tolist()
    return golds, preds


def test_bad_model_spans_eval(bad_sample):
    report, true_positive_spans, total_gold_positives = get_span_recall(bad_sample, "MUS", "PER")
    assert float(report["B-MUS"][2]) == 0.6, report
    true_positives = true_positive_spans.values[0]
    assert true_positives == 0
    assert true_positives / total_gold_positives.values[0] == 0, f"{true_positive_spans}, {total_gold_positives}"


def test_good_model_spans_eval(good_sample):
    report, true_positive_spans, total_gold_positives =  get_span_recall(good_sample, "MUS", "PER")
    assert float(report["B-MUS"][2]) == 1.00, report
    true_positives = true_positive_spans.values[0]
    assert true_positives == 4
    assert true_positives / total_gold_positives.values[0] == 1, f"{true_positive_spans}, {total_gold_positives}"


def test_good_confusion_matrix(good_sample, labels):
    golds, preds = get_golds_and_predictions(good_sample)
    matrix = get_confusion_matrix(golds, preds, labels)
    assert list(matrix.diagonal()) == [1.0, 0.75, 1.0, 0.889, 1.0]  # normalised


def test_bad_confusion_matrix( bad_sample, labels):
    golds, preds = get_golds_and_predictions(bad_sample)
    matrix = get_confusion_matrix(golds, preds, labels)
    assert list(matrix.diagonal()) == [0.75, 0.583, 0.5, 0.778, 0.994]  # normalised