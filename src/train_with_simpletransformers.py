import logging

import pandas as pd
from simpletransformers.ner import NERModel
import re
import json
import jsonlines
import argparse

from sklearn.metrics import confusion_matrix
import numpy as np
import shutil

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


def set_args():
    return {
            "seed": 42,
            "labels_list": list(LABELS.values()),
            "reprocess_input_data": True,
            "overwrite_output_dir": True,
            "train_batch_size": BATCH_SIZE,
            "eval_batch_size": BATCH_SIZE,
            "num_train_epochs": EPOCHS,
            "save_eval_checkpoints": False,
            "save_steps": -1,
            "use_multiprocessing": False,
            "use_multiprocessing_for_evaluation": False,
            "evaluate_during_training": True,
            "evaluate_during_training_steps": 50,
            "evaluate_during_training_verbose": True,
            "fp16": False,
            "wandb_project": WANDB_NAME,
            "learning_rate": 0.0003,
            "warmup_ratio": 0.1,
            "logging_steps": 1,
        }


def ingest_data_to_df(filepath, label_map):
    tagged_data = []
    sentence_number = 0
    with open(filepath, 'r') as f:
        for line in f.readlines():
            for tagged_word in line.split():
                try:
                    word, tag = tagged_word.split("-[", 1)
                    old_tag = tag.split("]", 1)[0]
                    new_tag = label_map[old_tag]
                    tagged_data.append([sentence_number, word, new_tag])
                except Exception as e:
                    raise Exception((tagged_word, line), e)
            sentence_number += 1
    return pd.DataFrame(tagged_data, columns=["sentence_id", "words", "labels"])


def get_dataset_parts(dataset_path, dev_name, train_name, test_name):
    devpath, trainpath, testpath = f"{dataset_path}/{dev_name}.txt", f"{dataset_path}/{train_name}.txt", f"{dataset_path}/{test_name}.txt"
    train = ingest_data_to_df(trainpath, LABELS)
    test = ingest_data_to_df(testpath, LABELS)
    dev = ingest_data_to_df(devpath, LABELS)
    return dev, train, test


def untag_dev_sentences(devpath):
    output_path = f"sentences_{devpath}"
    clean_sentences = []
    with open(devpath, "r") as fin:
        for line in fin.readlines():
            sentence = re.sub('\-\[[A-Z]*\]',  '', line)
            clean_sentences.append(sentence)
    return clean_sentences


# new data handling:
def import_jsons_to_df(dataset_path, filename):
    fp = f"{dataset_path}/{filename}.jsonl"
    tagged_data = []
    with jsonlines.open(fp, 'r') as f:
        for line in f:
            for token in line["sent_items"]:
                tagged_data.append([line["id"], token[0], token[1]])
    return pd.DataFrame(tagged_data, columns=["sentence_id", "words", "labels"])


def untagged_sentences(fp):
    clean = []
    with jsonlines.open(fp, "r") as f:
        for line in f:
            words = [w[0] for w in line["sent_items"]]
            clean.append(" ".join(words))
    return clean


def configure_model(args, model_type="roberta", model_name="roberta-base"):
    model = NERModel(
        model_type,
        model_name,
        args=args
    )
    return model


def send_prediction_to_conll(predictions, experiments_path):
    with open(f"{experiments_path}/predictions/{EXPERIMENT_NAME}.conll", "w") as f:
        for sent in predictions:
            for token_dict in sent:
                for k, v in token_dict.items():
                    f.write(f"{k} {v}\n")


def document_results(experiments_path, experiment_name, outputs):
    details = dict()
    with open(f"{experiments_path}/{experiment_name}.json", "w") as f:
        with open(f"{outputs}/best_model/model_args.json", "r") as args:
            details["model_args"] = json.load(args)
        with open(f"{outputs}/best_model/config.json", "r") as conf:
            details["best_model_conf"] = json.load(conf)
        with open(f"{outputs}/best_model/eval_results.txt", "r") as res:
            details["eval_results"] = dict()
            for r in res.readlines():
                k, v = r.split(" = ")
                details["eval_results"][k] = v
        json.dump(details, f)


def print_confusion_matrix(model):
    gold_tags = []
    pred_tags = []

    with jsonlines.open(f"{args.dataset_path}/{DEV}.jsonl", "r") as f:
        for line in f:
            words = [w[0] for w in line["sent_items"]]
            tokens = [w[1] for w in line["sent_items"]]
            predictions, raw_outputs = model.predict([" ".join(words)])
            if len(words) != len(predictions[0]):
                print(line)
            else:
                gold_tags.extend(tokens)
                pred_tags.extend([list(x.values())[0] for x in predictions[0]])

    matrix = confusion_matrix(gold_tags, pred_tags, labels=['B-MUS', 'B-PER', 'I-MUS', 'I-PER', 'O'], normalize="true")
    return np.round(matrix, 3)


def print_results(model):
    matrix = print_confusion_matrix(model)
    print("confusion matrix: \n", matrix)

    with open(f"./outputs/best_model/eval_results.txt", "r") as f:
        print("eval results: \n", f.read())


def main():
    model_args = set_args()
    outputs = "./outputs"
    experiment_name = f"{args.wandb_project}-{args.experiment_suffix}"
    dev_set = import_jsons_to_df(args.dataset_path, DEV)
    train_set = import_jsons_to_df(args.dataset_path, TRAIN)
    model = configure_model(args=model_args)
    model.train_model(train_set, eval_data=dev_set)
    clean_sentences = untagged_sentences(f"{args.dataset_path}/{DEV}.jsonl")
    predictions, raw_outputs = model.predict(clean_sentences)
    document_results(EXPERIMENTS, experiment_name, outputs)
    send_prediction_to_conll(predictions, EXPERIMENTS)
    print_results(model)
    outputs_dir = f"{outputs}/best_model"
    best_models_dir = f"./best_models/{args.experiment_name}"
    shutil.move(outputs_dir, best_models_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    EXPERIMENTS = '../experiments'
    BATCH_SIZE = 64
    EPOCHS = 3
    LABELS = {"B": "B-MUS", "PB": "B-PER", "I": "I-MUS", "PI": "I-PER", "O": "O"}
    DEV, TRAIN = "split_dev_only_hearst_uniques", "split_train_only_hearst_uniques"
    parser.add_argument('--wandb_project', help='', default="only_hearst_uniques")
    parser.add_argument('--experiment_suffix', help='', default="../data")
    parser.add_argument('--dataset_path', help='', default="../data/musicians_dataset")
    parser.add_argument('--version_name', help='', default="all_without_person")
    parser.add_argument('--suffix', help='', default="_unique")

    args = parser.parse_args()
    main()