import logging

import pandas as pd
from simpletransformers.ner import NERModel
import re
import json
import jsonlines
import argparse

from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import shutil

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


def set_args():
    return {
            "seed": 42,
            "labels_list": LABELS,
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
            "wandb_project": args.wandb_project,
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
    with open(f"{experiments_path}/predictions/{args.wandb_project}-{args.experiment_suffix}.conll", "w") as f:
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
        
        json.dump(details, f, indent=4)
        

def get_golds_and_predictions(model):
    gold_tags = []
    pred_tags = []

    with jsonlines.open(f"{args.dataset_path}/{DEV}.jsonl", "r") as f:
        for line in f:
            words = [w[0] for w in line["sent_items"]]
            tokens = [w[1] for w in line["sent_items"]]
            predictions, raw_outputs = model.predict([" ".join(words)])
            gold_tags.extend(tokens)
            pred_tags.extend([list(x.values())[0] for x in predictions[0]])
            
    return gold_tags, pred_tags


def tag_hits(df, pos, neg):
    neg_labels = ["O", f"B-{neg}", f"I-{neg}"]
    for i, row in df.iterrows():
        if row["match"] == "_":
            if (row["gold"] in neg_labels) and (row["pred"] in neg_labels):
                df.at[i,'match'] = "tn"
            elif (row["gold"] in neg_labels) and (row["pred"].endswith(pos)):
                df.at[i,'match'] = "fp"
            elif (row["gold"].endswith(pos)) and not (row["pred"].endswith(pos)):
                df.at[i,'match'] = "fn"
            elif (row["gold"] == f"B-{pos}") and (row["pred"] == f"B-{pos}"):
                span = []
                x = i+1
                error = False
                while df.at[x, "gold"] == f"I-{pos}":
                    span.append(x)
                    x += 1
                    if df.at[x, "gold"] != df.at[x, "pred"]:
                        error = True                    
                if error:
                    df.at[i,'match'] = "fn"
                else:
                    df.at[i,'match'] = "tp"
                for x in span:
                    df.at[x,'match'] = "--"
                    

def get_span_recall(filepath, positive_label, negative_label):
    df = pd.read_csv(filepath, sep="\t", names=["gold", "pred"])
    golds = df["gold"].tolist()
    preds = df["pred"].tolist()
    report = classification_report(golds, preds)
    report = {x.strip()[0:5]:x.strip()[5:].split() for x in report.split("\n")[2:] if not x.startswith("O")}
    df["match"] = "_"
    tag_hits(df, positive_label, negative_label)
    true_positive_spans = df[df["match"] == "tp"].count()
    total_gold_positives = df[df["gold"] == f"B-{positive_label}"].count()
    return report, true_positive_spans, total_gold_positives


def get_confusion_matrix(gold_tags, pred_tags, labels):
    matrix = confusion_matrix(gold_tags, pred_tags, labels=labels, normalize="true")
    return np.round(matrix, 3)


def main():
    model_args = set_args()
    outputs = "./outputs"
    experiment_name = f"{args.wandb_project}-{args.experiment_suffix}"
    dev_set = import_jsons_to_df(args.dataset_path, DEV)
    train_set = import_jsons_to_df(args.dataset_path, TRAIN)
    model = configure_model(args=model_args)
    model.train_model(train_set, eval_data=dev_set)
    golds, preds = get_golds_and_predictions(model)
    matrix = get_confusion_matrix(golds, preds, LABELS)
    cls_report = classification_report(golds, preds, labels=LABELS)
    document_results(args.experiments_path, experiment_name, outputs)
    outputs_dir = f"{outputs}/best_model"
    best_models_dir = f"./best_models/{args.experiment_name}"
    shutil.move(outputs_dir, best_models_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    BATCH_SIZE = 64
    EPOCHS = 3
    parser.add_argument('--wandb_project', help='', default="only_hearst_uniques")
    parser.add_argument('--experiment_suffix', help='', default="manual")
    parser.add_argument('--dataset_path', help='', default="../data/musicians_dataset")
    parser.add_argument('--version_name', help='', default="all_without_person")
    parser.add_argument('--suffix', help='', default="_unique")
    parser.add_argument('--target_tag', help='', default="MUS")
    parser.add_argument('--superclass_tag', help='', default="PER")
    parser.add_argument('--experiments_path', help='', default='../experiments')
    

    args = parser.parse_args()
    if args.experiment_suffix == "manual":
        DEV, TRAIN = f"dev_converted", f"split_train_{args.version_name}"
    else:
        DEV, TRAIN = f"split_dev_{args.version_name}", f"split_train_{args.version_name}"
    LABELS = [f"B-{args.target_tag}",
              f"B-{args.superclass_tag}",
              f"I-{args.target_tag}",
              f"I-{args.superclass_tag}",
              "O"]
    main()