from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import argparse
from simpletransformers.ner import NERModel
import jsonlines
import numpy as np
import json


def document_results(experiments_path, experiment_name, best_model_dir):
    details = dict()

    with open(f"{experiments_path}/{experiment_name}.json", "w") as f:
        with open(f"{best_model_dir}/model_args.json", "r") as args:
            details["model_args"] = json.load(args)
        with open(f"{best_model_dir}/config.json", "r") as conf:
            details["best_model_conf"] = json.load(conf)
        with open(f"{best_model_dir}/eval_results.txt", "r") as res:
            details["eval_results"] = dict()
            for r in res.readlines():
                k, v = r.split(" = ")
                details["eval_results"][k] = v
        json.dump(details, f, indent=4)


def get_golds_and_predictions(model):
    gold_tags = []
    pred_tags = []

    with jsonlines.open(f"{args.dataset_path}/{DEV}.jsonl", "r") as f:
        for idx, line in enumerate(f):
            words = [w[0] for w in line["sent_items"]]
            tokens = [w[1] for w in line["sent_items"]]
            predictions, raw_outputs = model.predict([" ".join(words)])
            if len(tokens) != len(predictions[0]):
                predicted_words = [list(x.keys())[0] for x in predictions[0]]
                print(f"Skipped snetnece {idx}.Original sentence:\n{' '.join(words)}\nPredicted Sentence:\n{' '.join(predicted_words)}")
                continue
            gold_tags.extend(tokens)
            pred_tags.extend([list(x.values())[0] for x in predictions[0]])

    return gold_tags, pred_tags


def get_span_recall(golds, preds, positive_label, negative_label):
    df = pd.DataFrame(zip(golds,preds), columns=["gold", "pred"])
    
    df["match"] = "_"
    tag_hits(df, positive_label, negative_label)
    true_positive_spans = df[df["match"] == "tp"]["gold"].count()
    total_gold_positives = df[df["gold"] == f"B-{positive_label}"]["gold"].count()
    return true_positive_spans, total_gold_positives


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
                    df.at[i, 'match'] = "fn"
                else:
                    df.at[i, 'match'] = "tp"
                for x in span:
                    df.at[x, 'match'] = "--"


def get_confusion_matrix(gold_tags, pred_tags, labels):
    matrix = confusion_matrix(gold_tags, pred_tags, labels=labels, normalize="true")
    return np.round(matrix, 3)


def main():
    best_model_dir = f"./experiments/{args.wandb_project}-{args.experiment_suffix}/best_model"
    model = NERModel(
        "roberta", best_model_dir
    )
    golds, preds = get_golds_and_predictions(model)
    matrix = get_confusion_matrix(golds, preds, LABELS)
    cls_report = classification_report(golds, preds, labels=LABELS)
    parsed_report = {x.strip()[0:5]:x.strip()[5:].split() for x in cls_report.split("\n")[2:] if not x.startswith("O")}
    true_positive_spans, total_gold_positives = get_span_recall(golds, preds, args.target_tag, args.superclass_tag)
    print(f"confusion matrix:\n{matrix}")
    print(f"span recall:\n{true_positive_spans}\n{total_gold_positives}")
    print(f"span recall:\n{true_positive_spans / total_gold_positives}")
    print(f"classification report:\n{cls_report}")
    print(f"parsed classification report:\n{parsed_report}")
    document_results("./experiments", f"{args.wandb_project}-{args.experiment_suffix}", best_model_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--version_name', help='', default="all_without_person")
    parser.add_argument('--suffix', help='', default="_unique")
    
    parser.add_argument('--target_tag', help='', default="MUS")
    parser.add_argument('--superclass_tag', help='', default="PER")
    parser.add_argument('--wandb_project', help='', default="only_hearst_uniques")
    parser.add_argument('--experiment_suffix', help='', default="manual")
    parser.add_argument('--dataset_path', help='', default="./data/musicians_dataset")

    args = parser.parse_args()
    
    if args.experiment_suffix == "manual":
        DEV = f"dev_converted"
    else:
        DEV = f"split_dev_{args.version_name}"
    if args.superclass_tag:
        LABELS = [f"B-{args.target_tag}",
                  f"B-{args.superclass_tag}",
                  f"I-{args.target_tag}",
                  f"I-{args.superclass_tag}",
                  "O"]
    else:
        LABELS = [f"B-{args.target_tag}",
                  f"I-{args.target_tag}",
                  "O"]        
    main()