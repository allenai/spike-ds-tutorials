import sklearn.metrics as slmet
import seqeval.metrics as sqmet
import matplotlib.pyplot as plt
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


def predict_sentence(model, idx, sentence):
    words = [w[0] for w in sentence["sent_items"]]
    tokens = [w[1] for w in sentence["sent_items"]]
    predictions, raw_outputs = model.predict([" ".join(words)])
    preds = predictions[0]
    if len(tokens) != len(preds):
        predicted_words = [list(x.keys())[0] for x in preds]
        print(f"Skipped snetnece {idx}.Original sentence:\n{' '.join(words)} Predicted Sentence:\n{' '.join(predicted_words)}")
        return None, None   
    return tokens, preds
        
        
def get_golds_and_predictions(model):
    gold_tags = []
    pred_tags = []

    with jsonlines.open(f"./data/{args.dataset}/{eval_set}.jsonl", "r") as f:
        for idx, line in enumerate(f):
            golds, preds = predict_sentence(model, idx, line)
            if golds and preds:
                gold_tags.append(golds)
                pred_tags.append([list(x.values())[0] for x in preds])
    return gold_tags, pred_tags


def get_train_captures():
    train_captures = set()
    with jsonlines.open(f"./data/{args.dataset}/{TRAIN}.jsonl", "r") as f:
        for line in f:
            capture = ""
            for item in line['sent_items']:
                if item[1] == "B-SCHOOL":
                    capture = item[0]
                elif item[1] == "I-SCHOOL":
                    capture += " " + item[0]
                elif capture != "":
                    train_captures.add(capture)
                    capture = ""
    return train_captures


def get_complements_of_train(model, eval_set):
    """
    This function skips sentences if they have an entity that appeared in the train set.
    """
    gold_tags = []
    pred_tags = []
    
    train_captures = get_train_captures()
    with jsonlines.open(f"./data/{args.dataset}/{eval_set}.jsonl", "r") as f:
        for idx, line in enumerate(f):
            capture = ""
            found = False
            for item in line['sent_items']:
                if item[1] == "B-SCHOOL":
                    capture = item[0]
                elif item[1] == "I-SCHOOL":
                    capture += " " + item[0]
                elif capture != "":
                    if capture in train_captures:
                        found = True
                    capture = ""
            if not found:
                golds, preds = predict_sentence(model, idx, line)
                if golds and preds:
                    gold_tags.append(golds)
                    pred_tags.append([list(x.values())[0] for x in preds])
    return gold_tags, pred_tags    


def get_span_recall(golds, preds, positive_label, negative_label="O"):
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
    print(labels)
    matrix = slmet.confusion_matrix(gold_tags, pred_tags, labels=labels, normalize="true")
    return np.round(matrix, 3)


def flatten(nested_list):
    return [item for sublist in nested_list for item in sublist]


def main():
    best_model_dir = f"./experiments/{wandb_project}-{args.experiment}/best_model"
    model = NERModel(
        "roberta", best_model_dir
    )
    if args.eval_on_entire_set:
        golds, preds = get_golds_and_predictions(model)
    else:
        golds, preds = get_complements_of_train(model, eval_set)
    matrix = get_confusion_matrix(flatten(golds), flatten(preds), LABELS)
    cls_report = sqmet.classification_report(golds, preds)
    parsed_report = {x.strip()[0:5]:x.strip()[5:].split() for x in cls_report.split("\n")[2:] if not x.startswith("O")}
    print(f"confusion matrix:\n{matrix}")
    print(f"classification report:\n{cls_report}")
    print(f"parsed classification report:\n{parsed_report}")
    document_results("./experiments", f"{wandb_project}-{args.experiment}", best_model_dir)
    disp = slmet.ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=LABELS)
    disp.plot()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", help="Name of source path. Only used here for deriving names.", type=str,
                        required=True)
    parser.add_argument("-t", "--target_tag", help="label of tagged target entities.", type=str, required=True)
    parser.add_argument('--superclass_tag', help='The canonical NER entity type to which the target tag belongs.',
                        default="")
    parser.add_argument('--experiment', help='If you run several experiments with the same dataset name '
                                             '(e.g. grid-search over hyper-parameters), specify a name for each '
                                             'specific experiment.', default="")
    parser.add_argument('--prefix', help='A prefix that was added to the tagged files. This is helpful for tracking '
                                         'which data were collected for which version.', default="")
    parser.add_argument('--eval_on_test', dest="eval_on_test", action="store_true")
    parser.add_argument('--eval_on_entire_set', dest="eval_on_entire_set", action="store_true")

    args = parser.parse_args()
    wandb_project = f"{args.prefix}{args.dataset}"

    if args.experiment == "manual":
        DEV = f"dev_converted"
    else:
        DEV = f"{args.prefix}split_dev"
    TEST = f"{args.prefix}split_test"
    TRAIN = f"{args.prefix}split_train"
        
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

    eval_set = TEST if args.eval_on_test else DEV
    main()
