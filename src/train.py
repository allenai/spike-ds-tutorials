import logging

from simpletransformers.ner import NERModel
import jsonlines
import argparse
import pandas as pd

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


def set_args():
    model_args = {
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
            "learning_rate": 0.0003,
            "warmup_ratio": 0.1,
            "logging_steps": 1,
            "best_model_dir": best_model_dir
        }
    if args.show_on_wandb:
        model_args.update({"wandb_project": project})
    return model_args


def import_jsons_to_df(filename):
    fp = f"./data/{args.dataset}/{filename}.jsonl"
    tagged_data = []
    with jsonlines.open(fp, 'r') as f:
        for line in f:
            for token in line["sent_items"]:
                tagged_data.append([line["id"], token[0], token[1]])
    return pd.DataFrame(tagged_data, columns=["sentence_id", "words", "labels"])


def configure_model(args, model_type="roberta", model_name="roberta-base"):
    model = NERModel(
        model_type,
        model_name,
        args=args
    )
    return model


def main():
    model_args = set_args()
    dev_set = import_jsons_to_df(DEV)
    train_set = import_jsons_to_df(TRAIN)
    model = configure_model(args=model_args)
    model.train_model(train_set, eval_data=dev_set)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='', default="schools")
    parser.add_argument('--prefix', help='', default="unique_")
    parser.add_argument('--target_tag', help='', default="SCHOOL")
    parser.add_argument('--superclass_tag', help='', default="ORG")
    parser.add_argument('--batch_size', help='', default="64")
    parser.add_argument('--epochs', help='', default="3")
    parser.add_argument('--experiment', help='', default="")
    parser.add_argument('--show_on_wandb', help="",
                        dest="show_on_wandb", action="store_true")

    args = parser.parse_args()
    project = f"{args.prefix}{args.dataset}"
    BATCH_SIZE = int(args.batch_size)
    EPOCHS = int(args.epochs)
    
    if args.experiment == "manual":
        DEV, TRAIN = f"dev_converted", f"{args.prefix}split_train"
    else:
        DEV, TRAIN = f"{args.prefix}split_dev", f"{args.prefix}split_train"
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
    best_model_dir = f"./experiments/{project}-{args.experiment}/best_model"
    main()
